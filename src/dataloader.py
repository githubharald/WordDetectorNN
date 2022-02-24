from collections import namedtuple

import cv2
import numpy as np
import torch

from aabb import AABB
from coding import encode
from utils import compute_scale_down, prob_true

DataLoaderItem = namedtuple('DataLoaderItem', 'batch_imgs,batch_gt_maps,batch_aabbs')


class DataLoaderIAM:
    """loader for IAM dataset"""

    def __init__(self, dataset, batch_size, input_size, output_size):
        self.dataset = dataset
        self.batch_size = batch_size
        self.input_size = input_size
        self.output_size = output_size
        self.scale_down = compute_scale_down(input_size, output_size)
        self.shuffled_indices = np.arange(len(self.dataset))
        self.curr_idx = 0
        self.is_random = False

    def __getitem__(self, item):
        batch_imgs = []
        batch_gt_maps = []
        batch_aabbs = []
        for b in range(self.batch_size):
            if self.is_random:
                shuffled_idx = self.shuffled_indices[item * self.batch_size + b]
            else:
                shuffled_idx = item * self.batch_size + b

            img, aabbs = self.dataset[shuffled_idx]

            if self.is_random:
                # geometric data augmentation (image [0..255] and gt)
                if prob_true(0.75):
                    # random scale
                    fx = np.random.uniform(0.5, 1.5)
                    fy = np.random.uniform(0.5, 1.5)

                    # random position around center
                    txc = self.input_size[1] * (1 - fx) / 2
                    tyc = self.input_size[0] * (1 - fy) / 2
                    freedom_x = self.input_size[1] // 10
                    freedom_y = self.input_size[0] // 10
                    tx = txc + np.random.randint(-freedom_x, freedom_x)
                    ty = tyc + np.random.randint(-freedom_y, freedom_y)

                    # map image into target image
                    M = np.float32([[fx, 0, tx], [0, fy, ty]])
                    white_bg = np.ones(self.input_size, np.uint8) * 255
                    img = cv2.warpAffine(img, M, dsize=self.input_size[::-1], dst=white_bg,
                                         borderMode=cv2.BORDER_TRANSPARENT)

                    # apply the same transformations to gt, and clip/remove aabbs outside of target image
                    aabb_clip = AABB(0, img.shape[1], 0, img.shape[0])
                    aabbs = [aabb.scale(fx, fy).translate(tx, ty).clip(aabb_clip) for aabb in aabbs]
                    aabbs = [aabb for aabb in aabbs if aabb.area() > 0]

                # photometric data augmentation (image [-0.5..0.5] only)
                img = (img / 255 - 0.5)
                if prob_true(0.25):  # random distractors (lines)
                    num_lines = np.random.randint(1, 20)
                    for _ in range(num_lines):
                        rand_pt = lambda: (np.random.randint(0, img.shape[1]), np.random.randint(0, img.shape[0]))
                        color = np.random.triangular(-0.5, 0, 0.5)
                        thickness = np.random.randint(1, 3)
                        cv2.line(img, rand_pt(), rand_pt(), color, thickness)
                if prob_true(0.75):  # random contrast
                    img = (img - img.min()) / (img.max() - img.min()) - 0.5  # stretch
                    img = img * np.random.triangular(0.1, 0.9, 1)  # reduce contrast
                if prob_true(0.25):  # random noise
                    img = img + np.random.uniform(-0.1, 0.1, size=img.shape)
                if prob_true(0.25):  # change thickness of text
                    img = cv2.erode(img, np.ones((3, 3)))
                if prob_true(0.25):  # change thickness of text
                    img = cv2.dilate(img, np.ones((3, 3)))
                if prob_true(0.25):  # invert image
                    img = 0.5 - img

            else:
                img = (img / 255 - 0.5)

            gt_map = encode(self.output_size, aabbs, self.scale_down)

            batch_imgs.append(img[None, ...].astype(np.float32))
            batch_gt_maps.append(gt_map)
            batch_aabbs.append(aabbs)

        batch_imgs = np.stack(batch_imgs, axis=0)
        batch_gt_maps = np.stack(batch_gt_maps, axis=0)

        batch_imgs = torch.from_numpy(batch_imgs).to('cuda')
        batch_gt_maps = torch.from_numpy(batch_gt_maps.astype(np.float32)).to('cuda')

        return DataLoaderItem(batch_imgs, batch_gt_maps, batch_aabbs)

    def reset(self):
        self.curr_idx = 0

    def random(self, enable=True):
        np.random.shuffle(self.shuffled_indices)
        self.is_random = enable

    def __len__(self):
        return len(self.dataset) // self.batch_size


class DataLoaderImgFile:
    """loader which simply goes through all jpg files of a directory"""

    def __init__(self, root_dir, input_size, device, max_side_len=1024):
        self.fn_imgs = root_dir.files('*.jpg')
        self.input_size = input_size
        self.device = device
        self.max_side_len = max_side_len

    def ceil32(self, val):
        if val % 32 == 0:
            return val
        val = (val // 32 + 1) * 32
        return val

    def __getitem__(self, item):
        orig = cv2.imread(self.fn_imgs[item], cv2.IMREAD_GRAYSCALE)

        f = min(self.max_side_len / orig.shape[0], self.max_side_len / orig.shape[1])
        if f < 1:
            orig = cv2.resize(orig, dsize=None, fx=f, fy=f)
        img = np.ones((self.ceil32(orig.shape[0]), self.ceil32(orig.shape[1])), np.uint8) * 255
        img[:orig.shape[0], :orig.shape[1]] = orig

        img = (img / 255 - 0.5).astype(np.float32)
        imgs = img[None, None, ...]
        imgs = torch.from_numpy(imgs).to(self.device)
        return DataLoaderItem(imgs, None, None)

    def get_scale_factor(self, item):
        img = cv2.imread(self.fn_imgs[item], cv2.IMREAD_GRAYSCALE)
        f = min(self.max_side_len / img.shape[0], self.max_side_len / img.shape[1])
        return f if f < 1 else 1

    def get_original_img(self, item):
        img = cv2.imread(self.fn_imgs[item], cv2.IMREAD_GRAYSCALE)
        img = (img / 255 - 0.5).astype(np.float32)
        return img

    def __len__(self):
        return len(self.fn_imgs)
