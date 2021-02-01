import pickle
import xml.etree.ElementTree as ET

import cv2
from path import Path

from aabb import AABB


class DatasetIAM:
    """loads the image and ground truth data of the IAM dataset"""

    def __init__(self, root_dir, input_size, output_size, caching=True):

        self.caching = caching
        self.input_size = input_size
        self.output_size = output_size
        self.loaded_img_scale = 0.25
        self.fn_gts = []
        self.fn_imgs = []
        self.img_cache = []
        self.gt_cache = []
        self.num_samples = 0

        fn_cache = root_dir / 'cache.pickle'
        if self.caching and fn_cache.exists():
            self.img_cache, self.gt_cache = pickle.load(open(fn_cache, 'rb'))
            self.num_samples = len(self.img_cache)
            return

        gt_dir = root_dir / 'gt'
        img_dir = root_dir / 'img'
        for fn_gt in sorted(gt_dir.files('*.xml')):
            fn_img = img_dir / fn_gt.stem + '.png'
            if not fn_img.exists():
                continue

            self.fn_imgs.append(fn_img.abspath())
            self.fn_gts.append(fn_gt.abspath())
            self.num_samples += 1

            if self.caching:
                img = cv2.imread(fn_img.abspath(), cv2.IMREAD_GRAYSCALE)
                img = cv2.resize(img, dsize=None, fx=self.loaded_img_scale, fy=self.loaded_img_scale)
                gt = self.parse_gt(fn_gt.abspath())

                img, gt = self.crop(img, gt)
                img, gt = self.adjust_size(img, gt)

                self.img_cache.append(img)
                self.gt_cache.append(gt)

        if self.caching:
            pickle.dump([self.img_cache, self.gt_cache], open(fn_cache, 'wb'))

    def parse_gt(self, fn_gt):
        tree = ET.parse(fn_gt)
        root = tree.getroot()

        aabbs = []  # list of all axis aligned bounding boxes of current sample

        # go over all lines
        for line in root.findall("./handwritten-part/line"):

            # go over all words
            for word in line.findall('./word'):
                xmin, xmax, ymin, ymax = float('inf'), 0, float('inf'), 0
                success = False

                # go over all characters
                for cmp in word.findall('./cmp'):
                    success = True
                    x = float(cmp.attrib['x'])
                    y = float(cmp.attrib['y'])
                    w = float(cmp.attrib['width'])
                    h = float(cmp.attrib['height'])

                    # aabb around all characters is aabb around word
                    xmin = min(xmin, x)
                    xmax = max(xmax, x + w)
                    ymin = min(ymin, y)
                    ymax = max(ymax, y + h)

                if success:
                    aabbs.append(AABB(xmin, xmax, ymin, ymax).scale(self.loaded_img_scale, self.loaded_img_scale))

        return aabbs

    def crop(self, img, gt):
        xmin = min([aabb.xmin for aabb in gt])
        xmax = max([aabb.xmax for aabb in gt])
        ymin = min([aabb.ymin for aabb in gt])
        ymax = max([aabb.ymax for aabb in gt])

        gt_crop = [aabb.translate(-xmin, -ymin) for aabb in gt]
        img_crop = img[int(ymin):int(ymax), int(xmin):int(xmax)]
        return img_crop, gt_crop

    def adjust_size(self, img, gt):
        h, w = img.shape
        fx = self.input_size[1] / w
        fy = self.input_size[0] / h
        gt = [aabb.scale(fx, fy) for aabb in gt]
        img = cv2.resize(img, dsize=self.input_size)
        return img, gt

    def __getitem__(self, idx):

        if self.caching:
            img = self.img_cache[idx]
            gt = self.gt_cache[idx]
        else:
            img = cv2.imread(self.fn_imgs[idx], cv2.IMREAD_GRAYSCALE)
            img = cv2.resize(img, dsize=None, fx=self.loaded_img_scale, fy=self.loaded_img_scale)
            gt = self.parse_gt(self.fn_gts[idx])
            img, gt = self.crop(img, gt)
            img, gt = self.adjust_size(img, gt)

        return img, gt

    def __len__(self):
        return self.num_samples


class DatasetIAMSplit:
    """wrapper which provides a dataset interface for a split of the original dataset"""
    def __init__(self, dataset, start_idx, end_idx):
        assert start_idx >= 0 and end_idx <= len(dataset)

        self.dataset = dataset
        self.start_idx = start_idx
        self.end_idx = end_idx

    def __getitem__(self, idx):
        return self.dataset[self.start_idx + idx]

    def __len__(self):
        return self.end_idx - self.start_idx


if __name__ == '__main__':
    from visualization import visualize
    from coding import encode, decode
    import matplotlib.pyplot as plt

    dataset = DatasetIAM(Path('../data'), (350, 350), (350, 350), caching=False)
    img, gt = dataset[0]
    gt_map = encode(img.shape, gt)
    gt = decode(gt_map)

    plt.imshow(visualize(img / 255 - 0.5, gt))
    plt.show()
