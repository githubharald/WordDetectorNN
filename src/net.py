import torch
import torch.nn.functional as F

from resnet import resnet18
from coding import MapOrdering
from utils import compute_scale_down


class UpscaleAndConcatLayer(torch.nn.Module):
    """
    take small map with cx channels
    upscale to size of large map (s*s)
    concat large map with cy channels and upscaled small map
    apply conv and output map with cz channels
    """

    def __init__(self, cx, cy, cz):
        super(UpscaleAndConcatLayer, self).__init__()
        self.conv = torch.nn.Conv2d(cx + cy, cz, 3, padding=1)

    def forward(self, x, y, s):
        x = F.interpolate(x, s)
        z = torch.cat((x, y), 1)
        z = F.relu(self.conv(z))
        return z


class WordDetectorNet(torch.nn.Module):
    # fixed sizes for training
    input_size = (448, 448)
    output_size = (224, 224)
    scale_down = compute_scale_down(input_size, output_size)

    def __init__(self):
        super(WordDetectorNet, self).__init__()

        self.backbone = resnet18()

        self.up1 = UpscaleAndConcatLayer(512, 256, 256)  # input//16
        self.up2 = UpscaleAndConcatLayer(256, 128, 128)  # input//8
        self.up3 = UpscaleAndConcatLayer(128, 64, 64)  # input//4
        self.up4 = UpscaleAndConcatLayer(64, 64, 32)  # input//2

        self.conv1 = torch.nn.Conv2d(32, MapOrdering.NUM_MAPS, 3, 1, padding=1)

    @staticmethod
    def scale_shape(s, f):
        assert s[0] % f == 0 and s[1] % f == 0
        return s[0] // f, s[1] // f

    def output_activation(self, x, apply_softmax):
        if apply_softmax:
            seg = torch.softmax(x[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1], dim=1)
        else:
            seg = x[:, MapOrdering.SEG_WORD:MapOrdering.SEG_BACKGROUND + 1]
        geo = torch.sigmoid(x[:, MapOrdering.GEO_TOP:]) * self.input_size[0]
        y = torch.cat([seg, geo], dim=1)
        return y

    def forward(self, x, apply_softmax=False):
        # x: BxCxHxW
        # eval backbone with 448px: bb1: 224px, bb2: 112px, bb3: 56px, bb4: 28px, bb5: 14px
        s = x.shape[2:]
        bb5, bb4, bb3, bb2, bb1 = self.backbone(x)

        x = self.up1(bb5, bb4, self.scale_shape(s, 16))
        x = self.up2(x, bb3, self.scale_shape(s, 8))
        x = self.up3(x, bb2, self.scale_shape(s, 4))
        x = self.up4(x, bb1, self.scale_shape(s, 2))
        x = self.conv1(x)

        return self.output_activation(x, apply_softmax)
