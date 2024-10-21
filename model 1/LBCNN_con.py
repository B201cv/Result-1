import argparse

import torch
import torch.nn as nn

from model import *


parser = argparse.ArgumentParser()
parser.add_argument("--img_height", type=int, default=128, help="size of image height")
parser.add_argument("--img_width", type=int, default=128, help="size of image width")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--dropout_prob", type=int, default=0.5) # CUMT 0.4
opt = parser.parse_args()
print(opt)

input_shape = (opt.channels, opt.img_height, opt.img_width)


class BPVNet(nn.Module):
    def __init__(self, num_class):
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        super(BPVNet, self).__init__()
        self.net1 = MobleNetV1()
        self.net2 = MobleNetV1()
        self.net3 = MobleNetV1()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        ##
        self.fc = nn.Sequential(
            nn.Linear(in_features=512*3, out_features=num_class)
        )

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.net1(x1)
        x1 = self.pool(x1)

        x2 = self.net2(x2)
        x2 = self.pool(x2)

        x3 = self.net3(x3)
        x3 = self.pool(x3)

        B, _, _, _ = x1.shape
        ## 后加
        x1_a = x1.reshape(B, -1)
        x2_a = x2.reshape(B, -1)
        x3_a = x3.reshape(B, -1)

        x_embrace = torch.cat([x1_a,x2_a,x3_a],dim=1)
        x = self.fc(x_embrace)

        ## 结束
        return x
