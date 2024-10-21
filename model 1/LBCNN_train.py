import argparse

import torch
import torch.nn as nn

from model import *


parser = argparse.ArgumentParser()
parser.add_argument("--num_class", type=int, default=290, help="epoch to start training from")
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
        self.net4 = MobleNetV1()
        self.net5 = MobleNetV1()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cbamblock = cbamblock(channel=512)
        ##
        self.pre_output_size = 512
        self.embracenet = EmbraceNet(device=self.device, input_size_list=[self.pre_output_size,self.pre_output_size],
                                     embracement_size=1024)
        self.fc1 = nn.Linear(512, num_class)
        self.fc2 = nn.Linear(512, num_class)
        self.fc3 = nn.Linear(512, num_class)
        self.fc4 = nn.Linear(512, num_class)
        self.fc5 = nn.Linear(512, num_class)
        self.fc = nn.Linear(in_features=1024, out_features=num_class)
        self.drop = nn.Dropout(0.3)

    def forward(self, x1, x2, x3, x4, x5):
        x1 = self.net1(x1)
        x1 = self.cbamblock(x1)
        x1 = self.pool(x1)

        x2 = self.net2(x2)
        x2 = self.cbamblock(x2)
        x2 = self.pool(x2)

        x3 = self.net3(x3)
        x3 = self.cbamblock(x3)
        x3 = self.pool(x3)

        x4 = self.net4(x4)
        x4 = self.cbamblock(x4)
        x4 = self.pool(x4)

        x5 = self.net5(x5)
        x5 = self.cbamblock(x5)
        x5 = self.pool(x5)
        B, _, _, _ = x1.shape
        ## 后加
        x1_a = x1.reshape(B, -1)
        x2_a = x2.reshape(B, -1)
        x3_a = x3.reshape(B, -1)
        x4_a = x4.reshape(B, -1)
        x5_a = x5.reshape(B, -1)

        trust1 = self.fc1(x1_a)
        trust2 = self.fc2(x2_a)
        trust3 = self.fc3(x3_a)
        trust4 = self.fc4(x4_a)
        trust5 = self.fc5(x5_a)

        w1 = torch.sigmoid(trust1)
        w2 = torch.sigmoid(trust2)
        w3 = torch.sigmoid(trust3)
        w4 = torch.sigmoid(trust4)
        w5 = torch.sigmoid(trust5)

        w1, _ = torch.max(w1, dim=1, keepdim=True)
        w2, _ = torch.max(w2, dim=1, keepdim=True)
        w3, _ = torch.max(w3, dim=1, keepdim=True)
        w4, _ = torch.max(w4, dim=1, keepdim=True)
        w5, _ = torch.max(w5, dim=1, keepdim=True)

        w_1 = torch.mean(w1).item()
        w_2 = torch.mean(w2).item()
        w_3 = torch.mean(w3).item()
        w_4 = torch.mean(w4).item()
        w_5 = torch.mean(w5).item()
        w_ = [(w_1,w1,x1),(w_2,w2,x2),(w_3,w3,x3),(w_4,w4,x4),(w_5,w5,x5)]
        w_ = sorted(w_, key=lambda item: item[0])
        (w_1,w1,x1),(w_2,w2,x2) = w_[-1], w_[-2]
        w = torch.sum(w1 + w2 , dim=1, keepdim=True)
        w11 = torch.div(w1, w)
        w12 = torch.div(w2, w)
        x1 = x1.reshape(B, -1)
        x2 = x2.reshape(B, -1)

        selection_probabilities = torch.cat([w11,w12], dim=1)
        # 结束
        x_embrace = self.embracenet([x1, x2],
                                    selection_probabilities=selection_probabilities)
        x = self.drop(x_embrace)
        x = self.fc(x_embrace)

        ## 结束
        return x, trust1, trust2,trust3,trust4,trust5
