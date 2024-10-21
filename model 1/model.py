import torch
from torch import nn


# class f_Net(nn.Module):
#     def __init__(self):
#         super(f_Net,self).__init__()
#         self.layer1 = MobleNetV1().conv1
#         self.layer2 = MobleNetV1().conv_dw1
#         self.layer3 = MobleNetV1().conv_dw2
#         self.layer4 = MobleNetV1().conv_dw3
#         self.layer5 = MobleNetV1().conv_dw4
#         self.layer6 = MobleNetV1().conv_dw5
#         self.conv = nn.Sequential(
#             self.layer1,
#             self.layer2,
#             self.layer3,
#             self.layer4,
#             self.layer5,
#             self.layer6,
#         )
#
#     def forward(self, x):
#         x = self.conv(x)
#         return x


class f_Net(nn.Module):
    def __init__(self):
        super(f_Net,self).__init__()
        self.in_channel = 64
        self.layer1 = resnet18().layer1
        self.layer2 = resnet18().layer2
        self.f_before = nn.Sequential(
            nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                      padding=3, bias=False),
            nn.BatchNorm2d(self.in_channel),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
        )

    def forward(self, x):
        x = self.f_before(x)
        x = self.conv(x)
        return x


class GfusionNet(nn.Module):
    def __init__(self):
        super(GfusionNet, self).__init__()
        self.fc1 = nn.Linear(512, 512)
        self.fc2 = nn.Linear(1024, 512)
        self.drop =nn.Dropout(0.5)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.softmax = nn.functional.softmax

    def forward(self,x1,x2):
        x11 = self.relu(self.fc1(x1))
        x22 = self.relu(self.fc1(x2))
        # x11 = self.softmax(x11, dim=1) * x1
        # x22 = self.softmax(x22, dim=1) * x2
        x3 = torch.cat((x11, x22),dim=1)
        x3 = self.fc2(x3)
        w1 = self.sigmoid(x3)
        w2 = 1-w1
        x3 = x11*w1+x22*w2
        return x3


class LfusionNet(nn.Module):
    def __init__(self,num):
        super(LfusionNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(in_channels=2*num, out_channels=num, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(in_channels=num, out_channels=num, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self,x1,x2):
        x11 = self.relu(self.conv1(x1))+x1
        x22 = self.relu(self.conv2(x2))+x2
        x3 = torch.cat((x11,x22),dim=1)
        w1 = self.sigmoid(self.conv3(x3))
        w2 = 1-w1
        x3 = x11*w1+x22*w2
        return x3


class choiceNet(nn.Module):
    def __init__(self):
        super(choiceNet, self).__init__()
        self.fc = nn.Linear(512,290)

    def forward(self,x1,x2):
        x_f = nn.functional.softmax(x1, dim=1)
        x_p = nn.functional.softmax(x2, dim=1)
        # x_max_prob = torch.stack([x_f, x_p], dim=1)
        # x_max, max_indices = torch.max(x_max_prob, dim=1)
        x_p_max_prob, x_p_max = torch.max(x_p, dim=1)
        x_f_max_prob, x_f_max = torch.max(x_f, dim=1)
        x_max_prob = torch.stack([x_f_max_prob,x_p_max_prob], dim=1)
        max_indices = torch.argmax(x_max_prob, dim=1)
        x_max = torch.zeros_like(x_p_max)
        for i in range(len(max_indices)):
            if max_indices[i] == 0:
                x_max[i] = x_f_max[i]
            else:
                x_max[i] = x_p_max[i]
        return x_max


class OneNet(nn.Module):
    def __init__(self):
        super(OneNet, self).__init__()
        self.conv = Vgg16()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.cbamblock = cbamblock(channel=512)

    def forward(self, x):
        x = self.conv(x)
        x = self.cbamblock(x)
        x = self.pool(x)
        return x


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64),  # Batch Normalization强行将数据拉回到均值为0，方差为1的正太分布上，一方面使得数据分布一致，另一方面避免梯度消失。
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),  # 224 * 224 * 64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(kernel_size=2, stride=2)  # 112 * 112 * 64
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),  # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),  # 112 * 112 * 128
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 56 * 56 * 128
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),  # 56 * 56 * 256
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 28 * 28 * 256
        )

        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 28 * 28 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 14 * 14 * 512
        )

        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),  # 14 * 14 * 512
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.MaxPool2d(2, 2)  # 7 * 7 * 512
        )

        self.conv = nn.Sequential(
            self.layer1,
            self.layer2,
            self.layer3,
            self.layer4,
            self.layer5
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class MobleNetV1(nn.Module):
    def __init__(self):
        super(MobleNetV1, self).__init__()
        self.conv1 = self._conv_st(3, 32, 2)
        self.conv_dw1 = self._conv_dw(32, 64, 1)
        self.conv_dw2 = self._conv_dw(64, 128, 2)
        self.conv_dw4 = self._conv_dw(128, 256, 2)
        self.conv_dw6 = self._conv_dw(256, 512, 2)
        self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv_dw1(x)
        x = self.conv_dw2(x)
        x = self.conv_dw4(x)
        x = self.conv_dw6(x)

        return x

    def _conv_x5(self, in_channel, out_channel, blocks):
        layers = []
        for i in range(blocks):
            layers.append(self._conv_dw(in_channel, out_channel, 1))
        return nn.Sequential(*layers)

    def _conv_st(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )

    def _conv_dw(self, in_channels, out_channels, stride):
        return nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=stride, padding=1, groups=in_channels, bias=False),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        )


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x


def resnet18():
    return ResNet(BasicBlock, [2, 2, 2, 2])


# CBMA  通道注意力机制和空间注意力机制的结合
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)  # 平均池化高宽为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)  # 最大池化高宽为1

        # 利用1x1卷积代替全连接
        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 平均池化---》1*1卷积层降维----》激活函数----》卷积层升维
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        # 最大池化---》1*1卷积层降维----》激活函数----》卷积层升维
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out  # 加和操作
        return out
        # return self.sigmoid(out)  # sigmoid激活操作


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = kernel_size // 2
        # 经过一个卷积层，输入维度是2，输出维度是1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()  # sigmoid激活操作

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的平均值  b,1,h,w
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # 在通道的维度上，取所有特征点的最大值  b,1,h,w
        x = torch.cat([avg_out, max_out], dim=1)  # 在第一维度上拼接，变为 b,2,h,w
        x = self.conv1(x)  # 转换为维度，变为 b,1,h,w
        return self.sigmoid(x)  # sigmoid激活操作


class cbamblock(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=7): # kernel_size=7
        super(cbamblock, self).__init__()
        self.channelattention = ChannelAttention(channel, ratio=ratio)
        self.spatialattention = SpatialAttention(kernel_size=kernel_size)

    def forward(self, x):
        x = x * self.channelattention(x)  # 将这个权值乘上原输入特征层
        x = x * self.spatialattention(x)  # 将这个权值乘上原输入特征层
        return x

class EmbraceNet(nn.Module):

    def __init__(self, device, input_size_list, embracement_size=256, bypass_docking=False):

        super(EmbraceNet, self).__init__()

        self.device = device
        self.input_size_list = input_size_list
        self.embracement_size = embracement_size
        self.bypass_docking = bypass_docking

        if (not bypass_docking):
            for i, input_size in enumerate(input_size_list):
                setattr(self, 'docking_%d' % (i), nn.Linear(input_size, embracement_size))

    def forward(self, input_list, availabilities=None, selection_probabilities=None):

        # check input data
        assert len(input_list) == len(self.input_size_list)
        num_modalities = len(input_list)
        batch_size = input_list[0].shape[0]

        # docking layer
        docking_output_list = []
        if (self.bypass_docking):
            docking_output_list = input_list
        else:
            for i, input_data in enumerate(input_list):
                x = getattr(self, 'docking_%d' % (i))(input_data)
                x = nn.functional.relu(x)
                docking_output_list.append(x)

        # check availabilities
        if (availabilities is None):
            availabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        else:
            availabilities = availabilities.float()

        # adjust selection probabilities
        if (selection_probabilities is None):
            selection_probabilities = torch.ones(batch_size, len(input_list), dtype=torch.float, device=self.device)
        selection_probabilities = torch.mul(selection_probabilities, availabilities)

        probability_sum = torch.sum(selection_probabilities, dim=-1, keepdim=True)
        selection_probabilities = torch.div(selection_probabilities, probability_sum)

        # stack docking outputs
        docking_output_stack = torch.stack(docking_output_list,
                                           dim=-1)  # [batch_size, embracement_size, num_modalities]

        # embrace
        modality_indices = torch.multinomial(selection_probabilities, num_samples=self.embracement_size,
                                             replacement=True)  # [batch_size, embracement_size]
        modality_toggles = nn.functional.one_hot(modality_indices,
                                                 num_classes=num_modalities).float()  # [batch_size, embracement_size, num_modalities]

        embracement_output_stack = torch.mul(docking_output_stack, modality_toggles)
        embracement_output = torch.sum(embracement_output_stack, dim=-1)  # [batch_size, embracement_size]

        return embracement_output




