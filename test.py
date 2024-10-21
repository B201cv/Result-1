import os
import sys
import numpy as np
import torch
from tqdm import tqdm
from LBCNN_con import BPVNet
from dataloader import TestData,TrainData
from thop import profile
import argparse
import time


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, default="CUMT", help="name of the dataset")
    parser.add_argument("--name", type=str, default="VP", help="PV,PV1,VP,VP1")
    parser.add_argument("--img_num", type=int, default=10, help="img_num of dataset")
    parser.add_argument("--num_class", type=int, default=532, help="epoch to start training from")
    parser.add_argument("--batch_size", type=int, default=8, help="size of the batches")
    opt = parser.parse_args()

    batch_size = 8
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = BPVNet(num_class=opt.num_class)
    # input1 = torch.randn(1, 3, 224, 224)  # (1, 3, 128, 128)分别表示输入的图片数量，图像通道处，图像高度和宽度
    # input2 = torch.randn(1, 3, 224, 224)  # (1, 3, 128, 128)分别表示输入的图片数量，图像通道处，图像高度和宽度
    # input3 = torch.randn(1, 3, 224, 224)  # (1, 3, 128, 128)分别表示输入的图片数量，图像通道处，图像高度和宽度
    # input4 = torch.randn(1, 3, 224, 224)  # (1, 3, 128, 128)分别表示输入的图片数量，图像通道处，图像高度和宽度
    # input5 = torch.randn(1, 3, 224, 224)  # (1, 3, 128, 128)分别表示输入的图片数量，图像通道处，图像高度和宽度
    #
    # ## 统计flops, params
    # flops, params = profile(model, inputs=(input1, input2,input3, input4,input5))
    #
    # print('FLOPs = ' + str(flops / 1000 ** 3) + 'G')
    # print('Params = ' + str(params / 1000 ** 2) + 'M')
    # total = sum([param.nelement() for param in model.parameters()])
    # print("Number of parameters: %.2fM" % (total / 1e6))

    weights_path = "../Model_{}/path_{}/CUMT_50_BPVNet.pth".format(opt.dataset_name,
                                                                  opt.dataset_name,
                                                                  opt.dataset_name)  #{}_99_BPVNet.pth bestBPVNet.pth

    assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)

    model.load_state_dict(torch.load(weights_path , map_location=device))
    model = model.to(device)
    data_test = TestData("../../hand-multi-dataset/palmvein_test/",
                         "../../hand-multi-dataset/palmprint_test/",
                         "../../hand-multi-dataset/print_test/",
                         "../../hand-multi-dataset/knuckle_test/",
                         "../../hand-multi-dataset/fingervein_test/", opt.img_num)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )

    print("data_loader = ", test_loader)
    print("start test......")
    model.eval()
    arr = []
    total_frames = 0
    time_start = time.time()
    for epoch in range(10):
        acc = 0.0
        test_steps = 0
        with torch.no_grad():
            test_bar = tqdm(test_loader, file=sys.stdout)
            for data in test_bar:
                img1, img2, img3, img4, img5, person_name = data
                batch_size = img1.size(0)
                total_frames +=batch_size
                outputs = model(img1.to(device), img2.to(device), img3.to(device),
                                                                      img4.to(device), img5.to(device))
                predict = torch.max(outputs, dim=1)[1]
                label = [int(_) - 1 for _ in person_name]
                label = torch.tensor(label).to(device)
                acc += torch.eq(predict, label.to(device)).sum().item()
                test_steps = len(test_loader) * batch_size
        accurate = acc / test_steps
        end_time = time.time()
        total_time = end_time - time_start
        fps = total_frames/total_time
        fps = float("%.2f"%fps)
        print(f"FPS:{fps}")
        arr.append(accurate)
        print("[epoch %d]" % (epoch + 1))
        print("num:{}, test_accuracy:{:.4f},acc:{}".format(test_steps, accurate, acc))
        ave_accurate = np.mean(arr)
        std = np.std(arr)  # 计算标准差
        print("ave_accurate:{:.4f}, std:{}".format(ave_accurate, std))


if __name__ == "__main__":
    main()
