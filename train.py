import argparse
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from LBCNN_con import BPVNet
from dataloader import TrainData, TestData
from thop import profile


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epoch", type=int, default=50, help="epoch to start training from")
    parser.add_argument("--n_epochs", type=int, default=70, help="number of epochs of training")
    parser.add_argument("--num_class", type=int, default=532, help="epoch to start training from")
    parser.add_argument("--img_num", type=int, default=1, help="img_num of dataset")
    parser.add_argument("--dataset_name", type=str, default="CUMT", help="name of the dataset")
    parser.add_argument("--batch_size", type=int, default=16, help="size of the batches")
    parser.add_argument("--lr", type=float, default=0.0001, help="adam: learning rate")
    parser.add_argument("--checkpoint_interval", type=int, default=2, help="interval between saving model checkpoints")
    parser.add_argument("--lambda1", type=float, default=1, help="identity loss weight")
    parser.add_argument("--lambda2", type=float, default=1, help="identity loss weight")
    opt = parser.parse_args()
    print(opt)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    model = BPVNet(num_class=opt.num_class)
    # input1 = torch.randn(1,3,128,128)
    # input2 = torch.randn(1, 3, 128, 128)
    # input3 = torch.randn(1,3,128,128)
    # input4 = torch.randn(1, 3, 128, 128)
    # input5 = torch.randn(1,3,128,128)
    # flops,params = profile(model,inputs=(input1,input2,input3,input4,input5))
    # print('Params='+str(params/1000**2)+'M')

    if opt.epoch != 0:
        weights_path = "../Model_{}/path_{}/{}_{}_BPVNet.pth".format(opt.dataset_name, opt.dataset_name,
                                                                     opt.dataset_name, opt.epoch)
        model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    loss_1 = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=opt.lr, weight_decay=1e-5)
    # optimizer = optim.SGD(model.parameters(),lr = opt.lr,momentum=0.9,weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer,25,0.1)
    data_train = TrainData("./data/train/palmvein_augmented/",
                           "./data/train/palmprint_augmented/",
                           "./data/train/print_augmented/",
                           "./data/train/knuckle_augmented/",
                           "./data/train/fingervein_augmented/"
                           )
    # data_train = TrainData("../../hand-multi-dataset/palmvein_train/",
    #                      "../../hand-multi-dataset/palmprint_train/",
    #                      "../../hand-multi-dataset/print_train/",
    #                      "../../hand-multi-dataset/knuckle_train/",
    #                      "../../hand-multi-dataset/fingervein_train/")
    data_loader = torch.utils.data.DataLoader(
        data_train,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0
    )
    train_steps = len(data_loader) * opt.batch_size

    data_test = TestData("./data/test/palmvein/",
                           "./data/test/palmprint/",
                           "./data/test/print/",
                           "./data/test/print/",
                           "./data/test/fingervein/", opt.img_num)
    # data_test = TestData("../../hand-multi-dataset/palmvein_test/",
    #                      "../../hand-multi-dataset/palmprint_test/",
    #                      "../../hand-multi-dataset/print_test/",
    #                      "../../hand-multi-dataset/knuckle_test/",
    #                      "../../hand-multi-dataset/fingervein_test/", opt.img_num)

    test_loader = torch.utils.data.DataLoader(
        data_test,
        batch_size=opt.batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=0,
    )
    best_acc = 0

    for epoch in range(opt.epoch, opt.n_epochs):
        save_path = '../Model_{}/path_{}/{}_{}_{}Net.pth'.format(opt.dataset_name, opt.dataset_name, opt.dataset_name,
                                                                 epoch + 1, 'BPV')
        os.makedirs('../Model_{}/path_{}'.format(opt.dataset_name, opt.dataset_name), exist_ok=True)
        model.train()
        acc = 0.0
        running_loss = 0.0
        train_bar = tqdm(data_loader, file=sys.stdout)
        for step, data in enumerate(train_bar):
            img1, img2, img3, img4, img5, person_name = data
            optimizer.zero_grad()
            label = [int(_) - 1 for _ in person_name]
            label = torch.tensor(label).to(device)
            x = model(img1.to(device), img2.to(device), img3.to(device),
                                                              img4.to(device), img5.to(device))

            loss0 = loss_1(x, label.to(device))
            #
            # c_1 = torch.nn.functional.softmax(trust1, dim=1)
            # c_2 = torch.nn.functional.softmax(trust2, dim=1)
            # c_3 = torch.nn.functional.softmax(trust1, dim=1)
            # c_4 = torch.nn.functional.softmax(trust2, dim=1)
            # c_5 = torch.nn.functional.softmax(trust1, dim=1)
            #
            # c_1 = torch.squeeze(c_1).to(device)
            # c_2 = torch.squeeze(c_2).to(device)
            # c_3 = torch.squeeze(c_3).to(device)
            # c_4 = torch.squeeze(c_4).to(device)
            # c_5 = torch.squeeze(c_5).to(device)

            # person_labels = label.cpu().numpy()

            # y_label = torch.LongTensor(np.array([person_labels])).cuda()  # gather函数中的index参数类型必须为LongTensor
            # p_m1 = c_1.gather(1, y_label.view(-1, 1))
            # p_m2 = c_2.gather(1, y_label.view(-1, 1))
            # p_m3 = c_3.gather(1, y_label.view(-1, 1))
            # p_m4 = c_4.gather(1, y_label.view(-1, 1))
            # p_m5 = c_5.gather(1, y_label.view(-1, 1))
            #
            # c1 = torch.nn.functional.softmax(trust1, dim=1)
            # c2 = torch.nn.functional.softmax(trust2, dim=1)
            # c3 = torch.nn.functional.softmax(trust3, dim=1)
            # c4 = torch.nn.functional.softmax(trust4, dim=1)
            # c5 = torch.nn.functional.softmax(trust5, dim=1)
            #
            # p_1 = torch.max(c1, dim=1)[0]
            # p_1 = torch.squeeze(p_1).unsqueeze(1)
            # p_2 = torch.max(c2, dim=1)[0]
            # p_2 = torch.squeeze(p_2).unsqueeze(1)
            # p_3 = torch.max(c3, dim=1)[0]
            # p_3 = torch.squeeze(p_3).unsqueeze(1)
            # p_4 = torch.max(c4, dim=1)[0]
            # p_4 = torch.squeeze(p_4).unsqueeze(1)
            # p_5 = torch.max(c5, dim=1)[0]
            # p_5 = torch.squeeze(p_5).unsqueeze(1)
            #
            # loss2_1 = loss_1(trust1, label.to(device))
            # loss2_2 = loss_1(trust2, label.to(device))
            # loss2_3 = loss_1(trust3, label.to(device))
            # loss2_4 = loss_1(trust4, label.to(device))
            # loss2_5 = loss_1(trust5, label.to(device))
            #
            # loss2_6 = loss_2(p_1, p_m1)
            # loss2_7 = loss_2(p_2, p_m2)
            # loss2_8 = loss_2(p_3, p_m3)
            # loss2_9 = loss_2(p_4, p_m4)
            # loss2_10 = loss_2(p_5, p_m5)
            #
            # loss2 = loss2_1 + loss2_2 + loss2_3 + loss2_4 + loss2_5 + loss2_6 + loss2_7 + loss2_8 + loss2_9 + loss2_10
            # ## 结束
            #
            loss = loss0
            predict = torch.max(x, dim=1)[1]
            acc += torch.eq(predict, label.to(device)).sum().item()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                     opt.n_epochs,
                                                                    running_loss / (step + 1))
        accurate = acc / train_steps
        print('num:{},train_accuracy:{:.4f},acc:{}'.format(train_steps, accurate, acc))
        scheduler.step()
        # test
        if (epoch + 1) >= 5:
            model.eval()
            acc_f = 0.0
            test_bar = tqdm(test_loader, file=sys.stdout)
            with torch.no_grad():
                for step, data in enumerate(test_bar):
                    img1, img2, img3, img4, img5, person_name = data
                    x = model(img1.to(device), img2.to(device), img3.to(device),
                                                                      img4.to(device), img5.to(device))
                    label = [int(_) - 1 for _ in person_name]
                    label = torch.tensor(label).to(device)
                    predict_f = torch.max(x,dim=1)[1]
                    acc_f += torch.eq(predict_f, label.to(device)).sum().item()
                    test_steps = len(test_loader) * opt.batch_size
                    accurate_f = acc_f / test_steps

                print(
                    "num:{},accuracy_f:{:.4f}".format(test_steps, accurate_f))
                if best_acc < accurate_f:
                    best_acc = accurate_f
                    best_batch = epoch + 1
                    save_path = '../Model_{}/path_{}/bestBPVNet.pth'.format(opt.dataset_name, opt.dataset_name)
                    torch.save(model.state_dict(), save_path)

                print("best_acc = ", best_acc)
                print("best_batch=", best_batch)

        if (epoch + 1) % opt.checkpoint_interval == 0 and (epoch + 1) >= 15:
            torch.save(model.state_dict(), save_path)
        if (accurate == 1.00 and (epoch + 1) >= 15):
            torch.save(model.state_dict(), save_path)


if __name__ == '__main__':
    main()
