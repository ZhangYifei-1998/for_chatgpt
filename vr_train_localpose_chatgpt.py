
import time
import torch
from torch import nn
from torch import optim
import torch.nn.utils as utils


import numpy as np
import os
import sys
import math
from losses.vr_PoseLoss import LocalPoseLoss
from losses.vr_PoseLoss import PoseLoss2D
from dataset.vr_posedataset import PoseDataSet
from OdoNets import FcnOdoNetV4
from OdoNets import TransAm
import matplotlib.pyplot as plt
np.seterr(invalid='ignore')

import warnings
warnings.filterwarnings('ignore')

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

def train_conv_odo_imu_net(train_dataset, checkpoint_path):
    batch_size = 1
    learning_rate = 0.005
    epochs = 1000
    odo_frames = 40
    dataset_list = []
    for dir in os.listdir(train_dataset):
        dataset = os.path.join(train_dataset, dir) + "/0000/ODO"
        dataset_list.append(dataset)
    dataset = PoseDataSet(dataset_list, odo_frames)
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=10, shuffle=True, pin_memory=True)
    print("load dataset finished")
    odonet = FcnOdoNetV4(18 * odo_frames, 1024, 521, 256, 128, 64, 6)
    odonet = odonet.to(device)
    odonet.train()
    optimizer = optim.SGD(odonet.parameters(), lr=learning_rate)
    pose_loss = PoseLoss2D(0.004)
    print('Start training:   .......................................................')
    torch.autograd.set_detect_anomaly(True)
    loss_list = []
    for epoch in range(epochs):
        epoch_loss = []
        rot_error = []
        trans_error = []
        for batch_idx, (raw_odo, gt_pose) in enumerate(train_loader):

            raw_odo = raw_odo.to(device).to(torch.float32)
            raw_odo = raw_odo.view(raw_odo.size(0), -1)
            gt_pose = gt_pose.to(device).to(torch.float32)

            pred_pose_euler = odonet.forward(raw_odo)
            loss = pose_loss(gt_pose, pred_pose_euler)

            epoch_loss.append(loss.data.item())
            rot_error.append(pose_loss.rot_error.data.item())
            trans_error.append(pose_loss.trans_error.data.item())

            optimizer.zero_grad()
            loss.backward()
            max_norm = 1.0
            grad_norm = utils.clip_grad_norm_(odonet.parameters(), max_norm)
            optimizer.step()

        print('epoch: {}, loss: {:.10}, rot_err: {:.10}, trans_err: {:.10}'\
        .format(epoch, np.mean(np.array(epoch_loss)),\
                       np.mean(np.array(rot_error)), \
                       np.mean(np.array(trans_error))))

        torch.save({
        'epoch': epoch,
        'model_state_dict': odonet.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss}, checkpoint_path + str(epoch).zfill(4)+".pth")

        loss_list.append(loss)


    print(type(loss_list))
    plt.figure()
    plt.plot(range(1, len(loss_list)+1),
         loss_list, 'r', label='Training loss')

    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()#


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    #训练集目录
    train_dataset = 'C:/Users/75901/Desktop/transformer/code/IdeaProjects/transformer/data_vr_move/'

    #网络参数保存位置
    checkpoint_path = 'C:/Users/75901/Desktop/transformer/code/IdeaProjects/transformer/data_vr_move/'

    #训练网络
    train_conv_odo_imu_net(train_dataset, checkpoint_path)
