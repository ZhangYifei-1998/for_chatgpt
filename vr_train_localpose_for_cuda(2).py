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
import warnings
import torch.cuda.amp as amp

# 实例化 GradScaler
scaler = amp.GradScaler()

warnings.filterwarnings('ignore')
np.seterr(invalid='ignore')

device = torch.device("cuda"if torch.cuda.is_available() else "cpu")

def train_conv_odo_imu_net(train_dataset, checkpoint_path):
    # 训练的参数
    batch_size = 1
    learning_rate = 0.001
    epochs = 100
    odo_frames = 40 #？？？我目前理解，这个就是batch Size，代表每次选40帧原始数据进行训练
    # 读入数据
    dataset_list = []
    for dir in os.listdir(train_dataset):
        dataset = os.path.join(train_dataset, dir) + "/0000/evaluate"
        dataset_list.append(dataset)
    dataset = PoseDataSet(dataset_list, odo_frames)
    #用十个子进程加载数据， 每个epoch重新打乱数据，pin:内存的Tensor转义到GPU的显存会更快一些
    train_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, num_workers=14, shuffle=True, pin_memory=True)
    print("load dataset finished")
    #定义网络 6层 7参 输入10*40维度 输出2维度
    # odonet = FcnOdoNetV4(15 * odo_frames, 1024, 521, 256, 128, 64, 6)
    odonet =TransAm(    feature_size=15 * odo_frames  ,num_layers=1,dropout=0.01  )
    odonet = odonet.to(device)
    # odonet = nn.DataParallel(odonet)
    # model.train()的作用是启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()是随机取一部分网络连接来训练更新参数。
    #model.eval()的作用是不启用 Batch Normalization 和 Dropout。如果模型中有BN层(Batch Normalization）和Dropout，在测试时添加model.eval()。model.eval()是保证BN层能够用全部训练数据的均值和方差，即测试过程中要保证BN层的均值和方差不变。对于Dropout，model.eval()是利用到了所有网络连接，即不进行随机舍弃神经元。训练完train样本后，生成的模型model要用来测试样本。在model(test)之前，需要加上model.eval()，否则的话，有输入数据，即使不训练，它也会改变权值。这是model中含有BN层和Dropout所带来的的性质。

    #定义loss5
    #SGD 是最普通的优化器, 也可以说没有加速效果, 而 Momentum 是 SGD 的改良版, 它加入了动量原则. 后面的 RMSprop 又是 Momentum 的升级版. 而 Adam 又是 RMSprop 的升级版. 不过从这个结果中我们看到, Adam 的效果似乎比 RMSprop 要差一点. 所以说并不是越先进的优化器, 结果越佳. 我们在自己的试验中可以尝试不同的优化器, 找到那个最适合你数据/网络的优化器
    # optimizer = optim.SGD(odonet.parameters(), lr=learning_rate) #SGD比ADAM效果好
    optimizer = torch.optim.AdamW(odonet.parameters(), lr = learning_rate)
    # pose_loss = LocalPoseLoss()
    pose_loss = PoseLoss2D(0.004)
    #训练网络

    print('Start training:   .......................................................')
    torch.autograd.set_detect_anomaly(True)

    loss_list = []
    for epoch in range(epochs):
        odonet.train() #设置训练模式  使用PyTorch进行训练和测试时一定注意要把实例化的model指定train/eval。model.eval()时，框架会自动把BN和Dropout固定住，不会取平均，而是用训练好的值
        epoch_loss = []
        rot_error = []
        trans_error = []

        # if(epoch == 130):
        #     optimizer = optim.SGD(odonet.parameters(), lr=(learning_rate * 0.1))

            # torch.save({
            # 'epoch': epoch,
            # 'model_state_dict': odonet.state_dict(),
            # 'optimizer_state_dict': optimizer.state_dict(),
            # 'loss': loss}, checkpoint_path[0:-6]+"_index60.pth")
        # if(epoch == 300):
        #     optimizer = optim.SGD(odonet.parameters(), lr=(learning_rate * 0.01))
        #     torch.save({
        #     'epoch': epoch,
        #     'model_state_dict': odonet.state_dict(),
        #     'optimizer_state_dict': optimizer.state_dict(),
        #     'loss': loss}, checkpoint_path[0:-6]+"_index300.pth")

        for batch_idx, (raw_odo, gt_pose) in enumerate(train_loader):
            raw_odo = raw_odo.to(device).to(torch.float32)
            print(raw_odo.shape)
            print("-------------------")

            #将每个输入中的每个批次都拉成一个维度，也就是40帧输入不变
            raw_odo = raw_odo.view(raw_odo.size(0), -1)
            print(raw_odo.shape)

            gt_pose = gt_pose.to(device).to(torch.float32)

            #梯度清零
            optimizer.zero_grad()

            # 进入自动混合精度上下文
            with amp.autocast():
                #前向传播，将原始数据变成1*2的预测
                print(gt_pose.shape)
                pred_pose_euler = odonet.forward(raw_odo)
                # print("pred_pose_euler:{}".format(pred_pose_euler))
                loss = pose_loss(gt_pose, pred_pose_euler)
                # print("loss-typep:{}".format(loss))
                # print(loss)

            # 缩放损失
            scaler.scale(loss).backward(retain_graph = True)
            # loss.backward()
            # 使用autograd.grad()计算梯度
            grads = torch.autograd.grad(loss, odonet.parameters(), retain_graph=True)
            # max_norm = 1.0  # 设置最大的 L2 范数为 1
            # grad_norm = utils.clip_grad_norm_(odonet.parameters(), max_norm)


            scaler.step(optimizer)
            # optimizer.step()

            scaler.update()

            epoch_loss.append(loss.data.item())
            rot_error.append(pose_loss.rot_error.data.item())
            trans_error.append(pose_loss.trans_error.data.item())

        print('epoch: {}, loss: {:.10}, rot_err: {:.10}, trans_err: {:.10}'\
        .format(epoch, np.mean(np.array(epoch_loss)),\
                       np.mean(np.array(rot_error)), \
                       np.mean(np.array(trans_error))))
        if epoch % 10 == 0:
            torch.save({
            'epoch': epoch,
            'model_state_dict': odonet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss}, checkpoint_path + str(epoch).zfill(4)+".pth")

        loss_list.append(loss)


    print(type(loss_list))
    plt.figure()
#plt.plot(range(0, len(P_train_loss)), P_train_loss, 'r', label='P_Training loss')
    plt.plot(range(1, len(loss_list)+1),
         loss_list, 'r', label='Training loss')

#plt.plot(range(1, len(P_test_loss)+1), P_test_loss,
    #     'bo', label='P_Validation loss')

    #plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.legend()
    plt.show()#


    #plt.plot(range(1, len(R_test_loss)+1), R_test_loss,
        #     'g', label='R_Validation loss')


if __name__ == '__main__':

    SEED = 0
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)

    #训练集目录
   # train_dataset = "/data1/MAF_DATASET/421_POSE/"
    train_dataset = 'C:/zyf_transformer/transformer/data_vr_move/'

    #网络参数保存位置
    # checkpoint_path = "/home/everglow/momenta_workspace/odo_net/checkpoints/local_pose/421_local_pose_40frame"
    checkpoint_path = 'C:/zyf_transformer/transformer/data_vr_move/'

    #训练网络
    train_conv_odo_imu_net(train_dataset, checkpoint_path)
