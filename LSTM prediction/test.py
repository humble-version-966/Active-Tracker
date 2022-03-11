import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
#引入相关的文件及数据集,数据集的数据来源于csv文件
import pandas as pd 

df = pd.read_csv("random walk simulation/postion.csv")
data = np.transpose(df.to_numpy()).astype(np.float32)

x_data = data[0, :3000]                         #数据切片，x_data表示自变量
y_data = data[1, :3000]

input_size = 1                                  #定义超参数输入层，输入数据为1维
output_size = 1                                 #定义超参数输出层，输出数据为1维
num_layers = 1                                  #定义超参数rnn的层数，层数为1层
hidden_size = 32                                #定义超参数rnn的循环神经元个数，个数为32个
learning_rate = 0.02                            #定义超参数学习率
train_step = 1000                                #定义训练的批次，3000个数据共训练1000次，
time_step = 3                                  #定义每次训练的样本个数每次传入3个样本
h_state = None                                  #初始化隐藏层状态
use_gpu = torch.cuda.is_available()             #使用GPU加速训练
class RNN(nn.Module):
    """搭建rnn网络"""
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(RNN, self).__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,)                  #传入四个参数，这四个参数是rnn()函数中必须要有的
        self.output_layer = nn.Linear(in_features=hidden_size, out_features=output_size)
    def forward(self, x, h_state):
        # x (batch, time_step, input_size)
        # h_state (n_layers, batch, hidden_size)
        # rnn_out (batch, time_step, hidden_size)
        rnn_out, h_state = self.rnn(x, h_state)     #h_state是之前的隐层状态
        out = []
        for time in range(rnn_out.size(1)):
            every_time_out = rnn_out[:, time, :]    #相当于获取每个时间点上的输出，然后过输出层
            out.append(self.output_layer(every_time_out))
        return torch.stack(out, dim=1), h_state     #torch.stack扩成[1, output_size, 1]
        
# 显示由csv提供的样本数据图
# plt.figure(1)
# plt.plot(x_data, y_data, 'r-', label='target (Ca)')
# plt.legend(loc='best')
# plt.savefig("LSTM prediction/prepredict.png")


#对CLASS RNN进行实例化时向其中传入四个参数
rnn = RNN(input_size, hidden_size, num_layers, output_size)
# 设置优化器和损失函数
#使用adam优化器进行优化，输入待优化参数rnn.parameters，优化学习率为learning_rate
optimizer = torch.optim.Adam(rnn.parameters(), lr=learning_rate)
loss_function = nn.MSELoss()                #损失函数设为常用的MES均方根误差函数
plt.figure(2)                               #新建一张空白图片2
plt.ion()
# 按照以下的过程进行参数的训练
for step in range(train_step):
    start, end = step*time_step, (step+1)*time_step#
    steps = np.linspace(start, end, (end-start), dtype=np.float32)#该参数仅仅用于画图过程中使用
    x_np = x_data[start:end]        #按照批次大小从样本中切片出若干个数据，用作RNN网络的输入
    y_np = y_data[start:end]        #按照批次大小从样本中切片出若干个数据，用作与神经网络训练的结果对比求取损失
    x = torch.from_numpy(x_np[np.newaxis, :, np.newaxis])
    y = torch.from_numpy(y_np[np.newaxis, :, np.newaxis])
    pridect, h_state = rnn.forward(x, h_state)
    h_state = h_state.detach()     # 重要！！！ 需要将该时刻隐藏层的状态作为下一时刻rnn的输入

    loss = loss_function(pridect, y)#求解损失值，该损失值用于后续参数的优化
    optimizer.zero_grad()           #优化器的梯度清零，这一步必须要做

    loss.backward()                 #调用反向传播网络对损失值求反向传播，优化该网络
    optimizer.step()                #调用优化器对rnn中所有有关参数进行优化处理

    plt.plot(steps, pridect.detach().numpy().flatten(), 'b-')
    plt.draw()
    plt.pause(0.05)
    plt.ioff()
    # plt.show()
    plt.savefig("LSTM prediction/predict.png")