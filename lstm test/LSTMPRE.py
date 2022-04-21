# Import the necessary modules and libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from tqdm import tqdm
import torch
import torch.nn as nn
from pandas import read_csv
from torch.autograd import Variable

data_cv = pd.read_csv('Good_Data/mergedata.csv')
data_csv=data_cv['Value']
data_csv = data_csv.dropna()  # 滤除缺失数据
dataset = data_csv.values   # 获得csv的值
all_data = dataset.astype('float32')


test_data_size = 900


from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(-1, 1))
all_data_normalized = scaler.fit_transform(all_data .reshape(-1, 1))
all_data_normalized = torch.FloatTensor(all_data_normalized).view(-1).cuda()

test_data = all_data_normalized[:-test_data_size]
train_data_normalized= all_data_normalized[-test_data_size:]

train_window = 60
def create_inout_sequences(input_data, tw):
    inout_seq = []
    L = len(input_data)
    for i in range(L-tw):
        train_seq = input_data[i:i+tw]
        train_label = input_data[i+tw:i+tw+1]
        inout_seq.append((train_seq ,train_label))
    return inout_seq
train_inout_seq = create_inout_sequences(train_data_normalized, train_window)

class LSTM(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=200, output_size=1):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_layer_size).cuda()

        self.linear = nn.Linear(hidden_layer_size, output_size).cuda()

        self.hidden_cell = (torch.zeros(1,1,self.hidden_layer_size,device='cuda'),
                            torch.zeros(1,1,self.hidden_layer_size,device='cuda'))

    def forward(self, input_seq):
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq) ,1, -1), self.hidden_cell)
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        return predictions[-1].cuda()

model = LSTM().cuda()
loss_function = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
epochs = 2

for i in tqdm(range(epochs)):
    for seq, labels in train_inout_seq:
        optimizer.zero_grad()
        model.hidden_cell = (torch.zeros(1, 1, model.hidden_layer_size,device='cuda'),
                        torch.zeros(1, 1, model.hidden_layer_size,device='cuda'))

        y_pred = model(seq)

        single_loss = loss_function(y_pred, labels)
        single_loss.backward()
        optimizer.step()

    if i%25 == 1:
        print(f'epoch: {i:3} loss: {single_loss.item():10.8f}')

print(f'epoch: {i:3} loss: {single_loss.item():10.10f}')

fut_pred = test_data_size

test_inputs = train_data_normalized[-train_window:].tolist()
# print(test_inputs)

model.eval()

for i in tqdm(range(len(all_data)-fut_pred-train_window)):
    seq = test_data[i:train_window+i]
    with torch.no_grad():
        model.hidden = (torch.zeros(1, 1, model.hidden_layer_size),
                        torch.zeros(1, 1, model.hidden_layer_size))
        test_inputs.append(model(seq).item())


actual_predictions = scaler.inverse_transform(np.array(test_inputs ).reshape(-1, 1))
np.savetxt( "Other's bad solution/LSTM_bit_front.csv",actual_predictions,delimiter=',')

x = np.arange(0, len(all_data)-fut_pred, 1)
# print(x)

plt.title('Month vs Passenger')
plt.ylabel('Bitcoin ($)')
plt.grid(True)
plt.autoscale(axis='x', tight=True)
plt.plot(all_data)
plt.plot(x,actual_predictions)
plt.show()
