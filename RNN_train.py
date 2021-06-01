import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

torch.manual_seed(1)


# 训练集
train_dataset = torchvision.datasets.MNIST(root="MINIST",train=True,transform=torchvision.transforms.ToTensor(), download=True)
# 测试集
test_dataset = torchvision.datasets.MNIST(root="MINIST", train=False, transform=torchvision.transforms.ToTensor())
test_x = test_dataset.data.type(torch.FloatTensor)[:2000]/255.
test_y = test_dataset.targets[:2000]
test_x = test_x.cuda()
test_y = test_y.cuda()
# print(test_dataset.test_data)
# print(test_dataset.test_data.size())
# plt.imshow(test_dataset.test_data[1].numpy())
# plt.show()

# 设置超参数

epoches = 20
batch_size = 2048
time_step = 28
input_size = 28
learning_rate = 0.01
hidden_size = 64        # rnn 隐藏单元数
num_layers = 1          # rnn 层数

# 将训练级集入Loader中
train_loader = Data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=3)

class LSTM_RNN(nn.Module):
    """搭建LSTM神经网络"""
    def __init__(self):
        super(LSTM_RNN, self).__init__()
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size,   # rnn 隐藏单元数
                            num_layers=num_layers,     # rnn 层数
                            batch_first=True, # If ``True``, then the input and output tensors are provided as (batch, seq, feature). Default: False
                            )
        self.output_layer = nn.Linear(in_features=64, out_features=10)

    def forward(self, x):
        # x shape (batch, time_step, input_size)
        # lstm_out shape (batch, time_step, output_size)
        # h_n shape (n_layers, batch, hidden_size)
        # h_c shape (n_layers, batch, hidden_size)
        lstm_out, (h_n, h_c) = self.lstm(x, None)   # If `(h_0, c_0)` is not provided, both **h_0** and **c_0** default to zero.
        output = self.output_layer(lstm_out[:, -1, :])   # 选择最后时刻lstm的输出
        return output


def main():
    lstm = LSTM_RNN()
    lstm = lstm.cuda()
    print(lstm)
    # 定义优化器和损失函数
    optimizer = torch.optim.Adam(lstm.parameters(), lr=learning_rate)
    loss_function = nn.CrossEntropyLoss()
    best_prec1 = 10000
    for epoch in range(epoches):
        print("进行第{}个epoch".format(epoch))
        valid_prec1 = 0
        for step, (batch_x, batch_y) in enumerate(train_loader):
            batch_x = batch_x.cuda()
            batch_y = batch_y.cuda()
            batch_x = batch_x.view(-1, 28, 28)
            output = lstm(batch_x)
            loss = loss_function(output, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            valid_prec1 = valid_prec1 + loss.data
            print('Epoch: ', epoch, 'step:', step, '| train loss: %.4f' % loss.data)
        is_best = valid_prec1 < best_prec1
        best_prec1 = min(valid_prec1, best_prec1)
        print('valid_prec1', valid_prec1, 'best_prec1', best_prec1)
        if is_best:
            torch.save({
                        'state_dict': lstm.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'epoch': epoch
                        },
                       'modelpara.pth')
            # if step % 50 == 0:
            #     test_output = lstm(test_x)
            #     pred_y = torch.max(test_output, dim=1)[1].data
            #     accuracy = ((pred_y == test_y.data).sum()) / float(test_y.size(0))
            #     print('Epoch: ', epoch, '| train loss: %.4f' % loss.data, '| test accuracy: %.2f' % accuracy)


    # test_output = lstm(test_x[:30])
    # pred_y = torch.max(test_output, dim=1)[1].squeeze()
    #
    # print(pred_y)
    # print(test_y[:30])

if __name__ == "__main__":
    main()
