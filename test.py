import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
from RNN_train import LSTM_RNN

test_dataset = torchvision.datasets.MNIST(root="MINIST", train=False, transform=torchvision.transforms.ToTensor())
test_x = test_dataset.data.type(torch.FloatTensor)[:50]/255.
test_y = test_dataset.targets[:50]
test_x = test_x.cuda()
test_y = test_y.cuda()

lstm = LSTM_RNN()
lstm = lstm.cuda()
checkpoint = torch.load('modelpara.pth')
lstm.load_state_dict(checkpoint['state_dict'])


test_output = lstm(test_x)
pred_y = torch.max(test_output, dim=1)[1].squeeze()
print(pred_y)
print(test_y)