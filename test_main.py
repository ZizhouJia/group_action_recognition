import torch
from model.NetVLADModel import *
from model.DbofModel import *
from model.NetFVModel import *
import numpy as np
import torch.optim as optim

if __name__ == '__main__':
    feature = torch.rand(5, 300, 1024 + 128)
    feature = feature.cuda()
    net = NetFVModellLF()
    net = net.cuda()
    # net._parameters['hidden1_weights'].data =
    s = net(feature)

    optimizerD = optim.Adam(net.parameters(), lr=0.001)
    criterion_c = nn.MSELoss()

    label = torch.from_numpy(np.zeros((5,3862)).astype('float'))
    label = label.float()
    label = label.cuda()

    err = criterion_c(s, label)

    err.backward(retain_graph=True)
    optimizerD.step()