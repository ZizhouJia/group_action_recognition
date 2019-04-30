import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.functional as F
from video_level_models import MoeModel

import copy
from torch.autograd import Variable

class NetVLADModelLF(nn.Module):
    def __init__(self,
                 opt = None,
                 model_input = None,
                 vocab_size = 3862,
                 feature_size = 1024 + 128,
                 num_frames = 300,
                 iterations=100,
                 add_batch_norm=True,
                 sample_random_frames=None,
                 cluster_size=256,
                 hidden_size=1024,
                 is_training=True,
                 **unused_params):
        super(NetVLADModelLF, self).__init__()

        iterations = iterations or opt.iterations
        add_batch_norm = add_batch_norm or opt.netvlad_add_batch_norm
        # random_frames = sample_random_frames or opt.sample_random_frames
        self.cluster_size = cluster_size or opt.netvlad_cluster_size
        self.hidden1_size = hidden_size or opt.netvlad_hidden_size

        if False:
            relu = opt.netvlad_relu
            dimred = opt.netvlad_dimred
            self.gating = opt.gating
            self.remove_diag = opt.gating_remove_diag
            lightvlad = opt.lightvlad
            vlagd = opt.vlagd
        else:
            relu = False
            dimred = -1
            self.gating = True
            self.remove_diag = False
            lightvlad = False
            vlagd = False

        self.num_frames = num_frames
        self.max_frames = num_frames
        self.add_batch_norm = add_batch_norm
        self.relu = relu
        self.feature_size = feature_size

        self.video_NetVLAD = NetVLAD(1024, self.max_frames, cluster_size, add_batch_norm, is_training)
        self.audio_NetVLAD = NetVLAD(128, self.max_frames, cluster_size / 2, add_batch_norm, is_training)

        vlad_dim = self.cluster_size * 1024 + cluster_size // 2 * 128

        self.hidden1_weights = Parameter(torch.Tensor(vlad_dim, self.hidden1_size))
        torch.nn.init.normal(self.hidden1_weights, 0, 1 / math.sqrt(self.cluster_size))

        self.bn1 = nn.BatchNorm1d(self.max_frames)
        self.bn3 = nn.BatchNorm1d(1)
        if self.add_batch_norm and self.relu:
            self.bn2 = nn.BatchNorm1d(1)
        else:
            self.hidden1_biases = Parameter(torch.Tensor(self.hidden1_size))
            torch.nn.init.normal(self.hidden1_biases, 0, 0.01)

        self.sig_layer = nn.Sigmoid()

        self.c_layer = MoeModel(opt=None,input_size=hidden_size,vocab_size = vocab_size,is_training = True)

        if self.relu:
            self.relu6_layer = nn.ReLU6()

        if self.gating:
            self.gating_weights = Parameter(torch.Tensor(self.hidden1_size,self.hidden1_size))
            torch.nn.init.normal(self.gating_weights, 0, 1/math.sqrt(self.hidden1_size))

            if add_batch_norm:
                self.gating_biases = Parameter(torch.Tensor(self.cluster_size))
                torch.nn.init.normal(self.gating_biases, 0, 1 / math.sqrt(feature_size))

            if self.remove_diag:
                pass



    def forward(self, reshaped_input):
        reshaped_input = self.bn1(reshaped_input)

        vlad_video = self.video_NetVLAD(reshaped_input[:,:,0:1024])
        vlad_audio = self.audio_NetVLAD(reshaped_input[:,:,1024:])

        vlad = torch.cat((vlad_video,vlad_audio),dim = 2)


        activation = torch.matmul(vlad,self.hidden1_weights)

        if self.add_batch_norm and self.relu:
            activation = self.bn2(activation)
        else:
            activation += self.hidden1_biases

        if self.relu:
            activation = self.relu6_layer(activation)

        if self.gating:
            gates = torch.matmul(activation,self.gating_weights)

            if self.add_batch_norm:
                gates = self.bn3(gates)
            else:
                gates += self.gating_biases

            gates = self.sig_layer(gates)

        activation = torch.mul(activation,gates)

        prob = self.c_layer(activation)
        return prob

class NetVLAD(nn.Module):
    def __init__(self, feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        super(NetVLAD,self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        self.fc1 = nn.Linear(self.feature_size, self.cluster_size, bias=False)
        for p in self.fc1.parameters():
            torch.nn.init.normal(p, 0, 1 / math.sqrt(self.feature_size))

        if self.add_batch_norm:
            self.bn1 = nn.BatchNorm1d(self.max_frames)

        else:
            self.bias1 = Parameter(torch.Tensor(self.cluster_size))
            torch.nn.init.normal(self.bias1, 0, 1 / math.sqrt(self.feature_size))

        self.cluster_weights2 = Parameter(torch.Tensor(1,self.feature_size,self.cluster_size))
        torch.nn.init.normal(self.cluster_weights2, 0, 1 / math.sqrt(self.feature_size))

        self.softmax_layer = nn.Softmax()

    def forward(self, reshaped_input):
        activation = self.fc1(reshaped_input)
        if self.add_batch_norm:
            activation = self.bn1(activation)
        else:
            activation = self.activation + self.bias1

        activation = self.softmax_layer(activation)
        activation = activation.view(reshaped_input.shape[0], -1, self.max_frames, self.cluster_size)

        a_sum = torch.sum(activation,keepdim = True,dim = -2)
        a = torch.mul(a_sum,self.cluster_weights2)
        activation = activation.permute(0,1,3,2)

        reshaped_input = reshaped_input.view(reshaped_input.shape[0], -1, self.max_frames, self.feature_size)
        vlad = torch.matmul(activation, reshaped_input)
        vlad = vlad.permute(0, 1, 3, 2)
        vlad = vlad - a

        vlad = nn.functional.normalize(vlad, dim=2, p=2)

        vlad = vlad.view(reshaped_input.shape[0], -1, self.cluster_size * self.feature_size)
        vlad = nn.functional.normalize(vlad, dim=2, p=2)

        return vlad



class LightVLAD(nn.Module):

    def __init__(self,feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        super(LightVLAD, self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        self.fc1 = nn.Linear(self.feature_size,self.cluster_size,bias=False)
        for p in self.fc1.parameters():
            torch.nn.init.normal(p,0,1 / math.sqrt(self.feature_size))

        if self.add_batch_norm:
            self.bn1 = nn.BatchNorm1d(self.max_frames)

        else:
            self.bias1 = Parameter(torch.Tensor(self.cluster_size))
            torch.nn.init.normal(self.bias1, 0, 1 / math.sqrt(self.feature_size))

        self.softmax_layer = nn.Softmax()
        self.fc2 = nn.Linear(self.cluster_size, self.feature_size, bias=False)

    def forward(self, reshaped_input):
        activation = self.fc1(reshaped_input)
        if self.add_batch_norm:
            activation = self.bn1(activation)
        else:
            activation = self.activation + self.bias1

        activation = self.softmax_layer(activation)
        activation = activation.view(reshaped_input.shape[0],-1,self.max_frames,self.cluster_size)
        activation = activation.permute(0,1,3,2)

        reshaped_input = reshaped_input.view(reshaped_input.shape[0],-1,self.max_frames,self.feature_size)
        vlad = self.fc2(activation,reshaped_input)

        vlad = vlad.permute(0,1,3,2)
        vlad = F.norm(vlad,dim=2,p=2)

        return vlad


# feature = torch.Tensor(5,300,1024 + 128)

# net = NetVLADModelLF()
# s = net(feature)
