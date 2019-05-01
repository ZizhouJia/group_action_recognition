import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.functional as F
from video_level_models import MoeModel
from util import *

class NetFV(nn.Module):

    def __init__(self, opt,feature_size, max_frames, cluster_size, add_batch_norm, is_training):
        super(NetFV, self).__init__()
        self.feature_size = feature_size
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = cluster_size

        if False:
            self.fv_couple_weights = opt.fv_couple_weights
            self.fv_couping_factor = opt.fc_couping_factor
        else:
            self.fv_couple_weights = True
            self.fv_couping_factor = 0.01

        self.cluster_weights = create_Param((self.feature_size,self.cluster_size),std=1 / math.sqrt(self.feature_size))
        self.covar_weights = create_Var((self.feature_size,self.cluster_size),mean=1.0,std=1 / math.sqrt(self.feature_size))

        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(max_frames)
        else:
            self.cluster_biases = create_Param((self.cluster_size),std=1/math.sqrt(self.feature_size))

        self.soft_layer = nn.Softmax()

        if not self.fv_couple_weights:
            self.cluster_weights2 = create_Param((1,self.feature_size,self.cluster_size),std = 1/math.sqrt(self.feature_size))


    def forward(self, reshaped_input):
        self.covar_weights = torch.pow(self.covar_weights,2)
        self.covar_weights = torch.add(self.covar_weights,1e-6)

        activation = torch.matmul(reshaped_input,self.cluster_weights)

        if self.add_batch_norm:
            activation = self.bn1(activation)
        else:
            activation += self.cluster_biases

        activation = self.soft_layer(activation)
        activation = activation.view(activation.shape[0],-1,self.max_frames,self.cluster_size)

        a_sum = torch.sum(activation,dim = -2, keepdim = True)

        if self.fv_couple_weights:
            cluster_weights2 = torch.mul(self.cluster_weights,self.fv_couping_factor)
        else:
            cluster_weights2 = self.cluster_weights2

        a = torch.mul(a_sum,cluster_weights2)
        activation = activation.permute(0,1,3,2)

        reshaped_input = reshaped_input.view(reshaped_input.shape[0],-1,self.max_frames,self.feature_size)
        fv1 = torch.matmul(activation,reshaped_input)
        fv1 = fv1.permute(0,1,3,2)

        a2 = torch.mul(a_sum,torch.pow(cluster_weights2,2))

        b2 = torch.mul(fv1,cluster_weights2)
        fv2 = torch.matmul(activation,torch.sqrt(reshaped_input))

        fv2 = fv2.permute(0,1,3,2)
        fv2 = a2 + fv2 + torch.mul(-2,b2)

        fv2 = torch.div(fv2,torch.sqrt(self.covar_weights))
        fv2 = fv2 - a_sum

        fv2 = fv2.view(fv2.shape[0],-1,self.cluster_size*self.feature_size)
        fv2 = nn.functional.normalize(fv2, dim=2, p=2)

        fv2 = fv2.view(fv2.shape[0], -1, self.cluster_size * self.feature_size)
        fv2 = nn.functional.normalize(fv2, dim=2, p=2)

        fv1 = fv1 - a
        fv1 = torch.div(fv1,a)

        fv1 = nn.functional.normalize(fv1, dim=2, p=2)
        fv1 = fv1.view(fv1.shape[0], -1, self.cluster_size * self.feature_size)
        fv1 = nn.functional.normalize(fv1, dim=2, p=2)

        return torch.cat((fv1,fv2),dim = 2)

class NetFVModellLF(nn.Module):
    def __init__(self,
                 opt = None,
                 feature_size = 1024 + 128,
                 vocab_size = 3862,
                 num_frames = 300,
                 iterations=100,
                 add_batch_norm=True,
                 sample_random_frames=300,
                 cluster_size=64,
                 hidden_size=2048,
                 is_training=True,
                 **unused_params):
        super(NetFVModellLF,self).__init__()

        self.iterations = iterations or opt.iterations
        self.add_batch_norm = add_batch_norm or opt.netvlad_add_batch_norm
        self.random_frames = sample_random_frames or opt.sample_random_frames
        self.cluster_size = cluster_size or opt.fv_cluster_size
        self.hidden1_size = hidden_size or opt.fv_hidden_size
        self.feature_size = feature_size

        if False:
            self.relu = opt.fv_relu
            self.gating = opt.gating
        else:
            self.relu = True
            self.gating = True

        self.max_frames = num_frames

        self.video_NetFV = NetFV(opt,1024, self.max_frames, cluster_size, add_batch_norm, is_training)
        self.audio_NetFV = NetFV(opt,128, self.max_frames, cluster_size / 2, add_batch_norm, is_training)

        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(self.max_frames)

        self.fv_dim = cluster_size *  1024 + cluster_size//2 * 128
        self.fv_dim = self.fv_dim * 2

        self.hidden1_weights = create_Param((self.fv_dim,self.hidden1_size),std = 1/math.sqrt(cluster_size))

        if self.add_batch_norm and self.relu:
            self.bn2 = nn.BatchNorm1d(1)
        else:
            self.hidden1_biases = create_Param((self.hidden1_size),std=0.01)

        if self.relu:
            self.relu6_layer = nn.ReLU6()

        if self.gating:
            self.gating_weights = create_Param((self.hidden1_size,self.hidden1_size),std = 1/math.sqrt(self.hidden1_size))

            if self.add_batch_norm:
                self.bn3 = nn.BatchNorm1d(1)
            else:
                self.gating_biases = create_Param((self.cluster_size),std=1/math.sqrt(self.feature_size))

            self.sig_layer = nn.Sigmoid()

        self.c_layer = MoeModel(opt=None, input_size=self.hidden1_size, vocab_size=vocab_size, is_training=True)

    def forward(self, reshaped_input):
        reshaped_input = self.bn1(reshaped_input)
        fv_video = self.video_NetFV(reshaped_input[:,:,0:1024])
        fv_audio = self.audio_NetFV(reshaped_input[:,:,1024:])

        fv = torch.cat([fv_video,fv_audio],dim = 2)

        activaiton = torch.matmul(fv,self.hidden1_weights)

        if self.add_batch_norm and self.relu:
            activaiton = self.bn2(activaiton)
        else:
            activaiton += self.hidden1_biases

        if self.relu:
            activaiton = self.relu6_layer(activaiton)

        if self.gating:
            gates = torch.matmul(activaiton,self.gating_weights)

            if self.add_batch_norm:
                gates = self.bn3(gates)
            else:
                gates += self.gating_biases

            gates = self.sig_layer(gates)
            activaiton = torch.mul(activaiton,gates)

        prob = self.c_layer(activaiton)
        return prob

feature = torch.ones((5,300,1024 + 128))

net = NetFVModellLF()
s = net(feature)
