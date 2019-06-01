import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.functional as F
from .video_level_models import MoeModel
from .util import *

class NetFV(nn.Module):

    def __init__(self, opt,feature_size, max_frames, cluster_size, add_batch_norm, is_training,init_module=None,net_type='video_FV/'):
        super(NetFV, self).__init__()
        self.feature_size = int(feature_size)
        self.max_frames = max_frames
        self.is_training = is_training
        self.add_batch_norm = add_batch_norm
        self.cluster_size = int(cluster_size)
        self.init_module = init_module

        if False:
            self.fv_couple_weights = opt.fv_couple_weights
            self.fv_couping_factor = opt.fc_couping_factor
        else:
            self.fv_couple_weights = True
            self.fv_couping_factor = 0.01

        self.cluster_weights = create_Param((self.feature_size,self.cluster_size),std=1 / math.sqrt(self.feature_size),
                                            init_module=self.init_module,tensor_name=net_type + 'cluster_weights')

        # print('cluster_weights shape ' + str(self.cluster_weights.shape))


        self.covar_weights = create_Param((self.feature_size,self.cluster_size),mean=1.0,std=1 / math.sqrt(self.feature_size),
                                            init_module=self.init_module,tensor_name=net_type + 'covar_weights')

        # print('covar_weights shape ' + str(self.covar_weights.shape))
        # self.covar_weights = Parameter(torch.pow(self.covar_weights, 2))
        # self.covar_weights = Parameter(torch.add(self.covar_weights, 1e-6))

        if add_batch_norm:
            self.cluster_bn = bn_layer(self.cluster_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(self.cluster_bn, net_type + 'cluster_bn')
        else:
            self.cluster_biases = create_Param((self.cluster_size),std=1/math.sqrt(self.feature_size),
                                            init_module=self.init_module,tensor_name=net_type + 'cluster_biases')

        self.soft_layer = nn.Softmax()

        if not self.fv_couple_weights:
            self.cluster_weights2 = create_Param((1,self.feature_size,self.cluster_size),std = 1/math.sqrt(self.feature_size),
                                            init_module=self.init_module,tensor_name=net_type + 'cluster_weights2')


    def forward(self, reshaped_input):
        covar_weights = torch.pow(self.covar_weights, 2)
        covar_weights = torch.add(covar_weights, 1e-6)

        activation = torch.matmul(reshaped_input,self.cluster_weights)

        if self.add_batch_norm:
            activation = bn_action(activation,self.cluster_bn)
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
        fv2 = torch.matmul(activation,torch.pow(reshaped_input,2))

        fv2 = fv2.permute(0,1,3,2)
        # fv2 = a2 + fv2 + torch.mul(-2,b2)
        fv2 = a2 + fv2 -2*b2

        fv2 = torch.div(fv2,torch.pow(covar_weights,2))
        fv2 = fv2 - a_sum

        fv2 = fv2.view(fv2.shape[0],-1,self.cluster_size*self.feature_size)
        fv2 = nn.functional.normalize(fv2, dim=2, p=2)

        fv2 = fv2.view(fv2.shape[0], -1, self.cluster_size * self.feature_size)
        fv2 = nn.functional.normalize(fv2, dim=2, p=2)

        fv1 = fv1 - a
        fv1 = torch.div(fv1,covar_weights)

        fv1 = nn.functional.normalize(fv1, dim=2, p=2)
        fv1 = fv1.view(fv1.shape[0], -1, self.cluster_size * self.feature_size)
        fv1 = nn.functional.normalize(fv1, dim=2, p=2)

        return torch.cat((fv1,fv2),dim = 2)

class NetFVModellLF(nn.Module):
    def __init__(self,
                 opt = None,
                 video_size = 1024,
                 audio_size = 128,
                 vocab_size = 3862,
                 num_frames = 300,
                 iterations=100,
                 add_batch_norm=True,
                 sample_random_frames=300,
                 cluster_size=64,
                 hidden_size=2048,
                 is_training=True,
                 pretrain = False,
                 **unused_params):
        super(NetFVModellLF,self).__init__()

        self.iterations = iterations or opt.iterations
        self.add_batch_norm = add_batch_norm or opt.netvlad_add_batch_norm
        self.random_frames = sample_random_frames or opt.sample_random_frames
        self.cluster_size = cluster_size or opt.fv_cluster_size
        self.hidden1_size = hidden_size or opt.fv_hidden_size
        self.feature_size = video_size + audio_size
        self.video_size = video_size
        self.audio_size = audio_size

        if pretrain:
            self.init_module = init_Module(model_path='/mnt/mmu/liuchang/y8_tf/trained_models/NetFV/model.ckpt-244087',
                                       print_weight=False)
        else:
            self.init_module = None

        if False:
            self.relu = opt.fv_relu
            self.gating = opt.gating
        else:
            self.relu = True
            self.gating = True

        self.max_frames = num_frames

        if add_batch_norm:
            # self.input_bn = nn.BatchNorm1d(self.max_frames)
            self.input_bn = bn_layer(self.feature_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.input_bn, bn_name='tower/input_bn')

        self.video_NetFV = NetFV(opt,video_size, self.max_frames, cluster_size, add_batch_norm, is_training,init_module=self.init_module,net_type='tower/video_FV/')
        if self.audio_size > 0:
            self.audio_NetFV = NetFV(opt,audio_size, self.max_frames, cluster_size / 2, add_batch_norm, is_training,init_module=self.init_module,net_type='tower/audio_FV/')

        self.fv_dim = cluster_size * video_size + cluster_size // 2 * audio_size
        self.fv_dim = self.fv_dim * 2


        self.hidden1_weights = create_Param((self.fv_dim,self.hidden1_size),std = 1/math.sqrt(cluster_size),
                                            init_module=self.init_module, tensor_name='tower/hidden1_weights')

        if self.add_batch_norm and self.relu:
            # self.bn2 = nn.BatchNorm1d(1)
            self.hidden1_bn = bn_layer(self.hidden1_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.hidden1_bn, bn_name='tower/hidden1_bn')
        else:
            self.hidden1_biases = create_Param((self.hidden1_size),std=0.01,
                                  init_module = self.init_module, tensor_name = 'tower/hidden1_biases')

        if self.relu:
            self.relu6_layer = nn.ReLU6()

        if self.gating:
            self.gating_weights = create_Param((self.hidden1_size,self.hidden1_size),std = 1/math.sqrt(self.hidden1_size),
                                  init_module = self.init_module, tensor_name = 'tower/gating_weights_2')

            if self.add_batch_norm:
                self.gating_bn = bn_layer(self.hidden1_size)
                if type(self.init_module) != type(None):
                    self.init_module.init_bn(bn=self.gating_bn, bn_name='tower/gating_bn')
            else:
                self.gating_biases = create_Param((self.hidden1_size),std=1/math.sqrt(self.feature_size),
                                  init_module = self.init_module, tensor_name = 'tower/gating_biases')

            self.sig_layer = nn.Sigmoid()

        self.c_layer = MoeModel(opt=None, input_size=self.hidden1_size, vocab_size=vocab_size, is_training=True)

    def forward(self, reshaped_input):
        reshaped_input = bn_action(reshaped_input,self.input_bn)

        fv_video = self.video_NetFV(reshaped_input[:,:,0:self.video_size])

        if self.audio_size > 0:
            fv_audio = self.audio_NetFV(reshaped_input[:,:,self.video_size:])
            fv = torch.cat([fv_video,fv_audio],dim = 2)
        else:
            fv = fv_video

        activaiton = torch.matmul(fv,self.hidden1_weights)

        if self.add_batch_norm and self.relu:
            activaiton = bn_action(activaiton,self.hidden1_bn)
        else:
            activaiton += self.hidden1_biases

        if self.relu:
            activaiton = self.relu6_layer(activaiton)

        if self.gating:
            gates = torch.matmul(activaiton,self.gating_weights)

            if self.add_batch_norm:
                gates = bn_action(gates,self.gating_bn)
            else:
                gates += self.gating_biases

            gates = self.sig_layer(gates)
            activaiton = torch.mul(activaiton,gates)

        prob = self.c_layer(activaiton)
        return prob
