import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.functional as F
from .video_level_models import MoeModel
from .util import *

class DbofModel(nn.Module):
    def __init__(self,
                   opt = None,
                   model_input = None,
                   vocab_size = 3862,
                   num_frames = 300,
                   iterations=100,
                   add_batch_norm=True,
                   sample_random_frames=300,
                   cluster_size=8192,
                   hidden_size=1024,
                   is_training=True,
                   feature_size = 1024 + 128,
                   pretrain = True,
                   use_moe=False,
                   **unused_params):
        super(DbofModel,self).__init__()

        self.iterations = iterations or opt.iterations
        self.add_batch_norm = add_batch_norm or opt.dbof_add_batch_norm
        self.random_frames = sample_random_frames or opt.sample_random_frames
        self.cluster_size = cluster_size or opt.dbof_cluster_size
        self.feature_size = feature_size
        self.hidden1_size = hidden_size or opt.dbof_hidden_size
        self.add_batch_norm = add_batch_norm

        self.num_frames = num_frames;
        self.max_frames = num_frames;

        if pretrain:
            self.init_module = init_Module(model_path='/mnt/mmu/liuchang/y8_tf/trained_models/DbofModel/model.ckpt-246805',print_weight=False)
        else:
            self.init_module = None

        if add_batch_norm:
            self.input_bn = bn_layer(feature_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.input_bn,bn_name='tower/input_bn')

        self.cluster_weights = create_Param((self.feature_size,self.cluster_size),std=1 / math.sqrt(self.feature_size),
                                            init_module=self.init_module,tensor_name='tower/cluster_weights')

        if add_batch_norm:
            self.cluster_bn = nn.BatchNorm1d(self.cluster_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.cluster_bn,bn_name='tower/cluster_bn')
        else:
            self.cluster_biases = create_Param((self.cluster_size,),std = 1/math.sqrt(self.feature_size),
                                               init_module=self.init_module,tensor_name='tower/cluster_biases')

        self.relu6_layer1 = nn.ReLU6()
        self.hidden1_weights = create_Param((self.cluster_size,self.hidden1_size),std = 1/math.sqrt(self.cluster_size),
                                            init_module=self.init_module, tensor_name='tower/hidden1_weights')

        if add_batch_norm:
            self.hidden1_bn = bn_layer(self.hidden1_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.hidden1_bn,bn_name='tower/hidden1_bn')
        else:
            self.hidden1_bias = create_Param((self.hidden1_size,),std = 0.01,
                                             init_module=self.init_module,tensor_name='tower/hidden1_biases')

        self.relu6_layer2 = nn.ReLU6()

        self.use_moe = use_moe
        if self.use_moe:
            self.c_layer = MoeModel(opt=None, input_size=self.hidden1_size, vocab_size=vocab_size, is_training=True)
        else:
            self.c_layer = nn.Linear(self.hidden1_size, vocab_size)

        # self.c_layer = MoeModel(opt=None, input_size=self.hidden1_size, vocab_size=vocab_size, is_training=True)

    def forward(self, reshaped_input):
        if self.add_batch_norm:
            reshaped_input = bn_action(reshaped_input, self.input_bn)
        activations = torch.matmul(reshaped_input, self.cluster_weights)

        if self.add_batch_norm:
            activations = bn_action(activations, self.cluster_bn)
        else:
            activations += self.cluster_biases

        activations = self.relu6_layer1(activations)
        activations = activations.view(activations.shape[0], -1, self.max_frames, self.cluster_size)

        activations = FramePooling(activations, 'max')
        activations = torch.matmul(activations, self.hidden1_weights)
        if self.add_batch_norm:
            activations = bn_action(activations, self.hidden1_bn)
        else:
            activations += self.hidden1_bias

        activations = self.relu6_layer2(activations)

        if not self.use_moe:
            activations = activations.squeeze(1)
        prob = self.c_layer(activations)

        return prob
