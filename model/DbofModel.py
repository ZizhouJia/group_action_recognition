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

        if add_batch_norm:
            self.bn1 = nn.BatchNorm1d(self.max_frames)

        self.cluster_weights = create_Param((self.feature_size,self.cluster_size),std=1 / math.sqrt(self.feature_size))

        if add_batch_norm:
            self.bn2 = nn.BatchNorm1d(self.max_frames)
        else:
            self.cluster_biases = create_Param((self.cluster_size),std = 1/math.sqrt(self.feature_size))

        self.relu6_layer1 = nn.ReLU6()
        self.hidden1_weights = create_Param((self.cluster_size,self.hidden1_size),std = 1/math.sqrt(self.cluster_size))

        if add_batch_norm:
            self.bn3 = nn.BatchNorm1d(1)
        else:
            self.hidden1_bias = create_Param((self.hidden1_bias),std = 0.01)

        self.relu6_layer2 = nn.ReLU6()

        self.c_layer = MoeModel(opt=None, input_size=self.hidden1_size, vocab_size=vocab_size, is_training=True)

    def forward(self, reshaped_input):
        if self.add_batch_norm:
            reshaped_input = self.bn1(reshaped_input)
        activations = torch.matmul(reshaped_input,self.cluster_weights)

        if self.add_batch_norm:
            activations = self.bn2(activations)
        else:
            activations += self.cluster_biases

        activations = self.relu6_layer1(activations)
        activations = activations.view(activations.shape[0],-1,self.max_frames,self.cluster_size)

        activations = FramePooling(activations,'max')
        activations = torch.matmul(ctivations,self.hidden1_weights)
        if self.add_batch_norm:
            activations = self.bn3(activations)
        else:
            activations += self.hidden1_bias

        activations = self.relu6_layer2(activations)

        prob = self.c_layer(activations)

        return prob
