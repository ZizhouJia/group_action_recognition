import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
import torch.functional as F
from .video_level_models import MoeModel
from .util import *

class lstmModel(nn.Module):
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
                   video_size = 1024,
                   audio_size = 128,
                   pretrain = False,
                   use_moe=False,
                   **unused_params):
        super(lstmModel,self).__init__()

        self.iterations = iterations or opt.iterations
        self.feature_size = audio_size + video_size
        self.add_batch_norm = add_batch_norm
        self.hidden1_size = hidden_size

        self.num_frames = num_frames;
        self.max_frames = num_frames;

        if pretrain:
            self.init_module = init_Module(model_path='/mnt/mmu/liuchang/y8_tf/trained_models/DbofModel/model.ckpt-246805',print_weight=False)
        else:
            self.init_module = None

        print('pretrain ' + str(pretrain))
        print('feature size ' + str(self.feature_size))

        if add_batch_norm:
            self.input_bn = bn_layer(self.feature_size)
            if type(self.init_module) != type(None):
                self.init_module.init_bn(bn=self.input_bn,bn_name='tower/input_bn')

        self.lstm_layer = nn.LSTM(self.feature_size, self.hidden1_size, 2)

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

        activations, _ = self.lstm_layer(reshaped_input)
        a_shape = activations.shape
        activations = activations[:,-1,:].reshape(a_shape[0],self.feature_size)
        activations = self.relu6_layer2(activations)

        if not self.use_moe:
            activations = activations.squeeze(1)
        prob = self.c_layer(activations)

        return prob

# if __name__ == '__main__':
#     s = torch.Tensor(3,300,1024)
#     s = s.cuda()
#     model = DbofModel()
#     model = model.cuda()
#     t = model(s)