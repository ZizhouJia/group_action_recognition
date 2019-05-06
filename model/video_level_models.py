import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from util import *
import torch.functional as F

class MoeModel(nn.Module):
    def __init__(self,
                 opt,
                 input_size,
                 vocab_size,
                 is_training=True,
                 num_mixtures=2,
                 l2_penalty=1e-8,
                 **unused_params
                 ):
        super(MoeModel, self).__init__()
        self.input_size = input_size
        self.vocab_size = vocab_size

        if False:
            self.num_mixtures = num_mixtures or opt.moe_num_mixtures
            self.low_rank_gating = opt.moe_low_rank_gating
            self.l2_penalty = opt.moe_l2;
            self.gating_probabilities = opt.moe_prob_gating
            self.gating_input = opt.moe_prob_gating_input

            self.remove_diag = opt.gating_remove_diag
        else:
            self.num_mixtures = 2
            self.low_rank_gating = -1
            self.l2_penalty = 1e-6
            self.gating_probabilities = True
            self.gating_input = 'prob'


        if self.low_rank_gating == -1:
            self.gate_layer = nn.Linear(input_size,vocab_size*(1 + num_mixtures))
        else:
            self.gate_layer1 = nn.Linear(input_size,self.low_rank_gating)
            self.gate_layer = nn.Linear(self.low_rank_gating,vocab_size*(1 + num_mixtures))

        self.expert_layer = nn.Linear(input_size,vocab_size*num_mixtures)

        self.soft_layer = nn.Softmax()
        self.sig_layer = nn.Sigmoid()

        if self.gating_probabilities:
            if self.gating_input == 'prob':
                self.gating_weights = Parameter(torch.Tensor(self.vocab_size, self.vocab_size))
                torch.nn.init.normal(self.gating_weights, 0, 1 / math.sqrt(vocab_size))
            else:
                self.gating_weights = Parameter(torch.Tensor(self.input_size, self.vocab_size))
                torch.nn.init.normal(self.gating_weights, 0, 1 / math.sqrt(vocab_size))

        # self.bn_gate = nn.BatchNorm1d(1)
        self.bn_gate = bn_layer(vocab_size)
        self.sig_gate = nn.Sigmoid()

    def forward(self, model_input):
        if self.low_rank_gating == -1:
            gate_activations = self.gate_layer(model_input)
        else:
            gate_activations1 = self.gate_layer1(model_input)
            gate_activations = self.gate_layer(gate_activations1)

        expert_activations = self.expert_layer(model_input)

        gating_distribution = self.soft_layer(gate_activations.view(model_input.shape[0],-1,self.num_mixtures + 1))
        expert_distribution = self.sig_layer(expert_activations.view(model_input.shape[0],-1,self.num_mixtures))

        tmp_mul = gating_distribution[:,:,:self.num_mixtures] * expert_distribution
        probabilities_by_class_and_batch = torch.sum(tmp_mul,dim = 2)
        probabilities = probabilities_by_class_and_batch.view(model_input.shape[0],-1,self.vocab_size)

        if self.gating_probabilities:
            if self.gating_input == 'prob':
                gates = torch.matmul(probabilities,self.gating_weights)
            else:
                gates = torch.matmul(model_input,self.gating_weights)

            # gates = self.bn_gate(gates)
            gates = bn_action(gates,self.bn_gate)
            gates = self.sig_layer(gates)

            probabilities = torch.mul(probabilities,gates)

        probabilities = probabilities.view(probabilities.shape[0],-1)

        return probabilities