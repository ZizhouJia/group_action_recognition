import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from tensorflow.python import pywrap_tensorflow as pt

nn.BatchNorm1d

def bn_action(tensor,bn):
    tmp_list = []
    for i in range(tensor.shape[0]):
        tmp_batch = bn(tensor[i])
        tmp_list.append(tmp_batch.unsqueeze(0))
        # tensor[i] = tmp_batch
    return torch.cat(tmp_list,dim = 0)

class init_Module():
    # def __init__(self,model_path = '/home/hyw/y8_willow/gatednetvladLF-256k-1024-80-0002-300iter-norelu-basic-gatedmoe/model.ckpt-0',print_weight = False):
    def __init__(self,model_path = '/mnt/mmu/liuchang/hywData/model/NetVLADModel/model.ckpt-310001',print_weight = False):
        self.reader = pt.NewCheckpointReader(model_path)
        if print_weight:
            vars = self.reader.get_variable_to_shape_map()
            for k in sorted(vars):
                print(k, vars[k])

        # value = reader.get_tensor("tensor_name")

    def get_tensor(self,tensor_name):
        value = self.reader.get_tensor(tensor_name)
        return value

    def init_bn(self,bn,bn_name):
        if bn_name == '':
            return
        bn_weight = self.get_tensor(bn_name + '/gamma')
        bn_bias = self.get_tensor(bn_name + '/beta')
        tmp_weight = torch.from_numpy(bn_weight)
        tmp_bias = torch.from_numpy(bn_bias)
        bn._parameters['weight'].data = tmp_weight
        bn._parameters['bias'].data = tmp_bias
        return bn


class bn_layer(nn.Module):
    def __init__(self,feature_size):
        super(bn_layer, self).__init__()
        self.feature_size = feature_size
        # self.mean = Parameter(torch.zeros(int(feature_size)))
        # self.variance = Parameter(torch.ones(int(feature_size)))
        # self.mean = None
        # self.variance = None

        self.register_buffer('mean', torch.zeros(int(feature_size)))
        self.register_buffer('variance', torch.ones(int(feature_size)))

        # self.mean.requires_grad = False
        # self.variance.requires_grad = False
        self.weight = create_Param((int(feature_size),))
        self.bias = create_Param((int(feature_size),))

    def forward(self,input):
        # output = torch.mul(input,self.weight)
        # for i in range(output.shape[0]):
        #     output[i] += self.bias
        #
        # return output

        # self.mean = torch.zeros(int(self.feature_size)).cuda()
        # self.variance = torch.ones(int(self.feature_size)).cuda()

        return nn.functional.batch_norm(
            input, self.mean, self.variance, self.weight, self.bias,training = False)

    # def cuda(self):
    #     super(bn_layer, self).cuda()
    #     self.mean = self.mean.cuda()
    #     self.variance = self.variance.cuda()
    #     # self.weight = self.weight.cuda()
    #     # self.bias = self.bias.cuda()

def Dequantize(feat_vector, max_quantized_value=2, min_quantized_value=-2):
  """Dequantize the feature from the byte format to the float format.

  Args:
    feat_vector: the input 1-d vector.
    max_quantized_value: the maximum of the quantized value.
    min_quantized_value: the minimum of the quantized value.

  Returns:
    A float vector which has the same shape as feat_vector.
  """
  assert max_quantized_value > min_quantized_value
  quantized_range = max_quantized_value - min_quantized_value
  scalar = quantized_range / 255.0
  bias = (quantized_range / 512.0) + min_quantized_value
  return feat_vector * scalar + bias

def create_Param(p_size,mean = 0, std = 1,init_module = None,tensor_name = ''):
    if type(init_module) == type(None):
        weights = Parameter(torch.Tensor(*p_size))
        torch.nn.init.normal(weights, mean, std)
    else:
        tmp_np = init_module.get_tensor(tensor_name)
        tmp_tensor = torch.from_numpy(tmp_np)
        weights = Parameter(tmp_tensor)
    return weights

def create_Var(p_size,mean = 0, std = 1):
    weights = Variable(torch.Tensor(*p_size))
    torch.nn.init.normal(weights, mean, std)

    return weights


# s = torch.Tensor(*[1,2,3])
# c = 1
def FramePooling(frames, method, **unused_params):
  """Pools over the frames of a video.

  Args:
    frames: A tensor with shape [batch_size, num_frames, feature_size].
    method: "average", "max", "attention", or "none".
  Returns:
    A tensor with shape [batch_size, feature_size] for average, max, or
    attention pooling. A tensor with shape [batch_size*num_frames, feature_size]
    for none pooling.

  Raises:
    ValueError: if method is other than "average", "max", "attention", or
    "none".
  """
  if method == "average":
    return torch.mean(frames, 2)[0]
  elif method == "max":
    return torch.max(frames, 2)[0]
