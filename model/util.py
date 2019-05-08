import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def bn_action(tensor,bn):
    for i in range(tensor.shape[0]):
        tmp_batch = bn(tensor[i])
        tensor[i] = tmp_batch
        return tensor

class bn_layer(nn.Module):
    def __init__(self,feature_size):
        super(bn_layer, self).__init__()
        self.mean = torch.zeros(int(feature_size))
        self.variance = torch.ones(int(feature_size))
        self.weight = create_Param((int(feature_size),))
        self.bias = create_Param((int(feature_size),))

    def forward(self,input):
        return nn.functional.batch_norm(
            input, self.mean, self.variance, self.weight, self.bias,training = False)


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
