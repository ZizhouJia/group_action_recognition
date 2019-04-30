import torch
import torch.nn as nn
import math
from torch.nn.parameter import Parameter
from torch.autograd import Variable

def create_Param(p_size,mean = 0, std = 1):
    weights = Parameter(torch.Tensor(*p_size))
    torch.nn.init.normal(weights, mean, std)

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
