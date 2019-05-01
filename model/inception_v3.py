import torchvision.models as models
import torch.nn as nn
import numpy as np

class inception_v3(nn.Module):
    def __init__(self,pca_dir="./pca_matrix/yt8m_pca"):
        self.base=models.inception_v3(pretrained=True)
        self.pca_dir=pca_dir
        if(self.pca_dir is not None):
            self._load_pca()

    def forward(self,x):
        items=self.base._modules.items()
        for i range(0,len(items)-1):
            x=items[i](x)
        if(self.pca_dir is not None):
            x=x-self.pca_mean
            x=x.mm(self.pca_eigenvecs)
            x=x/torch.sqrt(1e-8+self.pca_eigenvals)
        return x

    def _load_pca(self):
        self.pca_mean = numpy.load(
            os.path.join(self.pca_dir, 'mean.npy'))[:, 0]
        self.pca_mean=torch.Tensor(self.pca_mean).float().cuda().view(1,-1)
        self.pca_eigenvals = numpy.load(
            os.path.join(self.pca_dir, 'eigenvals.npy'))[:1024, 0]
        self.pca_eigenvals=torch.Tensor(self.pca_eigenvals).float().cuda().view(1,-1)
        self.pca_eigenvecs = numpy.load(
            os.path.join(self.pca_dir, 'eigenvecs.npy')).T[:, :1024]
        self.pca_eigenvecs=torch.Tensor(self.pca_eigenvecs).float().cuda().view(2048,1024)
