import torch
import os
import h5py

class KsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir = '/mnt/mmu/liuchang/hywData/ksData_split/', transform=None):
        super(KsDataset, self).__init__()
        # fh = open(root + datatxt, 'r')
        items = os.listdir(data_dir)

        self.items = os.listdir(data_dir)
        self.transform = transform
        self.data_dir = data_dir

    def __getitem__(self, index):

        data = h5py.File(self.data_dir + self.items[index])
        id = self.items[index].split('.')[0]
        frames = data[id + "_frames"]
        label = data[id + "_label"]

        # fn, label = self.items[index]
        # img = Image.open(root + fn).convert('RGB')
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, label

    def __len__(self):
        return len(self.imgs)

