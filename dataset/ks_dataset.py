import torch
import os
import h5py
import torch.utils.data
import numpy as np

class KsDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir = 'save_data', transform=None):
        super(KsDataset, self).__init__()

        self.items = os.listdir(data_dir)
        self.transform = transform
        self.data_dir = data_dir

    def __getitem__(self, index):

        data = h5py.File(os.path.join(self.data_dir,self.items[index]))
        id = self.items[index].split('.')[0]
        frames = data[id + "_frames"]
        label = data[id + "_label"]

        frames = np.array(frames,dtype = 'float')
        label = np.array(label,dtype = 'uint8')

        frames = torch.from_numpy(frames)
        label = torch.from_numpy(label)


        # fn, label = self.items[index]
        # img = Image.open(root + fn).convert('RGB')
        if self.transform is not None:
            frames = self.transform(frames)
        return frames, label

    def __len__(self):
        return len(self.items)


if __name__ == "__main__":
    ks_dataset = KsDataset()
    source_trainloader = torch.utils.data.DataLoader(ks_dataset, batch_size=1, shuffle=False,
                                                     num_workers=1, drop_last=True)
    tmp_loader = iter(source_trainloader)
    for i in range(10):
        input,batch = tmp_loader.next()
    print('input shape ' + str(input.shape))
    c = 1