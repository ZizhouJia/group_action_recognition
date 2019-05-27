import torch
import os
import h5py
import torch.utils.data
import numpy as np
import random

test = False

class split_feature_dataset(torch.utils.data.Dataset):
    def __init__(self, data_dir = None, mode = 'train'):
        if not test:
            data_dir = '/mnt/mmu/liuchang/hywData/ksData_gfeature'
        else:
            data_dir = '../feature'

        super(split_feature_dataset, self).__init__()

        self.items = os.listdir(data_dir)
        train_num = 0.8 * len(self.items)
        train_num = int(train_num)
        random.seed(9)
        random.shuffle(self.items)
        if mode == 'train':
            self.items = self.items[:train_num]
        else:
            self.items = self.items[train_num:]

        self.data_dir = data_dir

    def __getitem__(self, index):
        while(True):
            # try:
                data = h5py.File(os.path.join(self.data_dir,self.items[index]))
                id = self.items[index].split('.')[0]
                frames = data[id + "_features"]
                audio = data[id + "_audio"]
                label = data[id + "_label"]

                if label == 0:
                    rand_num = np.random.randint(0,100)
                    if rand_num > 60:
                        index=(index+1)%len(self.items)
                        continue

                frames = np.array(frames,dtype = 'uint8')
                audio = np.array(audio,dtype = 'float32')
                label=np.array(label,dtype='int64')

                data.close()

                frames = self.resize_frames(frames)
                audio = self.resize_frames(audio)

                features = np.concatenate((frames,audio),axis=-1)

                return id,features, label
            # except Exception as e:
            #     print("error detected")
            #     index=(index+1)%len(self.items)

    def __len__(self):
        return len(self.items)
        #return 2

    def gen_ind(self,fnum, ex_fnum=300):
        if ex_fnum > fnum:

            cp_time = ex_fnum // fnum

            rest_num = ex_fnum - fnum * cp_time
            if rest_num > 0:
                cp_inter = fnum // rest_num

            f_inds = []
            for i in range(fnum):
                for _ in range(cp_time):
                    f_inds.append(i)
                    if rest_num > 0 and i % cp_inter == 0:
                        f_inds.append(i)
                        rest_num -= 1

        elif ex_fnum < fnum:
            f_inds = []
            rm_times = fnum // ex_fnum
            for i in range(fnum):
                if i % rm_times == 0:
                    f_inds.append(i)

            n_fnum = len(f_inds)
            rest_num = n_fnum - ex_fnum

            if rest_num > 0:
                rm_inter = n_fnum // rest_num
                i = n_fnum - 1
                c = 0
                while i >= 0:
                    del f_inds[i]
                    # print(i)
                    i -= rm_inter
                    c += 1
                    if i < 0 or c == rest_num:
                        break
        else:
            f_inds = list(range(fnum))

        return f_inds

    def resize_frames(self,frames):

        f_shape = list(frames.shape)
        new_shape = [1]
        new_shape.extend(f_shape[1:])
        new_shape = tuple(new_shape)

        a_inds = self.gen_ind(fnum=f_shape[0])
        batch_list = [frames[ind].reshape(new_shape) for ind in a_inds]
        new_frames = np.concatenate(batch_list, axis=0)
        return new_frames

if __name__ == "__main__":
    ks_dataset = split_feature_dataset(mode='train')
    source_trainloader = torch.utils.data.DataLoader(ks_dataset, batch_size=1, shuffle=False,
                                                     num_workers=1, drop_last=True)
    tmp_loader = iter(source_trainloader)
    for i in range(10):
        _,input,batch = tmp_loader.next()
    print('input shape ' + str(input.shape))
    c = 1
