import torch.utils.data as Data
import torch
import h5py as h5
import os
import numpy as np
import random

class feature_dataset(Data.Dataset):
    def __init__(self,frame_path ='/mnt/mmu/liuchang/kuaishou_feature.h5',data_type = 'train'):
        data = h5.File(frame_path,'r')
        # print('read_finish!')
        self.frames = np.array(data['feature']).reshape((-1,300,1024))
        # print('frames shape ' + str(self.frames.shape))
        # print('read_finish 1!')
        self.labels=np.array(data['label']).reshape((-1,))
        # print('read_finish 2!')
        self.image_id = data['image_id']
        # print('read_finish 3!')
        self.audio_path = '/mnt/mmu/liuchang/hywData/ksData_audio300'
        self.a_path_list = os.listdir(self.audio_path)
        # self.audio_id = [path.split[0] for path in a_path_list]

        data_list = []
        self.data_num = self.frames.shape[0]
        for i in range(self.data_num):

            if data_type == 'train':
                if i % 10 >= 0 and i % 10 <= 7:
                    data_list.append(i)

            elif data_type == 'valid':
                if i % 10 == 8:
                    data_list.append(i)
            else:
                if i % 10 == 9:
                    data_list.append(i)

        self.frames = self.frames[data_list]
        self.labels = self.labels[data_list]
        self.image_id = [self.image_id[i] for i in range(len(self.image_id)) if i in data_list]

    def __getitem__(self,index):
        have_read = False

        while not have_read:
            try:
                id = self.image_id[index]
                while not id+'.h5' in self.a_path_list:
                    index = (index + 1)%self.frames.shape[0]
                    id = self.image_id[index]

                # print('get item')
                a_data = h5.File(os.path.join(self.audio_path,id + '.h5'),'r')
                have_read = True
            except:
                index = (index + 1) % self.frames.shape[0]


        f_feature = self.frames[index]
        a_feature = np.array(a_data[id + '_audio'])

        # print('f_feature ' + str(f_feature.shape))
        # print('a_feature ' + str(a_feature.shape))

        return id, np.concatenate((f_feature,a_feature),axis=1).astype(np.float32),int(self.labels[index])


    def __len__(self):
        return self.frames.shape[0]

if __name__ == '__main__':
    ks_dataset = feature_dataset()
    source_trainloader = torch.utils.data.DataLoader(ks_dataset, batch_size=10, shuffle=False,
                                                     num_workers=1, drop_last=True)
    tmp_loader = iter(source_trainloader)
    for i in range(500):
        id,input,label = tmp_loader.next()
        print('input shape ' + str(input.shape))
    c = 1
