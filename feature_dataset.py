import torch.utils.data as Data
import torch
import h5py as h5
import os
import numpy as np

class feature_dataset(Data.Dataset):
    def __init__(self,frame_path ='/mnt/mmu/liuchang/kuaishou_feature.h5'):
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

    def __getitem__(self,index):
        id = self.image_id[index]
        while not id+'.h5' in self.a_path_list:
            index = (index + 1)%self.frames.shape[0]
            id = self.image_id[index]

        # print('get item')
        a_data = h5.File(os.path.join(self.audio_path,id + '.h5'),'r')
        f_feature = self.frames[index]
        a_feature = np.array(a_data[id + '_audio'])

        # print('f_feature ' + str(f_feature.shape))
        # print('a_feature ' + str(a_feature.shape))

        return id, np.concatenate((f_feature,a_feature),axis=1),self.labels[index]


    def __len__(self):
        return self.frames.shape[0]

if __name__ == '__main__':
    ks_dataset = feature_dataset()
    source_trainloader = torch.utils.data.DataLoader(ks_dataset, batch_size=1, shuffle=False,
                                                     num_workers=1, drop_last=True)
    tmp_loader = iter(source_trainloader)
    for i in range(10):
        id,input,label = tmp_loader.next()
    print('input shape ' + str(input.shape))
    c = 1