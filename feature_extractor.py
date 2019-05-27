import os
from torchvision import transforms
import h5py
import numpy as np
import video_transforms
import model.inception_v3 as inception_v3
from multiprocessing import Process

test = False

if not test:
    normal_dir = '/mnt/mmu/liuchang/hywData/ksData_normal/'
    people_dir = '/mnt/mmu/liuchang/hywData/ksData_group/'

    save_dir = '/mnt/mmu/liuchang/hywData/ksData_gfeature/'
else:
    normal_dir = 'ksData_normal/'
    people_dir = 'ksData_normal/'
    save_dir = 'tmp_save/'

if not os.path.exists(save_dir):
    os.mkdir(save_dir)

video_t=transforms.Compose([
                video_transforms.center_crop(),
                video_transforms.resize(299),
                video_transforms.to_tensor(),
                video_transforms.normalize()
            ])

def ex_feature():
    ids = []

    normal_ids = os.listdir(normal_dir)
    people_ids = os.listdir(people_dir)

    # if len(normal_ids) > 15000:
    #     normal_ids = normal_ids[:15000]

    ids.extend(normal_ids)
    ids.extend(people_ids)

    normal_paths = [os.path.join(normal_dir,vid) for vid in normal_ids]
    people_paths = [os.path.join(people_dir,vid) for vid in people_ids]



    video_paths = []
    video_paths.extend(normal_paths)
    video_paths.extend(people_paths)

    if not test:
        ex_model = inception_v3.inception_v3(pca_dir = None)
        ex_model = ex_model.cuda()
    else:
        ex_model = None

    num_cset = 2000
    process_num = len(video_paths) //num_cset

    for set_ind in range(process_num + 1):
        save_function(
        video_paths[set_ind * num_cset:(set_ind + 1) * num_cset], ids[set_ind * num_cset:(set_ind + 1) * num_cset],
        ex_model)

        # if set_ind < process_num:
        #     p = Process(target=save_function,
        #                 args=(video_paths[set_ind * num_cset:(set_ind + 1) * num_cset], ids[set_ind * num_cset:(set_ind + 1) * num_cset],ex_model,))
        # else:
        #     p = Process(target=save_function,
        #                 args=(video_paths[set_ind * num_cset:],ids[set_ind * num_cset:], ex_model,))
        #
        # p.start()

def save_function(video_paths,ids,ex_model):
    for i,path in enumerate(video_paths):
            try:
                vid = ids[i].replace('.h5','')
                if os.path.exists(save_dir + str(vid) + ".h5"):
                    print(vid + ' exists')
                    continue

                f = h5py.File(path)
                frames = f[vid + '_frames']
                label = f[vid + '_label']
                audio = f[vid + '_audio']


                frames = np.array(frames).astype('float32')
                label = np.array(label)
                audio = np.array(audio)

                f.close()

                # for i in range(frames.shape[0]):
                frames = video_t(frames)
                frames = frames.cuda()
                f_shape = frames.shape
                features = ex_model(frames)
                features = features.view(f_shape[0],2048)
                features = features.detach().cpu().numpy()

                wf = h5py.File(save_dir + str(vid) + ".h5", "w")

                    # print(save_dir + str(id) + ".h5")
                wf.create_dataset(str(vid) + "_label", data=label)
                wf.create_dataset(str(vid) + "_features", data=features)
                # print('feature shape ' + str(features.shape))
                wf.create_dataset(str(vid) + "_audio", data=audio)
                # print('audio shape ' + str(audio.shape))
                print('vid ' + vid + ' is finished')
                wf.close()
            except:
                os.system('rm ' + path)
                print('wrong')
            # if os.path.exists(save_dir + str(vid) + ".h5"):
            #     os.system('rm' + save_dir + str(vid) + ".h5")


if __name__ == '__main__':
    pca_dir = '/mnt/mmu/liuchang/youtube8_torch/yt8m_pca'
    # pca_dir = None

    split_ind = 0
    cuda_device = 'cuda:' + str(split_ind)
    ex_model = model.inception_v3.inception_v3(pca_dir=pca_dir)
    ex_model = ex_model.cuda(cuda_device)
    # ex_model = None
    if type(pca_dir) == type(None):
        a_zero_feature = False
    else:
        a_zero_feature = True

    # train_dataset = dataset.end2end_dataset.end2end_dataset(data_type="train", ex_model=ex_model,cuda_device=cuda_device)
    train_dataset = dataset.end2end_dataset.end2end_dataset(data_type="train", ex_model=ex_model,
                                                            cuda_device=cuda_device,
                                                            pca_dir=pca_dir, a_zero_feature=a_zero_feature, split=8,
                                                            split_ind=split_ind)

    source_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=50, shuffle=True, drop_last=True)
    # tmp_loader = iter(source_trainloader)
    for i, id, input, label in enumerate(source_trainloader):
        pass
        # id,input,label = tmp_loader.next()
        # print('input shape ' + str(input.shape))