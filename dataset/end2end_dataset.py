import torch.utils.data as Data
import torch
import numpy as np
import six
import tensorflow as tf
import sys
sys.path.append("..")
import model.inception_v3 as inception_v3
import dataset
import model
import cv2
import os
import h5py
from multiprocessing import Process

import random
import copy

from torchvision import transforms
import video_transforms
import time

test = False
(major_ver, _,_) = (cv2.__version__).split('.')

if not test:
    pca_params = '/mnt/mmu/liuchang/youtube8_torch/audio_utils/vggish_pca_params.npz'
    checkpoint = '/mnt/mmu/liuchang/youtube8_torch/audio_utils/vggish_model.ckpt'

    normal_dir = '/mnt/mmu/liuchang/action5_video'
    people_dir = '/mnt/mmu/liuchang/people_video'

    wrong_dir = '/mnt/mmu/liuchang/wrong_video'
    if not os.path.exists(wrong_dir):
        os.mkdir(wrong_dir)

    # feature_dir = '/mnt/mmu/liuchang/hywData/gfeature_1024_pca'
    # feature_dir = '/mnt/mmu/liuchang/hywData/gfeature_2048'
    label_dir = '/mnt/mmu/liuchang/youtube8_torch/labels.txt'

    pca_dir = '/mnt/mmu/liuchang/youtube8_torch/yt8m_pca'

    img_dir = '/mnt/mmu/liuchang/hywData/video_pic'

    if not os.path.exists(img_dir):
        os.mkdir(img_dir)
else:
    pca_params = '/home/hyw/y8_torch/audio_utils/vggish_pca_params.npz'
    checkpoint = '/home/hyw/y8_torch/audio_utils/vggish_model.ckpt'

    normal_dir = '/home/hyw/y8_torch/video'
    people_dir = '/home/hyw/y8_torch/video'

    # feature_dir = '/home/hyw/y8_torch/gfeature_2048'
    label_dir = '/home/hyw/y8_torch/labels.txt'

    pca_dir = '/home/hyw/y8_torch/yt8m_pca'

wav_dir = '/mnt/mmu/liuchang/youtube8_torch/wav_dir/'


whe_audio = False
whe_video = True
sample_audio = False

label_names = ['普通','普通人群聚集','群体行为']

class end2end_dataset(Data.Dataset):
    def __init__(self,data_type = 'train',pca_dir = None,ex_model = None,a_zero_feature = False,device_ind = 0,
                 all_saved = False,with_yt8m = False,split = 0,split_ind = 0):
        self.people_paths,self.people_labels,self.people_ids,\
        self.normal_paths,self.normal_labels,self.normal_ids = match_path_with_label()
        self.pca_dir = pca_dir
        self.a_zero_feature = a_zero_feature
        self.cuda_device = torch.device('cuda:' + str(device_ind))
        self.read_count = 0

        if test:
            feature_dir = '/home/hyw/group_action_recognition/gfeature_2048'
            if os.path.exists(feature_dir):
                os.system('rm -r ' + feature_dir)
            os.mkdir(feature_dir)
        else:
            if type(pca_dir) == type(None):
                feature_dir = '/mnt/mmu/liuchang/hywData/gfeature_2048'
            else:
                feature_dir = '/mnt/mmu/liuchang/hywData/gfeature_1024_pca'

        if not os.path.exists(feature_dir):
            os.mkdir(feature_dir)

        self.feature_dir = feature_dir
        # print('feature dir ' + self.feature_dir)

        random.seed(9)
        self.video_t = transforms.Compose([
            video_transforms.center_crop(),
            video_transforms.resize(299),
            video_transforms.to_tensor(),
            video_transforms.normalize()
        ])

        for i in range(2):
            if i == 0:
                tmp_labels = self.people_labels
                tmp_paths = self.people_paths
                tmp_ids = self.people_ids
            else:
                tmp_labels = self.normal_labels
                tmp_paths = self.normal_paths
                tmp_ids = self.normal_ids

            tmp_inds = list(range(len(tmp_paths)))
            random.shuffle(tmp_inds)
            tmp_labels = [tmp_labels[ind] for ind in tmp_inds]
            tmp_paths = [tmp_paths[ind] for ind in tmp_inds]
            tmp_ids = [tmp_ids[ind] for ind in tmp_inds]


            tmp_train_num = len(tmp_labels) * 0.8
            tmp_train_num = int(tmp_train_num)
            tmp_valid_num = len(tmp_labels) * 0.1
            tmp_valid_num = int(tmp_valid_num)

            if data_type == 'train':
                print('start to train')
                tmp_paths = tmp_paths[:tmp_train_num]
                tmp_labels = tmp_labels[:tmp_train_num]
                tmp_ids = tmp_ids[:tmp_train_num]
            elif data_type == 'valid':
                print('begin to validate')
                tmp_paths = tmp_paths[tmp_train_num : tmp_train_num + tmp_valid_num]
                tmp_labels = tmp_labels[tmp_train_num : tmp_train_num + tmp_valid_num]
                tmp_ids = tmp_ids[tmp_train_num : tmp_train_num + tmp_valid_num]
            elif data_type == 'test':
                tmp_paths = tmp_paths[tmp_train_num + tmp_valid_num:]
                tmp_labels = tmp_labels[tmp_train_num + tmp_valid_num:]
                tmp_ids = tmp_ids[tmp_train_num + tmp_valid_num:]

            if i == 0:
                self.people_labels = tmp_labels
                self.people_paths = tmp_paths
                self.people_ids = tmp_ids
            else:
                self.normal_labels = tmp_labels
                self.normal_paths = tmp_paths
                self.normal_ids = tmp_ids


        if data_type == 'train':

            # if len(self.people_paths) > len(self.normal_paths):
            #     times = len(self.people_paths) // len(self.normal_paths)
            # else:
            #     times = len(self.normal_paths) // len(self.people_paths)
           if not with_yt8m:
               or_paths = copy.deepcopy(self.people_paths)
               or_labels = copy.deepcopy(self.people_labels)
               or_ids = copy.deepcopy(self.people_ids)
               for i in range(6):
                    # print('people copy')
                    self.people_paths.extend(or_paths)
                    self.people_labels.extend(or_labels)
                    self.people_ids.extend(or_ids)
           else:
               or_paths = copy.deepcopy(self.normal_paths)
               or_labels = copy.deepcopy(self.normal_labels)
               or_ids = copy.deepcopy(self.normal_ids)
               for i in range(2):
                   self.normal_paths.extend(copy.deepcopy(or_paths))
                   self.normal_labels.extend(copy.deepcopy(or_labels))
                   self.normal_ids.extend(copy.deepcopy(or_ids))


        self.people_paths.extend(self.normal_paths)
        self.paths = self.people_paths
        self.people_labels.extend(self.normal_labels)
        self.labels = self.people_labels
        self.people_ids.extend(self.normal_ids)
        self.ids = self.people_ids
        self.all_saved = all_saved
        self.data_type = data_type



        if not all_saved:

            if type(ex_model) == type(None):
                self.ex_model = inception_v3.inception_v3(pca_dir=self.pca_dir,device_ind=device_ind)
                self.ex_model = self.ex_model.cuda(self.cuda_device)
            else:
                self.ex_model = ex_model
        else:
            self.ex_model = None
            tmp_paths = []
            tmp_ids = []
            tmp_labels = []

            if type(pca_dir) == type(None):
                paths_file = data_type + '_paths2048'
            else:
                paths_file = data_type + '_paths'

            if not os.path.exists(paths_file):
            # if True:
                f = open(paths_file,'w')

                saved_fs = os.listdir(self.feature_dir)
                for i,path in enumerate(self.paths):
                    id = self.ids[i]
                    label = self.labels[i]
                    f_name = id + '.h5'

                    if f_name in saved_fs:
                        tmp_paths.append(os.path.join(self.feature_dir,f_name))
                        tmp_ids.append(id)
                        tmp_labels.append(label)
                        f.write(str(id) + ' ' + os.path.join(self.feature_dir,f_name) + ' ' + str(int(label)) + '\n')
                f.close()


            else:
                f = open(paths_file,'r')
                lines = f.readlines()
                for line in lines:
                    line.replace('\n','')
                    items = line.split()
                    id = items[0]
                    path = items[1]
                    label = int(items[2])

                    tmp_ids.append(id)
                    tmp_paths.append(path)
                    tmp_labels.append(label)

            self.paths = tmp_paths
            self.labels = tmp_labels
            self.ids = tmp_ids


        if data_type == 'train' and with_yt8m:
            yt_dir1 = '/mnt/mmu/liuchang/hywData/yt8m/frame/train_group'
            yt_dir2 = '/mnt/mmu/liuchang/hywData/yt8m/frame/validate_group'
            yt_paths = []
            yt_labels = []
            yt_ids = []
            yt_dirs = [yt_dir1,yt_dir2]
            for yt_dir in yt_dirs:
                names = os.listdir(yt_dir)
                for name in names:
                    yt_paths.append(os.path.join(yt_dir,name))
                    yt_labels.append(1)
                    yt_ids.append(name.replace('.h5',''))
            self.paths.extend(yt_paths)
            self.labels.extend(yt_labels)
            self.ids.extend(yt_ids)

        if split > 0:
            s_num = len(self.paths) // split
            if split_ind == split - 1:
                self.paths = self.paths[split_ind * s_num:]
                self.labels = self.labels[split_ind * s_num:]
                self.ids = self.ids[split_ind * s_num:]
            else:
                self.paths = self.paths[split_ind * s_num:(split_ind + 1) * s_num]
                self.labels = self.labels[split_ind * s_num:(split_ind + 1) * s_num]
                self.ids = self.ids[split_ind * s_num:(split_ind + 1) * s_num]

        label_nums = [0 for _ in range(max(self.labels) + 1)]
        for label in self.labels:
            label_nums[int(label)] += 1

        print('data_type ' + data_type)
        for i in range(len(label_nums)):
            print('label ' + str(i) + ' ' + str(label_nums[i]))

        print('model type ' + str(type(self.ex_model)))

    def __getitem__(self,index):
        if self.read_count % 1000 == 0:
            # self.read_count = 0
            print('read ' + str(self.read_count))

        if index == len(self.paths) - 1:
            self.read_count = 0

        w_count = 0
        while True:
            w_count += 1
            if w_count >= 1000:
                print('dead loop')


            feature_list = []
            path = self.paths[index]

            # from youtube feature
            if self.data_type != 'all':

                if path.startswith('/mnt/mmu/liuchang/hywData/yt8m'):
                    try:
                        f = h5py.File(path,'r')
                    except:
                        index = (index + 1) % len(self.paths)
                        # print('can not read')
                        continue
                    feature = f['feature']
                    label = 2

                    feature = np.array(feature).astype('float32')
                    self.read_count += 1
                    return '',feature,label

                # from feature
                with_feature = False
                try:
                    f = h5py.File(path,'r')
                    with_feature = True
                except:
                    if self.all_saved:
                        index = (index + 1) % len(self.paths)
                        continue

                if with_feature:
                    label = f['label']
                    label = np.array(label).astype('uint8').item()
                    if whe_video:
                        v_feature = f['v_feature']
                        v_feature = np.array(v_feature)
                        if v_feature.shape[0] <= 3:
                            index = (index + 1) % len(self.paths)
                            continue
                        v_feature = self.resize_frames(v_feature)
                        feature_list.append(v_feature)
                    if whe_audio:
                        v_feature = f['a_feature']
                        v_feature = np.array(v_feature)
                        v_feature = self.resize_frames(v_feature)
                        feature_list.append(v_feature)
                    elif self.a_zero_feature:
                        a_features = np.zeros((feature_list[0].shape[0], 128)).astype('float32')
                        feature_list.append(a_features)
                    f.close()

                    if len(feature_list) > 1:
                        feature = np.concatenate(feature_list, axis=-1)
                    else:
                        feature = feature_list[0]

                    self.read_count += 1
                    # if label >= 2:
                    #     label = 1

                    # print('feature shape ' + str(feature.shape))
                    return '', feature, label



            # from video

            id = path.split('/')[-1].replace('.mp4', '')
            id = id.replace('.h5', '')

            label = self.labels[index]
            video = cv2.VideoCapture(self.paths[index])

            if whe_video:
                frames = extract_frame_cv(video)

                if type(frames) == type(None):
                    os.system('mv ' + path + ' ' + wrong_dir)
                    # np.save(os.path.join(wrong_dir,id + ".npy"), np.array(label))
                    index = (index + 1) % len(self.paths)
                    print('move wrong 1')
                    continue

                # tmp_dir = os.path.join(img_dir,id)
                # if not os.path.exists(tmp_dir):
                #     os.mkdir(tmp_dir)
                # for f_count in range(frames.shape[0]):
                #     img = frames[f_count]
                #     cv2.imwrite(tmp_dir + '/' + str(f_count) + '.jpg', img)
                #
                # print('write finished')
                # return None,None,None

                if frames.shape[0] <= 3:
                    os.system('mv ' + path + ' ' + wrong_dir)
                    # np.save(os.path.join(wrong_dir, id + ".npy"), np.array(label))
                    index = (index + 1) % len(self.paths)
                    print('move wrong 2')
                    # print(str(id) + '  short shape')
                    continue
                # frames =frames.astype('float32')
                frames =frames.astype('uint8')

                try:
                # print('frames shape ' + str(frames.shape))
                    frames = self.video_t(frames)
                except:
                    os.system('mv ' + path + ' ' + wrong_dir)
                    np.save(os.path.join(wrong_dir, id + ".npy"), np.array(label))
                    index = (index + 1) % len(self.paths)
                    # print('move wrong 3')
                    # print('transform failed')
                    continue

                tmp_size = 20
                # if frames.shape[0] > tmp_size:
                ex_time = frames.shape[0] // tmp_size + 1
                rest_num =frames.shape[0] % tmp_size
                or_frames = frames
                v_feature_list = []

                for ex_ind in range(ex_time):
                    if ex_ind == ex_time - 1:
                        if rest_num == 0:
                            continue
                        else:
                            frames = or_frames[ex_ind*tmp_size:]
                    else:
                        frames = or_frames[ex_ind*tmp_size:(ex_ind + 1)*tmp_size]

                    frames = frames.cuda(self.cuda_device)

                    # frames = frames.cuda()
                    f_shape = frames.shape
                    # print('f_shape ' + str(f_shape))
                    features = self.ex_model(frames)

                    if type(self.pca_dir) == type(None):
                        features = features.view(f_shape[0], 2048)
                    else:
                        features = features.view(f_shape[0],1024)
                    features = features.detach().cpu().numpy()

                    v_feature_list.append(features)

                features = np.concatenate(v_feature_list,axis=0)
                features = self.resize_frames(features)
                feature_list.append(features)

            if whe_audio:
                video = VideoFileClip(self.paths[index])
                audio = video.audio
                # try:
                audio = extract_audio(audio, os.path.join(wav_dir, 'tmp_audio.wav'))
                if type(audio) == type(None):
                    print(str(id) + ' failed')
                    index = (index + 1) % len(self.paths)
                    continue
                # except:
                #     print('audio wrong !')
                #     continue
                features = self.resize_frames(audio)
                feature_list.append(features)
            elif self.a_zero_feature:
                a_features = np.zeros((feature_list[0].shape[0],128)).astype('float32')
                feature_list.append(a_features)

            f = h5py.File(os.path.join(self.feature_dir ,str(id) + ".h5"), "w")
                # print(save_dir + str(id) + ".h5")
            f.create_dataset("label", data=label)
            if whe_video:
                f.create_dataset("v_feature", data=feature_list[0])
                # print('frame shape ' + str(frames.shape))
            if whe_audio:
                f.create_dataset("_audio", data=audio)
                # print('audio shape ' + str(audio.shape))
            f.close()

            if len(feature_list) > 1:
                feature = np.concatenate(feature_list, axis=-1)
            else:
                feature = feature_list[0]
            label = self.labels[index]

            # if label >= 2:
            #     label = 1
            # print('read finished')
            self.read_count += 1
            return id, feature, label

    def __len__(self):
        # if self.data_type == 'train':
        #     return 10

        # else:
        #     return 10

        return len(self.paths)

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

def from_all_saved(index):
    pass

def extract_audio(audio,wav_file = 'tmp.wav'):

  audio.write_audiofile(wav_file)

  # In this simple example, we run the examples from a single audio file through
  # the model. If none is provided, we generate a synthetic input.
  if wav_file:
    wav_file = wav_file
  else:
    # Write a WAV of a sine wav into an in-memory file object.
    num_secs = 5
    freq = 1000
    sr = 44100
    t = np.linspace(0, num_secs, int(num_secs * sr))
    x = np.sin(2 * np.pi * freq * t)
    # Convert to signed 16-bit samples.
    samples = np.clip(x * 32768, -32768, 32767).astype(np.int16)
    wav_file = six.BytesIO()
    wavfile.write(wav_file, sr, samples)
    wav_file.seek(0)

  examples_batch = vggish_input.wavfile_to_examples(wav_file)
  # print(examples_batch)

  # Prepare a postprocessor to munge the model embeddings.
  pproc = vggish_postprocess.Postprocessor(pca_params)

  with tf.Graph().as_default(), tf.Session() as sess:
    # Define the model in inference mode, load the checkpoint, and
    # locate input and output tensors.
    vggish_slim.define_vggish_slim(training=False)
    vggish_slim.load_vggish_slim_checkpoint(sess, checkpoint)
    features_tensor = sess.graph.get_tensor_by_name(
        vggish_params.INPUT_TENSOR_NAME)
    embedding_tensor = sess.graph.get_tensor_by_name(
        vggish_params.OUTPUT_TENSOR_NAME)

    # Run inference and postprocessing.
    [embedding_batch] = sess.run([embedding_tensor],
                                 feed_dict={features_tensor: examples_batch})
    postprocessed_batch = pproc.postprocess(embedding_batch).astype('int8')

    if sample_audio:
        audio_dim = postprocessed_batch.shape[1]
        a_inds = gen_ind(postprocessed_batch.shape[0])
        batch_list = [postprocessed_batch[ind].reshape((1,audio_dim)) for ind in a_inds]
        postprocessed_batch = np.concatenate(batch_list,axis=0)

    return postprocessed_batch.astype('int8')

def extract_frame(video):
    # video = VideoFileClip(video_path)

    f_size = video.size
    m_size = float(min(f_size))
    ratio = min(300.,m_size)/m_size
    n_size = (int(f_size[0] * ratio),int(f_size[1] * ratio))
    n_shape = (1,n_size[1],n_size[0],3)
    cur_time = 0
    frame_list = []

    while cur_time < video.end:
        try:
            frame = video.get_frame(cur_time)
            frame = cv2.resize(frame, n_size)
            frame = frame.reshape(n_shape)
            frame_list.append(frame)
            cur_time += 1
        except:
            cur_time += 0.1

    if len(frame_list) == 0:
        return None

    return np.concatenate(frame_list,axis=0).astype('int16')


    #
    # for i in range():
    #     # print('f_inds ' + str(f_inds))
    #     try:
    #         while start < ex_fnum and f_inds[start] == i:
    #             f_matrix[start] = frame
    #             start += 1
    #     except:
    #         print('f inds' + str(f_inds))
    #         print('f num ' + str(total_fnum))
    #         print('len f inds ' + str(len(f_inds)))
    #         print('success ' + str(success))

    #         # cv2.imwrite(save_dir + '/' + str(start) + '.jpg',frame)
    #

def extract_frame_cv(video):
    if int(major_ver) < 3:
        fps = video.get(cv2.cv.CV_CAP_PROP_FPS)
    else:
        fps = video.get(cv2.CAP_PROP_FPS)

    fps = int(fps)

    if fps <= 3:
        return None

    frame_list = []
    nframes = video.get(7)

    if nframes == 0:
        return None

    start_frames = 0

    while start_frames < nframes:
        readed,frame = video.read(start_frames)

        if not readed:
            start_frames += fps
            continue

        f_shape = list(frame.shape)
        nf_shape = [1]
        nf_shape.extend(f_shape)
        frame = frame.reshape(nf_shape)
        frame_list.append(frame)
        start_frames += fps

    if len(frame_list) == 0:
        return None

    frames = np.concatenate(frame_list,axis=0).astype('int16')
    # frames = np.concatenate(frame_list,axis=0)

    # img_dir = str(frames.shape[0]) + 'before_ex'
    # if not os.path.exists(img_dir):
    #     os.mkdir(img_dir)
    # for f_count in range(frames.shape[0]):
    #     img = frames[i]
    #     cv2.imwrite(img_dir + '/' + str(f_count) + '.jpg', img)
    #     f_count += 1

    return frames

def match_path_with_label():
    # video_dir = '/mnt/mmu/liuchang/people_video'
    videos_dir = people_dir
    path_list = os.listdir(videos_dir)

    id_list = []
    label_list = []

    people_paths = []
    people_labels = []
    people_ids = []

    normal_paths = []
    normal_labels = []
    normal_ids = []

    with open(label_dir,'r',encoding='utf-8') as f:
        label_info = f.readlines()
        for line in label_info:
            items = line.split()
            # try:
            label = items[3]
            label = label.replace('\n','')
            label = label.replace('\r','')
            if label in label_names:
                label_list.append(label_names.index(label))
                id_list.append(items[0])


    for i in range(len(path_list)):
        path = path_list[i]
        id = path.replace('.mp4', '')
        try:
            id_ind = id_list.index(id)
        except:
            continue

        people_paths.append(os.path.join(people_dir,path))
        people_labels.append(label_list[id_ind])
        people_ids.append(id)

    path_list = os.listdir(normal_dir)
    for i in range(len(path_list)):
        path = path_list[i]
        id = path.replace('.mp4', '')
        normal_paths.append(os.path.join(normal_dir,path))
        normal_labels.append(0)
        normal_ids.append(id)

    return people_paths,people_labels,people_ids,normal_paths,normal_labels,normal_ids

def ex_process(split_ind,split = 0):
    # pca_dir = '/mnt/mmu/liuchang/youtube8_torch/yt8m_pca'
    # pca_dir = None

    ex_model = model.inception_v3.inception_v3(pca_dir=pca_dir,device_ind=split_ind)
    ex_model = ex_model.cuda('cuda:' + str(split_ind))
    # ex_model = None
    if type(pca_dir) == type(None):
        a_zero_feature = False
    else:
        a_zero_feature = True

    # train_dataset = dataset.end2end_dataset.end2end_dataset(data_type="train", ex_model=ex_model,cuda_device=cuda_device)
    train_dataset = dataset.end2end_dataset.end2end_dataset(data_type="all", ex_model=ex_model,
                                                            device_ind=split_ind,
                                                            pca_dir=pca_dir, a_zero_feature=a_zero_feature,split = split ,split_ind = split_ind)

    source_trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=100, shuffle=False,drop_last=True)
    tmp_loader = iter(source_trainloader)
    while True:
    # for i,id,input,label in enumerate(source_trainloader):
    #     pass
        id,input,label = tmp_loader.next()
        # dbof = model.DbofModel.DbofModel(pretrain=False)
        # dbof = dbof.cuda()
        # input = input.cuda()
        # d_feature = dbof(input)
        # x = 1

if __name__ == '__main__':
    # ex_process(split_ind = 1)
    split = 8
    for split_ind in range(8):
        p = Process(target=ex_process,args=(split_ind,split,))
        p.start()
        # p.join()
        # print('input shape ' + str(input.shape))
