# -*- coding: UTF-8 -*-
from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import vggish_input
import vggish_params
import vggish_postprocess
import vggish_slim
import os
from moviepy.editor import *

import h5py
from multiprocessing import Process

label_names = ['普通','普通人群聚集','群体行为']

videos_dir = '../video'
# videos_dir = '/mnt/mmu/liuchang/people_video'

save_dir = '../save_data/'
# save_dir = '/mnt/mmu/liuchang/hywData/ksData_split/'


def gen_ind(fnum,ex_fnum = 300):
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


# inds = gen_ind(70,20)
# c = 1

# def padding_frame(frame):
#     fshape = frame.shape
#     if fshape[0] > fshape[1]:
#         pad = np.zeros((fshape[0],(fshape[0] - fshape[1])//2,3),dtype='int8')
#         frame = np.concatenate((pad,frame,pad),axis=1)
#     else:
#         pad = np.zeros(((fshape[0] - fshape[1])//2,fshape[1],3),dtype='int8')
#         frame = np.concatenate((pad,frame,pad),axis=0)
#
#     image = cv2.resize(frame,(224, 224))


def loc_ind(num,num_list):
    # if len(num_list) == 0:
    #     print('wrong')

    mid_ind = len(num_list) // 2
    if num == num_list[mid_ind]:
        return mid_ind
    elif num > num_list[mid_ind]:
        return mid_ind + loc_ind(num,num_list[mid_ind + 1:])
    else:
        return loc_ind(num,num_list[:mid_ind])

def save_cset_split(path_list,id_list,label_list,save_dir,c_set_ind):

    saved_data = os.listdir(save_dir)

    for i in range(len(path_list)):
        path = path_list[i]
        print('path ' + path)
        id = int(path.replace('.mp4', ''))

        if str(id) + '_a.h5' in saved_data:
            print('already saved')
            continue

        # id_ind = loc_ind(id, id_list)

        if id in id_list:
            id_ind = id_list.index(id)
            print('id_ind ' + str(id_ind))
        else:
            print('the id is not in label.txt')
            continue

        # print(videos_dir + ' ' + path)
        # if os.path.exists(videos_dir + '/' + path):
           # print('True')

        audio = extract_audio(videos_dir + '/' + path,wav_file=str(c_set_ind) + '.wav')
        if type(audio) == None:
            print(str(id) + ' failed')
            continue
        label = label_list[id_ind]

        f = h5py.File(save_dir + str(id) + "_a.h5", "w")
        try:
            f.create_dataset(str(id) + "_audio", data=audio)
            f.create_dataset(str(id) + "_label", data=label)
            f.close()
        except:
            os.system('rm ' + save_dir + str(id) + "_a.h5")
            print('some thing wrong !')


def save_split_audio():
    # video_dir = '/mnt/mmu/liuchang/people_video'
    v_path_list = os.listdir(videos_dir)

    num_cset = 500
    set_num = len(v_path_list) // num_cset
    rest_num = len(v_path_list) - set_num * num_cset

    id_list = []
    label_list = []

    with open('../labels.txt','r') as f:
        label_info = f.readlines()
        for line in label_info:
            items = line.split()
            try:
                label = items[3]
                label = label.replace('\n','')
                label = label.replace('\r','')
                if label in label_names:
                    label_list.append(label_names.index(label))
                    id_list.append(int(items[0]))
            except:
                print('ineffective')

    sort_inds = np.argsort(id_list)
    sort_inds = sort_inds.tolist()

    tmp_list = [id_list[i_ind] for i_ind in sort_inds]
    id_list = tmp_list
    tmp_list = [label_list[l_ind] for l_ind in sort_inds]
    label_list = tmp_list

    frame_data = []
    # label_data = None
    # save_dir = 'save_data/'
    # save_dir = '/mnt/mmu/liuchang/hywData/ksData_split/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('set num ' + str(set_num))

    for set_ind in range(set_num + 1):
        try:
            # save_cset_split(v_path_list[set_ind * num_cset:(set_ind + 1) * num_cset], id_list, label_list, save_dir,set_ind)
            if set_ind < set_num:
                p = Process(target=save_cset_split, args=(v_path_list[set_ind * num_cset:(set_ind + 1)* num_cset],id_list,label_list,save_dir,set_ind,))
            else:
                p = Process(target=save_cset_split, args=(v_path_list[set_ind * num_cset:],id_list,label_list,save_dir,set_ind,))
            p.start()
        except:
            print('set ind ' + str(set_ind) + ' goes wrong!' )


def extract_audio(video_file,wav_file):
  pca_params = '/home/hyw/y8_torch/vggish_pca_params.npz'
  checkpoint = '/home/hyw/y8_torch/audio_utils/vggish_model.ckpt'
  video = VideoFileClip(video_file)
  audio = video.audio
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
    postprocessed_batch = pproc.postprocess(embedding_batch)
    # print(postprocessed_batch)

    a_inds = gen_ind(postprocessed_batch.shape[0])

    ex_frames_list = [postprocessed_batch[ind].reshape(1,postprocessed_batch.shape[1]) for ind in a_inds]
    # ex_frames_list = ex_frames_list.astype('int8')
    return np.concatenate(ex_frames_list,axis=0).astype('int8')

if __name__ == '__main__':
  save_split_audio()