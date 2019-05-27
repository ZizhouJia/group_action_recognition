from __future__ import print_function

import numpy as np
from scipy.io import wavfile
import six
import tensorflow as tf

import audio_utils.vggish_input as vggish_input
import audio_utils.vggish_params as vggish_params
import audio_utils.vggish_postprocess as vggish_postprocess
import audio_utils.vggish_slim as vggish_slim
import cv2
import os
from moviepy.editor import *
import h5py
from multiprocessing import Process

# import sys
# reload(sys)
# sys.setdefaultencoding('utf-8')

pca_params = '/mnt/mmu/liuchang/youtube8_torch/audio_utils/vggish_pca_params.npz'
checkpoint = '/mnt/mmu/liuchang/youtube8_torch/audio_utils/vggish_model.ckpt'

# videos_dir = 'video'

normal = True

if normal:
    videos_dir = '/mnt/mmu/liuchang/action5_video'
else:
    videos_dir = '/mnt/mmu/liuchang/people_video'

# save_dir = 'save_data/'
if normal:
    save_dir = '/mnt/mmu/liuchang/hywData/ksData_normal/'
else:
    save_dir = '/mnt/mmu/liuchang/hywData/ksData_group/'
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


if not normal:
    wav_dir = 'wav_dir/'
    if not os.path.exists(wav_dir):
        os.mkdir(wav_dir)
else:
    wav_dir = ''


whe_audio = True
whe_video = True
sample_audio = False

label_names = ['普通','普通人群聚集','群体行为']

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

    return np.concatenate(frame_list,axis=0).astype('int8')


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

def save_audio_and_frame():
    # video_dir = '/mnt/mmu/liuchang/people_video'
    v_path_list = os.listdir(videos_dir)

    if normal:
        num_cset = 5000
    else:
        num_cset = 1000
    set_num = len(v_path_list) // num_cset
    rest_num = len(v_path_list) - set_num * num_cset

    id_list = []
    label_list = []

    with open('labels.txt','r',encoding='utf-8') as f:
        label_info = f.readlines()
        for line in label_info:
            items = line.split()
            # try:
            label = items[3]
            label = label.replace('\n','')
            label = label.replace('\r','')
            if label in label_names:
                label_list.append(label_names.index(label))
                id_list.append(int(items[0]))
            # except:
            #     print('ineffective')

    # sort_inds = np.argsort(id_list)
    # sort_inds = sort_inds.tolist()

    # tmp_list = [id_list[i_ind] for i_ind in sort_inds]
    # id_list = tmp_list
    # tmp_list = [label_list[l_ind] for l_ind in sort_inds]
    # label_list = tmp_list


    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('set num ' + str(set_num))

    for set_ind in range(set_num + 1):
        # try:
        #     save_cset_split(v_path_list[set_ind * num_cset:(set_ind + 1) * num_cset], id_list, label_list,set_ind)
            if set_ind < set_num:
                p = Process(target=save_cset_split, args=(v_path_list[set_ind * num_cset:(set_ind + 1)* num_cset],id_list,label_list,set_ind,))
            else:
                p = Process(target=save_cset_split, args=(v_path_list[set_ind * num_cset:],id_list,label_list,set_ind,))
            p.start()
        # except:
        #     print('set ind ' + str(set_ind) + ' goes wrong!' )

def save_cset_split(path_list,id_list,label_list,c_set_ind):

    saved_data = os.listdir(save_dir)

    for i in range(len(path_list)):
        path = path_list[i]
        print('path ' + path)
        id = int(path.replace('.mp4', ''))

        # id_ind = loc_ind(id, id_list)

        if not normal:
            if id in id_list:
                id_ind = id_list.index(id)
                print('id_ind ' + str(id_ind))
            else:
                print('the id is not in label.txt')
                continue

        if str(id) + '.h5' in saved_data:
            print('already saved')
            continue

        try:
            video = VideoFileClip(os.path.join(videos_dir,path))
        except:
            print(str(id) + ' fail to read')

        if whe_video:
            frames = extract_frame(video)
            if type(frames) == type(None):
                print(str(id) + ' failed')
                continue

        if whe_audio:
            audio = video.audio
            try:
                audio = extract_audio(audio,wav_dir + str(c_set_ind) + '.wav')
                if type(audio) == type(None):
                    print(str(id) + ' failed')
                    continue
            except:
                print('audio wrong !')
                continue

        # print('start to save')

        if normal:
            label = 0
        else:
            label = label_list[id_ind]
        f = h5py.File(save_dir + str(id) + ".h5", "w")
        try:
            # print(save_dir + str(id) + ".h5")
            f.create_dataset(str(id) + "_label", data=label)
            if whe_video:
                f.create_dataset(str(id) + "_frames", data=frames)
                print('frame shape ' + str(frames.shape))
            if whe_audio:
                f.create_dataset(str(id) + "_audio", data=audio)
                print('audio shape ' + str(audio.shape))
            f.close()
            # print('save finished')
        except:
            os.system('rm ' + save_dir + str(id) + ".h5")
            print('some thing wrong !')


# extract_frame()
if __name__ == "__main__":
    save_audio_and_frame()