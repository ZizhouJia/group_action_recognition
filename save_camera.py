# -*- coding: UTF-8 -*-
import os
import cv2
import numpy as np
import h5py
import time
import json
from multiprocessing import Process
import sys

label_names = ['普通','普通人群聚集','群体行为']

# videos_dir = 'video'
videos_dir = '/mnt/mmu/liuchang/people_video'

# save_dir = 'save_data/'
save_dir = '/mnt/mmu/liuchang/hywData/ksData_split/'


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


def extract_frame(video_path = 'tmovie.mp4',ex_fnum= 300):
    camera=cv2.VideoCapture(video_path)

    # wid = int(camera.get(3))
    # hei = int(camera.get(4))
    # framerate = int(camera.get(5))
    total_fnum =int(camera.get(7))
    print('total_fnum ' + str(total_fnum))
    f_inds = gen_ind(total_fnum,ex_fnum)

    f_matrix = None

    # save_dir = video_path.split('.')[0]
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # for i in range(ex_fnum):
    #     # print('i ' + str(i))
    #
    #     camera.set(cv2.CAP_PROP_POS_FRAMES, f_inds[i])
    #     _, frame=camera.read()
    #     if i == 0:
    #         f_matrix = np.zeros((ex_fnum,frame.shape[0],frame.shape[1],frame.shape[2]),dtype='int8')
    #
    #     # print('i ' + str(i))
    #     # print('1 ' + str(i) + ' time ' + str(time.time() - start_time))
    #     f_matrix[i] = frame

        # cv2.imwrite(save_dir + '/' + str(i) + '.jpg', frame)

    start = 0
    for i in range(total_fnum):
        # print('i ' + str(i))
        success, frame=camera.read()


        # print('i ' + str(i))
        if i == 0:
            f_matrix = np.zeros((ex_fnum,frame.shape[0],frame.shape[1],frame.shape[2]),dtype='int8')

        # print('f_inds ' + str(f_inds))
        try:
            while start < ex_fnum and f_inds[start] == i:
                f_matrix[start] = frame
                start += 1
        except:
            print('f inds' + str(f_inds))
            print('f num ' + str(total_fnum))
            print('len f inds ' + str(len(f_inds)))
            print('success ' + str(success))

    #         # cv2.imwrite(save_dir + '/' + str(start) + '.jpg',frame)
    #

    return f_matrix


def extract_frame_with_false(video_path='tmovie.mp4', ex_fnum=300):
    camera = cv2.VideoCapture(video_path)

    # wid = int(camera.get(3))
    # hei = int(camera.get(4))
    # framerate = int(camera.get(5))
    total_fnum = int(camera.get(7))
    print('total_fnum ' + str(total_fnum))

    # save_dir = video_path.split('.')[0]
    # if not os.path.exists(save_dir):
    #     os.mkdir(save_dir)
    # for i in range(ex_fnum):
    #     # print('i ' + str(i))
    #
    #     camera.set(cv2.CAP_PROP_POS_FRAMES, f_inds[i])
    #     _, frame=camera.read()
    #     if i == 0:
    #         f_matrix = np.zeros((ex_fnum,frame.shape[0],frame.shape[1],frame.shape[2]),dtype='int8')
    #
    #     # print('i ' + str(i))
    #     # print('1 ' + str(i) + ' time ' + str(time.time() - start_time))
    #     f_matrix[i] = frame

    # cv2.imwrite(save_dir + '/' + str(i) + '.jpg', frame)


    suc_frames = []
    suc_inds = []
    for i in range(total_fnum):
        # print('i ' + str(i))
        success, frame = camera.read()
        if success:
            suc_frames.append(frame)
            suc_inds.append(i)

    if len(suc_frames) == 0:
        return None

    suc_num = len(suc_inds)
    print('suc num' + str(suc_num))
    # f_shape = suc_frames[0].shape
    # f_matrix = np.zeros((ex_fnum, f_shape[0], f_shape[1], f_shape[2]), dtype='int8')

    f_inds = gen_ind(suc_num)
        # print('i ' + str(i))
    f_shape = suc_frames[0].shape
    ex_frames_list = [suc_frames[ind].reshape(1,f_shape[0],f_shape[1],f_shape[2]) for ind in f_inds]
    # ex_frames_list = ex_frames_list.astype('int8')
    return np.concatenate(ex_frames_list,axis=0).astype('int8')


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

def save_cset(path_list,id_list,label_list,save_dir,c_set_ind):
    f = h5py.File(save_dir + "data" + str(c_set_ind) + ".h5", "w")
    s_id_list = []

    for i in range(len(path_list)):
        path = path_list[i]
        print('path ' + path)
        id = int(path.replace('.mp4', ''))

        # id_ind = loc_ind(id, id_list)

        if id in id_list:
            id_ind = id_list.index(id)
            print('id_ind ' + str(id_ind))
        else:
            print('some missing')
            continue

        # print(videos_dir + ' ' + path)
        # if os.path.exists(videos_dir + '/' + path):
           # print('True')

        frames = extract_frame_with_false(videos_dir + '/' + path)
        if type(frames) == type(None):
            print(str(id) + ' failed')
        label = label_list[id_ind]

        try:
            s_id_list.append(id)
            f.create_dataset(str(id) + "_frames", data=frames)
            f.create_dataset(str(id) + "_label", data=label)
        except:

            print('some thing wrong !')
    f.create_dataset('id', data=s_id_list)
    f.close()

def save_cset_split(path_list,id_list,label_list,save_dir,c_set_ind):

    saved_data = os.listdir(save_dir)

    for i in range(len(path_list)):
        path = path_list[i]
        print('path ' + path)
        id = int(path.replace('.mp4', ''))

        if str(id) + '.h5' in saved_data:
            print('already saved')
            continue

        # id_ind = loc_ind(id, id_list)

        if id in id_list:
            id_ind = id_list.index(id)
            print('id_ind ' + str(id_ind))
        else:
            print('some missing')
            continue

        # print(videos_dir + ' ' + path)
        # if os.path.exists(videos_dir + '/' + path):
           # print('True')

        frames = extract_frame_with_false(videos_dir + '/' + path)
        if type(frames) == type(None):
            print(str(id) + ' failed')
        label = label_list[id_ind]

        f = h5py.File(save_dir + str(id) + ".h5", "w")
        try:
            f.create_dataset(str(id) + "_frames", data=frames)
            f.create_dataset(str(id) + "_label", data=label)
            f.close()
        except:
            os.system('rm ' + save_dir + str(id) + ".h5")
            print('some thing wrong !')


def save_frames():
    # video_dir = '/mnt/mmu/liuchang/people_video'
    v_path_list = os.listdir(videos_dir)

    num_cset = 500
    set_num = len(v_path_list) // num_cset
    rest_num = len(v_path_list) - set_num * num_cset

    id_list = []
    label_list = []

    with open('labels.txt','r') as f:
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

    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    print('set num ' + str(set_num))

    for set_ind in range(set_num + 1):
        # save_cset(v_path_list[set_ind * num_cset:(set_ind + 1) * num_cset], id_list, label_list, save_dir,set_ind)
        try:
            if set_ind < set_num:
                p = Process(target=save_cset, args=(v_path_list[set_ind * num_cset:(set_ind + 1)* num_cset],id_list,label_list,save_dir,set_ind,))
            else:
                p = Process(target=save_cset, args=(v_path_list[set_ind * num_cset:],id_list,label_list,save_dir,set_ind,))
            p.start()
        except:
            print('set ind ' + str(set_ind) + ' goes wrong!' )
        # p.join()

def save_split_frames():
    # video_dir = '/mnt/mmu/liuchang/people_video'
    v_path_list = os.listdir(videos_dir)

    num_cset = 500
    set_num = len(v_path_list) // num_cset
    rest_num = len(v_path_list) - set_num * num_cset

    id_list = []
    label_list = []

    with open('labels.txt','r') as f:
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
        # p.join()


# extract_frame()
if __name__ == "__main__":
    save_split_frames()

    # id = 9023949033
    # f = h5py.File(save_dir + str(id) + ".h5")
    # # data = f['9023949033.h5']
    #
    # frames = f[str(id) + "_frames"]
    # label = f[str(id) + "_label"]
    #
    # c = 1