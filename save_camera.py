import os
import cv2
import numpy as np
import h5py

label_names = ['普通','普通人群聚集','群体行为']

def gen_ind(fnum,ex_fnum):
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

def extract_frame(video_path = 'tmovie.mp4',ex_fnum= 300):
    camera=cv2.VideoCapture(video_path)
    total_fnum =int(camera.get(7))
    f_inds = gen_ind(total_fnum,ex_fnum)

    f_matrix = None

    save_dir = video_path.split('.')[0]
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)


    start = 0
    for i in range(total_fnum):
        _, frame=camera.read()
        if i == 0:
            f_matrix = np.zeros((ex_fnum,frame.shape[0],frame.shape[1],frame.shape[2]),dtype='int8')

        while start < ex_fnum and f_inds[start] == i:
            f_matrix[start] = frame
            cv2.imwrite(save_dir + '/' + str(start) + '.jpg',frame)

            start += 1

    return f_matrix

def save_frames():
    videos_path = '/mnt/mmu/liuchang/people_video'
    v_path_list = os.listdir(videos_path)

    num_cset = 500
    set_num = len(v_path_list) // num_cset
    rest_num = len(v_path_list) - set_num * num_cset

    id_list = []
    label_list = []

    with open('labels.txt','r') as f:
        label_info = f.readlines()
        for line in label_info:
            items = line.split()
            id_list.append(items[0])
            label = items[3]
            label = label.replace('\n','')
            label = label.replace('\r','')
            label_list.append(label_names.index(label))

    c_set_ind = 0
    frame_data = []
    # label_data = None
    save_dir = '/mnt/mmu/liuchang/hywData/ksData/'
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    for i in range(len(v_path_list)):
        if i % num_cset == 0 or i == len(v_path_list) - 1:
            if i > 0:
                f = h5py.File(save_dir + "data" + str(c_set_ind) + ".h5", "w")
                f.create_dataset("data", data=frame_data)
                f.close()
                # f.create_dataset("set_y", data=label_data)
            if i < len(v_path_list) - 1:
                # if c_set_ind <= set_num:
                    # frame_data = np.zeros((num_cset,))
                frame_data = []

        path = v_path_list[i]
        id = path.replace('.mp4','')
        id_ind = id_list.index(id)
        frames = extract_frame(path)
        label = label_list[id_ind]

        frame_data.append((id,frames,label))

# extract_frame()
if __name__ == "__main__":
    save_frames()
