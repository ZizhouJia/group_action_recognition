import cv2
import numpy as np

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


    start = 0
    for i in range(total_fnum):
        _, frame=camera.read()
        if i == 0:
            f_matrix = np.zeros((ex_fnum,frame.shape[0],frame.shape[1],frame.shape[2]))

        while start < ex_fnum and f_inds[start] == i:
            f_matrix[start] = frame
            start += 1

    return f_matrix

extract_frame()
