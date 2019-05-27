# -*- coding: UTF-8 -*-
import os
import torch
import model
import time
import numpy as np
import cv2
import torch.nn.functional as F
from torchvision import transforms
import video_transforms
(major_ver, _,_) = (cv2.__version__).split('.')

def extract_frame_cv(path):
    video = cv2.VideoCapture(path)
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

def gen_ind(fnum, ex_fnum=300):
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

def resize_frames(frames):

    f_shape = list(frames.shape)
    new_shape = [1]
    new_shape.extend(f_shape[1:])
    new_shape = tuple(new_shape)

    a_inds = gen_ind(fnum=f_shape[0])
    batch_list = [frames[ind].reshape(new_shape) for ind in a_inds]
    new_frames = np.concatenate(batch_list, axis=0)
    return new_frames

def load_models(model_classes,model_params,model_save_path):
    return_models=[]
    for i in range(0,len(model_classes)):
        model=model_classes[i](**model_params[i])
        #model.load_state_dict(torch.load(model_save_path))
        return_models.append(model.cuda())
    return return_models

def predict(models,features):
    features=torch.Tensor(features).view(-1,features.shape[0],features.shape[1]).cuda()
    pred_result=np.zeros((1,2))
    for model in models:
        pred=model(features)
        pred=F.softmax(pred,dim=-1)
        pred_result+=pred.detach().cpu().numpy()
    if(pred_result[0,0]<pred_result[0,1]):
        return 1
    else:
        return 0


video_t=transforms.Compose([
        video_transforms.center_crop(),
        video_transforms.resize(299),
        video_transforms.to_tensor(),
        video_transforms.normalize()
                ])

if __name__=='__main__':
    video_dir="/mnt/mmu/liuchang/people_video"
    result_path="./result.txt"
    result_dict={}
    
    model_classes=[model.DbofModel.DbofModel]
    model_params=[{"vocab_size":2,"pretrain":False,"video_size":2048,"audio_size":0}]
    model_save_path=["check_points/"]
    models=load_models(model_classes,model_params,model_save_path)

    feature_extractor=model.inception_v3.inception_v3(pca_dir=None).cuda()

    #begin to classify
    video_list=os.listdir(video_dir)
    video_list=video_list[:2]
    print(video_list)
    for video in video_list:
        video_path=os.path.join(video_dir,video)
        frames=extract_frame_cv(video_path)
        if(frames is None):
            print("detect frame is none")
            print(video)
            result_dict[video]=0
            continue
        frames=video_t(frames)
        frames=frames.cuda()
        features=feature_extractor(frames)
        features=features.detach().cpu().numpy()
        features=resize_frames(features)
        result_dict[video]=predict(models,features)

    f=open(result_path,"w")
    mapping=["not","is"]
    for key in result_dict:
        f.write(str(key[:-4])+" "+mapping[result_dict[key]]+"\n")
    f.close()
         




    
