# -*- coding: UTF-8 -*-
import cv2
import random
import numpy as np

def crop(image,x1,y1,x2,y2):
    return image[:,y1:y2,x1:x2,:]

class center_crop(object):
    def __init__(self):
        pass

    def __call__(self,frames):
        height=frames.shape[1]
        width=frames.shape[2]
        #手机屏幕自适配
        if(height>width):
            return crop(frames,(height-width)/2,(height-width)/2+width,0,width)
        if(width>height):
            return crop(frames,0,height,(width-hegith)/2,(width-height)/2+height)

class resize(object):
    def __init__(self,size):
        self.size=size

    def __call__(self,frames):
        resized_frames=np.zeros((frames.shape[0],self.size,self.size,3)).astype(np.uint8)
        for i in range(0,frames.shape[0]):
            image=frames[i]
            image=cv2.resize(image,(self.size,self.size))
            resized_frames[i,:,:,:]=image
        return resized_frames


class random_crop_size(object):
    def __init__(self,size):
        self.size=size

    def __call__(self,frames):
        begin_y=random.randint(0,frames.shape[1]-self.size)
        begin_x=random.randint(0,frames.shape[2]-self.size)
        return crop(frames,begin_y,begin_y+self.size,begin_x,self.begin_x+self.size)

class random_crop_frames(object):
    def __init__(self,frames_remain):
        self.frames_remain=frames_remain

    def __call__(self,frames):
        total_frames=frames.shape[0]
        mask=np.zeros((total_frames),dtype=bool)
        seq=np.arange(total_frames)
        np.random.shuffle(seq)
        mask[seq[:total_frames]]=True
        frames=frames[mask]
        return frames

class random_horizontal_filp(object):
    def __init__(self):
        pass

    def __call__(self,frames):
        for i in range(0,frames.shape[0]):
            image=frames[i]
            image=cv2.flip(image,1)
            frames[i,:,:,:]=image
        return frames

class random_brightness(object):
    def __init__(self,value=50):
        self.value=50

    def __call__(self,frames):
        frames=frames.astype(np.int32)
        value=random.randint(-value,value)
        frames[:,:,:,:]+=value
        frames[frames[:,:,:,:]<0]=0
        frames[frames>255]=255
        return frames.astype(np.uint8)

class random_contrast(object):
    def __init__(self,value_low=0.5,value_high=1.5):
        self.value_low=value_low
        self.value_high=value_high

    def __call__(self,frames):
        frames=frames.astype(np.int32)
        value=random.uniform(self.value_low,self.value_high)
        frames[:,:,:,:]*=value
        frames[frames[:,:,:,:]<0]=0
        frames[frames>255]=255
        return frames.astype(np.uint8)

class random_rgb(object):
    def __init__(self,value_r,value_g,value_b):
        self.value_r=value_r
        self.value_g=value_g
        self.value_b=value_b

    def __call__(self,frames):
        frames=frames.astype(np.float32)
        value_r=1+random.uniform(-self.value_r,self.value_r)
        value_g=1+random.uniform(-self.value_g,self.value_g)
        value_b=1+random.uniform(-self.value_b,self.value_b)
        frames[:,:,:,0]=frames[:,:,:,0]*value_r
        frames[:,:,:,1]=frames[:,:,:,1]*value_g
        frames[:,:,:,2]=frames[:,:,:,2]*value_b
        return frames.astype(np.uint8)
