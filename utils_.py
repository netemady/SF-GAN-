"""
Some codes from https://github.com/Newmu/dcgan_code
"""
from __future__ import division
import math
import json
import random
import pprint
import scipy.misc
import numpy as np
from time import gmtime, strftime
import os
import csv
import numpy
from sklearn import preprocessing  
import urllib
from pyheatmap.heatmap import HeatMap
from PIL import Image
#from pyheatmap.heatmap import HeatMap
pp = pprint.PrettyPrinter()

get_stddev = lambda x, k_h, k_w: 1/math.sqrt(k_w*k_h*x.get_shape()[-1])

# -----------------------------
# new added functions for pix2pix

def load_data(path,type_):
   x=np.load(path+'data_x_40.npy')
   y=np.load(path+'data_y_40.npy')
   data=np.zeros((x.shape[3],x.shape[0],x.shape[1],5))
   for i in range(x.shape[3]):
     data[i,:,:,0:4]=x[:,:,:,i]
     data[i,:,:,4]=y[:,:,i]
   if type_=='train':  return data[:200]
   if type_=='test':   return data[200:343]
   

def load_data_diag(path,type_,size):
   a=np.ones((size,size))-np.eye(size)
   x=np.load('real/data_x_'+str(size)+'.npy')
   y=np.load('real/data_y_'+str(size)+'.npy')
   data=np.zeros((x.shape[0],x.shape[1],x.shape[2],5))
   for i in range(x.shape[0]):
       data[i,:,:,0:4]=x[i,:,:,:]
       data[i,:,:,4]=y[i,:,:]
   node=np.zeros((x.shape[0],x.shape[1]))
   for i in range(x.shape[0]):
       node[i]=sum(data[i,:,:,0]*np.eye(size))
   for i in range(x.shape[0]):
       data[i,:,:,0]=data[i,:,:,0]*a  #diagnal is removed
       data[i,:,:,4]=data[i,:,:,4]*a
   node=node.reshape(x.shape[0],size,1,1)
   
   if type_=='train':  return data[:200],node[:200]
   if type_=='test':   return data[200:343],node[200:343]


def minmax(data):
    b=data
    min_=data.min() 
    max_=data.max()
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            b[i][j]=(data[i][j]-min_)/(max_-min_)
    return b,min_,max_

# -----------------------------

def get_image(image_path, image_size, is_crop=True, resize_w=64, is_grayscale = False):
    return transform(imread(image_path, is_grayscale), image_size, is_crop, resize_w)

def save_images(images, size, image_path):
    return imsave(inverse_transform(images), size, image_path)

def imread(path, is_grayscale = False):
    if (is_grayscale):
        return scipy.misc.imread(path, flatten = True).astype(np.float)
    else:
        return scipy.misc.imread(path).astype(np.float)

def merge_images(images, size):
    return inverse_transform(images)

def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j*h:j*h+h, i*w:i*w+w, :] = image

    return img

def imsave(images, size, path):
    return scipy.misc.imsave(path, merge(images, size))

def transform(image, npx=64, is_crop=True, resize_w=64):
    # npx : # of pixels width/height of image
    if is_crop:
        cropped_image = center_crop(image, npx, resize_w=resize_w)
    else:
        cropped_image = image
    return np.array(cropped_image)/127.5 - 1.

def inverse_transform(images):
    return (images+1.)/2.

def mat2heatlist(base):
  base_list=[]
  for i in range(25):
     for j in range(60):
        if base[i][j]!=0:         
            base_list.append([j,i])
  return base_list
             
def gen_heatmap(output,mapname):
    data1 =mat2heatlist(output)
    with open('base.txt') as f:
            base=[]
            f_csv = csv.reader(f)
            for row in f_csv:
              base.append(row)
    base_=np.empty((25,60))
    for i in range(25):
      for j in range(60):
         base_[i][j]=base[i][j]          
    data2=mat2heatlist(base_)
    hm = HeatMap(data2)
    hit_img = hm.clickmap()
    hm2 = HeatMap(data1)
    hit_img2 = hm2.heatmap(base=hit_img,r=1)
    hit_img2.save(mapname)
    
def M2I(data):
    data = data*255
    new_im = Image.fromarray(data.astype(np.uint8))
    new_im.show()
    return new_im   

