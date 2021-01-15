#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 26 08:50:09 2020

@author: sam
"""

# In[]
import os
import skimage.io
import numpy as np
import keras
import matplotlib.pyplot as plt
import time
import skimage
from tqdm import tqdm
#vidName='../../../../media/sam/internalHDD/behavior/hiRes/012320_uv/'
vidName='../../../../media/sam/internalHDD/behavior/hiRes/012420_uv30s/'


tiffStack = []
folder = sorted(os.listdir(vidName))
for f in folder:
    if f == 'location' or f=='light':
        continue
    tiffStack.extend([vidName+f+'/'+file for file in sorted(os.listdir(vidName+f))])

#data=[tiffStack,]

# In[]


def prep_img(file,thresh_scale=.6, min_sz=20000, bound_scale=1.25, target_size=(28,28)):
    #load img and find region
    img = skimage.io.imread(file)
    bi_thresh=thresh_scale*np.percentile(img,99.9)
    img_bi = img>bi_thresh
    lab = skimage.measure.label(img_bi)
    regions = skimage.measure.regionprops(lab,)
    #get com and perimeter
    max_s=0
    for i,props in enumerate(regions):
        if (props.convex_area<min_sz):
            continue
        if props.convex_area>max_s:
            max_s=props.convex_area
            img2=np.pad(props.filled_image,(10,10),'constant',constant_values=(0,0))
            com = np.array(props.centroid)
#    print(max_s)
    if max_s==0:
        return None
    loc=np.where(img2>0)
    loc_com=[np.mean(loc[0]),np.mean(loc[1])]
    perim = skimage.measure.find_contours(img2.astype('int'),.5)[0]+com-loc_com
    #fing the longest point and rotation angle
    rad = np.linalg.norm(com-perim,axis=1)
    ind = np.argmax(rad)
    long = int(rad[ind])
    vec = perim[ind]-com
    ang = np.degrees(np.arctan2(vec[1],vec[0]))
#    print(ang)
    #center, rotate, and crop original image
    sc =2
    img = img[int(com[0]-sc*long):int(com[0]+sc*long),int(com[1]-sc*long):int(com[1]+sc*long)]
    img = skimage.transform.rotate(img,-ang)
    c = img.shape[0]//2
    img = img[int(c-bound_scale*long):int(c+bound_scale*long),int(c-bound_scale*long):int(c+bound_scale*long)]
    #resize to standard pixel
    img = skimage.transform.resize(img,target_size, anti_aliasing=True)
#    plt.imshow(img)
    return img

tic=time.time()
data=[]
ind = []
tic= time.time()
for i,file in tqdm(enumerate(tiffStack[len(data):10]),position=0,leave=True):
    q = prep_img(file)
    if q is None:
        continue
    data.append(q)
    ind.append(i)
print(time.time()-tic)
plt.imshow(data[-1])
data = np.array(data)
data = data[:,:,:,None]
ind = np.array(ind)
# In[]
#np.save('otherData/VAE/012420_uv30s_hiRes_align_28.npy',data)
#np.save('otherData/VAE/012420_uv30s_hiRes_align_28_ind.npy',ind)
np.save('otherData/VAE/012420_uv30s_hiRes_align_28_light.npy',light)


# In[]
"""
Lag time embedding
"""
tp = 6
#data0 = np.load('otherData/hiResImg_align_28.npy')
data0=np.load('otherData/VAE/012420_uv30s_hiRes_align_28.npy')
#data = []
#for i in tqdm(range(tp,data0.shape[0]),position=0):
#    data.append(data0[i-tp:i,...])
data = np.zeros((data0.shape[0],tp,data0.shape[1],data0.shape[2],data0.shape[3]))
for i in tqdm(range(tp,data0.shape[0]),position=0):
    data[i] = data0[i-tp:i,...]
#data = np.array(data)
data=data/data.max()



# In[]
"""
Light cropping
"""
tp = 6
data0=np.load('otherData/VAE/012420_uv30s_hiRes_align_28.npy')
light=np.load('otherData/VAE/012420_uv30s_hiRes_align_28_light.npy')

#vidName='../../../../media/sam/internalHDD/behavior/hiRes/012320_uv/'
#light='../../../../media/sam/internalHDD/behavior/hiRes/012420_uv30s/'

t_range = [-3,15]
data = []

if type(light)==str:
    light_files=sorted(os.listdir(light+'light/'))
    light = []
    for f in light_files:
        light.extend(np.load(vidName+'light/'+f))
    light = np.array(light)
    light=light>15.8

t_on=[0,]
for l in light[1:]:
    if l:
        t_on.append(t_on[-1]+1)
    else:
        t_on.append(0)
t_on = np.array(t_on)



ind = np.where(t_on==1)[0]
for loc in ind:
    if loc<-t_range[0]*120 or loc+t_range[-1]*120>data0.shape[0]:
        continue
    for i in range(loc+t_range[0]*120,loc+t_range[-1]*120):
        data.append(data0[i-tp:i,...])

data=np.array(data)



