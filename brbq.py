import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import h5py
from math import *
import matplotlib.patches as patches

import cooler
#%matplotlib inline

from math import sqrt
from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray

import cv2

def fileParser(
    loopfile, 
    layoutfile, 
    lborder, 
    rborder,
    chrom):

    c = cooler.Cooler(loopfile)
    
    known=pd.read_excel(layoutfile)
    known_chr1=known[known['Chr']==chrom]
    
    known_chr1_seg=known_chr1[known_chr1['Genomic bin, Left base']>=lborder]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Left base']<=rborder]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']>=lborder]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']<=rborder]
    
    image=c.matrix(balance=True, sparse=False)[lborder:rborder, lborder:rborder]
    
    return image, known_chr1_seg
    
def GetCenter(
    image, 
    x, y, 
    r):
    tmp=image[int(x-floor(r)):int(x+floor(r)), int(y-floor(r)):int(y+floor(r))]
    if tmp.size==0:
        return np.array([np.nan, np.nan])
    res=np.unravel_index(tmp.argmax(), tmp.shape)
    return np.array([int(x-floor(r))+res[0], int(y-floor(r))+res[1]])
    
def DiagonalFilter(
    image, 
    x, y, 
    r):
    for i in range(image.shape[0]):
        if (((i-x-0.5+2)**2+(i-y-0.5)**2)<=r**2) or (((i-x-0.5+2)**2+(i-y-0.5+1)**2)<=r**2):
            return False
    return True
    
def BarbekuFinderFrameDraw(
    loopfile, 
    layoutfile, 
    left, right, 
    k=3, 
    frame=7, 
    sigmaX=7,
    min_sigma=3,
    max_sigma=10,
    thresh=0.5, 
    thresh2=60):
    
    image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right)
    image=image/np.max(image)*255
    real=image.shape[0]
    enlarged=np.zeros((real*k, real*k))
    for i in range(real*k):
        for j in range(real*k):
            enlarged[i, j]=image[i//k, j//k]
    blur=cv2.GaussianBlur(enlarged, (frame,frame), sigmaX)
    blur=blur/np.max(blur)*255
    blobs_log = blob_log(blur, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=1000, threshold=0.001)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    detected=[]

    fig=plt.figure(figsize=(8,8))
    ax = fig.add_subplot(111)
    ax.imshow(image, cmap='YlOrRd')
    for i in range(blobs_log.shape[0]):
        y, x, r = blobs_log[i,:]
        if  (x>y) and DiagonalFilter(image, x//k, y//k, r/k):
        #if (x>y):
            detected.append([x//k, y//k, r/k])
            xc, yc=GetCenter(image, detected[-1][0], detected[-1][1], detected[-1][2])
            if not(np.isnan(xc)):
                test_thresh=np.sum(image[int(x//k-floor(r/k))-1:int(x//k+floor(r/k))+1, int(y//k-floor(r/k))-1:int(y//k+floor(r/k))+1])-np.sum(image[int(x//k-floor(r/k)):int(x//k+floor(r/k)), int(y//k-floor(r/k)):int(y//k+floor(r/k))])
                test_num=(int(x//k+floor(r/k))-int(x//k-floor(r/k))+2)*(int(y//k+floor(r/k))-int(y//k-floor(r/k))+2)-(int(x//k+floor(r/k))-int(x//k-floor(r/k)))*(int(y//k+floor(r/k))-int(y//k-floor(r/k)))
                test_thresh=test_thresh/test_num
                if ((image[xc,yc]-test_thresh)/test_thresh>thresh) and (image[xc, yc]>thresh2):
                #if image[xc, yc]>np.mean(image):
                #if image[xc,yc]>thresh:
                    print(x//k, y//k, r/k, '|||', xc, yc, ':::', image[xc,yc])
                    c = plt.Circle((x//k, y//k), r/k, color='black', linewidth=2, fill=False)
                    ax.plot(x//k, y//k, 'ko')
                    ax.add_patch(c)
                    ax.plot(xc, yc, 'bs')
    print(np.mean(image))
    ax.set_axis_off()
    ax.plot((known_chr1_seg['Genomic bin, Right base']-left-1), (known_chr1_seg['Genomic bin, Left base']-left-1), 'go')
    plt.show()
    
def BarbekuFinderFrameLog(
    whole_image,
    whole_layout,
    left, right, 
    k=3, 
    frame=7, 
    min_sigma=3,
    max_sigma=10,
    num_sigma=10,
    sigmaX=7, 
    thresh=0.5, 
    thresh2=60):
    
    #image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right)
    image=whole_image[left:right, left:right]
    known_chr1_seg=whole_layout[whole_layout['Genomic bin, Left base']>=left]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Left base']<=right]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']>=left]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']<=right]
    
    image=image/np.max(image)*255
    real=image.shape[0]
    enlarged=np.zeros((real*k, real*k))
    for i in range(real*k):
        for j in range(real*k):
            enlarged[i, j]=image[i//k, j//k]
    blur=cv2.GaussianBlur(enlarged, (frame,frame), sigmaX)
    blur=blur/np.max(blur)*255
    blobs_log = blob_log(blur, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma, threshold=0.001)
    blobs_log[:, 2] = blobs_log[:, 2] * sqrt(2)

    detected=[]
    res=[]
    
    for i in range(blobs_log.shape[0]):
        y, x, r = blobs_log[i,:]
        if  (x>y) and DiagonalFilter(image, x//k, y//k, r/k):
        #if (x>y):
            detected.append([x//k, y//k, r/k])
            xc, yc=GetCenter(image, detected[-1][0], detected[-1][1], detected[-1][2])
            if not(np.isnan(xc)):
                #if image[xc, yc]>np.mean(image):
                #if image[xc,yc]>thresh:
                test_thresh=np.sum(image[int(x//k-floor(r/k))-1:int(x//k+floor(r/k))+1, int(y//k-floor(r/k))-1:int(y//k+floor(r/k))+1])-np.sum(image[int(x//k-floor(r/k)):int(x//k+floor(r/k)), int(y//k-floor(r/k)):int(y//k+floor(r/k))])
                test_num=(int(x//k+floor(r/k))-int(x//k-floor(r/k))+2)*(int(y//k+floor(r/k))-int(y//k-floor(r/k))+2)-(int(x//k+floor(r/k))-int(x//k-floor(r/k)))*(int(y//k+floor(r/k))-int(y//k-floor(r/k)))
                test_thresh=test_thresh/test_num
                if ((image[xc,yc]-test_thresh)/test_thresh>thresh) and (image[xc, yc]>thresh2):
                    res.append([x//k, y//k, r/k, xc, yc, (image[xc,yc]-test_thresh)/test_thresh, image[xc, yc]])
                    #print(image[xc,yc], test_thresh, test_num)
    return np.array(res)
    
def IsDetected(x, y, found):
    for i in range(found.shape[0]):
        xt, yt, rt, xc, yc, thr1, thr2=found[i, :]
        #if ((xt-x)**2+(yt-y)**2 <= rt**2):
        if ((xt-x)**2+(yt-y)**2 <= ceil(rt)**2):
            return True
    return False
    
def IntersectTough(x1, y1, r1, x2, y2, r2):
    if sqrt(((x2-x1)**2+(y2-y1)**2))<(max(r1, r2)+2):
        return True
    else:
        return False
        
def IntersectEasy(x1, y1, r1, x2, y2, r2):
    if sqrt(((x2-x1)**2+(y2-y1)**2))<(r1+r2+1):
        return True
    else:
        return False        