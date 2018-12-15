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
    
    # Returns contact map from cooler file (loopfile) in borders (lborder:rborder) 
    # and list of corresponding manual layout (layout) for given chromosome (chrom)
    
    c = cooler.Cooler(loopfile)
    
    known=pd.read_excel(layoutfile)
    known_chr1=known[known['Chr']==chrom]
    
    known_chr1_seg=known_chr1[known_chr1['Genomic bin, Left base']>=0]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Left base']<=rborder-lborder]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']>=0]
    known_chr1_seg=known_chr1_seg[known_chr1_seg['Genomic bin, Right base']<=rborder-lborder]
    
    image=c.matrix(balance=True, sparse=False)[lborder:rborder, lborder:rborder]
    
    return image, known_chr1_seg
    
    
def GetCenter(
    image, 
    x, y, 
    r):
    
    # Returns center (the brightest spot) of the loop (x, y, r) at image
    
    tmp=image[int(x-floor(r)):int(x+floor(r)), int(y-floor(r)):int(y+floor(r))]
    if tmp.size==0:
        return np.array([np.nan, np.nan])
    res=np.unravel_index(tmp.argmax(), tmp.shape)
    return np.array([int(x-floor(r))+res[0], int(y-floor(r))+res[1]])
    
    
def DiagonalFilter(
    image, 
    x, y, 
    r):
    
    # Checks whether the loop intersects the main diagonal (possible TAD)
    
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
    
    # plotter of doubtful spots CURRENTLY DEPRECATED
    
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
    left, right, 
    k=5, 
    frame=7, 
    min_sigma=0.1,
    max_sigma=4,
    num_sigma=15,
    sigmaX=7, 
    thresh=1.0, 
    thresh2=65):
    
    # The most important thing: return array of loops in format
    # coordinats (geom, 2 columns), radius, center (actual, 2 columns), respective threshold, absolute threshold
    # input: 
    #       image from fileParser
    #       frame borders (recomended to use 30-width frame)
    #       k: technical, enlarging scale
    #       frame: technical, smoothing size
    #       sigmaX: technical, smoothing dispersion
    #       min_sigma, max_sigma: minimal and maximal "sizes" of loops. Highly dependent on k
    #       num_sigma: number of checked sizes
    #       thresh: respective threshold towards frame round the loop
    #       thresh2: absolute threshold -- 25% quartile of image in frame
    #       
    # To get better results for other species, recommended to change min_sigma, max_sigma and thresholds (more like thresholds)
    
    
    
    
    #image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right)
    image=whole_image[left:right, left:right]
    
    
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
    
    # Check manual loop for being found (array of entries from FrameFinder function)
    
    for i in range(found.shape[0]):
        xt, yt, rt, xc, yc, thr1, thr2=found[i, :]
        #if ((xt-x)**2+(yt-y)**2 <= rt**2):
        if ((xt-x)**2+(yt-y)**2 <= ceil(rt)**2):
            return True
    return False
    
def IntersectTough(x1, y1, r1, x2, y2, r2):
    # Check whether loops "intersect"
    if sqrt(((x2-x1)**2+(y2-y1)**2))<(max(r1, r2)+2):
        return True
    else:
        return False
        
def IntersectEasy(x1, y1, r1, x2, y2, r2):
    # Check whether loops "intersect" or too close
    # in debugged version not really a difference
    if sqrt(((x2-x1)**2+(y2-y1)**2))<(r1+r2+1):
        return True
    else:
        return False        
    
    
def BarbekuFinderWhole(
    loopfile,
    layoutfile,
    left, right,
    chrom,
    
    resultname,
    
    frame=7,
    k=5,
    thresh=1.0,
    thresh2=64,
    min_sigma=0.1,
    max_sigma=4,
    num_sigma=15
    ):
    
    #returns found for whole image and writes them for file with result name 
    #please see FinderFrameLog if you find unknown input
    
    image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right, chrom=chrom)


    found=[]
    for step in range(0, right-left-30, 1):
        found_tmp=BarbekuFinderFrameLog(frame=frame, whole_image=image, left=step, right=step+30, k=k, thresh=thresh, thresh2=thresh2, min_sigma=min_sigma, max_sigma=max_sigma, num_sigma=num_sigma)
        for count in range(found_tmp.shape[0]):
            x, y, r, xc, yc, thr1, thr2 = found_tmp[count,:]
            found.append([x+step, y+step, r, xc+step, yc+step, thr1, thr2])
        #print(step, end=' ')


    found=np.array(found)    
    np.savetxt('brbq-'+resultname+'.chr'+str(chrom), found, delimiter=',')
    
    
def crossFinder(
    loopfile,
    layoutfile,
    left, right,
    chrom
    ):
    image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right, chrom=chrom)

    cross=[]
    for i in range(image.shape[0]-2):
        if (image[i, i+2]==0):
            cross.append(i)
    return cross



def BarbekuFiltrationFromFound(
    loopfile,
    layoutfile,
    left, right,
    chrom,
    
    foundname
    ):
    
    image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right, chrom=chrom)
    found=np.loadtxt(foundname, delimiter=',')
    
    fl=np.ones(found.shape[0])
    
    sum=0
    for i in range(found.shape[0]):
        if fl[i]==0:
            continue
        intersect=[]
        intersect.append(i)
        for j in range(i+1, found.shape[0], 1):
            for tmp in intersect: 
                if IntersectTough(found[tmp, 0], found[tmp, 1], found[tmp, 2], found[j, 0], found[j, 1], found[j, 2]) and (fl[j]==1):
                    intersect.append(j)
                    break
        sum+=len(intersect)-1
        #print(sum, end=' ')
        if len(intersect)==1:
            continue
        maxes=[]
        for tmp in intersect:
            maxes.append(found[tmp, 6])
        max_pos=np.argmax(np.array(maxes))
        for j in range(len(intersect)):
            if j!=max_pos:
                fl[intersect[j]]=0
    
    filtered=found[fl.astype(bool),:]

    st=0
    for i in range(known_chr1_seg.shape[0]):
        knr=known_chr1_seg['Genomic bin, Right base'].values
        knl=known_chr1_seg['Genomic bin, Left base'].values
        if IsDetected(knr[i]-1, knl[i]-1, filtered):
            continue
        else:
            st+=1
            #l1=int(max(known_chr1_seg['Genomic bin, Left base'][i]-1-15, 0))
            #r1=l1+30
            print('        NOT FOUND:   ', knr[i]-1, knl[i]-1)
            #BarbekuFinderFrameDraw(loopfile=loopfile, layoutfile=layoutfile, left=l1, right=r1, k=5, thresh=0.5, thresh2=60, min_sigma=0.5, max_sigma=5)

    print(st, '/', known_chr1_seg.shape[0], '/', filtered.shape[0])
    print('Sensitivity: ', 1-st/known_chr1_seg.shape[0])
    print('FDR: ', (filtered.shape[0]+st-known_chr1_seg.shape[0])/filtered.shape[0])

    return filtered



def BarbekuMarkup(
    loopfile,
    layoutfile,
    left, right,
    chrom,
        
    cross,
    filtered,
    ran=15
):
    
    image, known_chr1_seg=fileParser(loopfile, layoutfile, left, right, chrom=chrom)

    chk=(-1)*np.ones(filtered.shape[0])
    for i in range(known_chr1_seg.shape[0]):
        knr=known_chr1_seg['Genomic bin, Right base'].values
        knl=known_chr1_seg['Genomic bin, Left base'].values
        xtest=knr[i]-1
        ytest=knl[i]-1
        for j in range(filtered.shape[0]):
            if ((xtest-filtered[j,0])**2+(ytest-filtered[j,1])**2 <= ceil(filtered[j,2])**2):
                if chk[j]!=(-1):
                    print('Loops are too close to safely find a detector')
                else:
                    chk[j]=i

    for i in range(filtered.shape[0]):
        if chk[i]==-1:
            for j in range(len(cross)):
                if abs(filtered[i, 0]-cross[j])<=ran:
                    chk[i]=-2

    return chk




def BarbekuFDR(
    filtered
    ):
    
    import scipy.stats as stats
    gamma = stats.gamma

    diag_whole=np.abs(filtered[:, 3]-filtered[:, 4])

    param = gamma.fit(diag_whole, floc=0)
    tmpx=np.linspace(-0.5, 30.5, 1000)
    pdf_fitted = gamma.pdf(tmpx, *param)

    h=plt.hist(diag_whole, bins=np.arange(31)-0.5, label='Discovered', normed=True)
    plt.close()
    
    fdr_diag=[1-np.sum(pdf_fitted[abs(tmpx-i)<=0.5])*(tmpx[1]-tmpx[0])/h[0][i] for i in range(30)]
    fdrs=np.zeros(filtered.shape[0])
    for i in range(filtered.shape[0]):
        fdrs[i]=fdr_diag[int(filtered[i, 3]-filtered[i, 4])]
        
    return fdrs




def BarbekuDfParser(
    filtered,
    chk,
    fdrs,
    
    file,
    isLayout
    ):
    if isLayout:
        df=pd.DataFrame({
            'x (geom)': filtered[:, 0], 
            'y (geom)': filtered[:, 1], 
            'radius': filtered[:, 2], 
            'x (center)': filtered[:, 3], 
            'y (center)': filtered[:, 4], 
            'thresh (respect)': filtered[:, 5],
            'thresh (absolute)': filtered[:, 6],
            'Mapped': chk>-1,
            #'Mapped': np.nan,
            'cross15': chk==-2,
            'fdr': fdrs
        })
    else:
        df=pd.DataFrame({
            'x (geom)': filtered[:, 0], 
            'y (geom)': filtered[:, 1], 
            'radius': filtered[:, 2], 
            'x (center)': filtered[:, 3], 
            'y (center)': filtered[:, 4], 
            'thresh (respect)': filtered[:, 5],
            'thresh (absolute)': filtered[:, 6],
            #'Mapped': chk>-1,
            'Mapped': np.nan,
            'cross15': chk==-2,
            'fdr': fdrs
        })
    df.to_csv(file+'.brbq')
    
    
    
def BarbekuJucierParser(
    chrom,
    filtered,
    
    file,
    res=2000
    ):
    
    df_juice=pd.DataFrame({
        'chr1': chrom*np.ones(filtered.shape[0]).astype(int),
        'x1': res*(filtered[:, 3]+1-filtered[:, 2]/2).astype(int),
        'x2': res*(filtered[:, 3]+1+filtered[:, 2]/2).astype(int),
        'chr2': chrom*np.ones(filtered.shape[0]).astype(int),
        'y1': res*(filtered[:, 4]+1-filtered[:, 2]/2).astype(int),
        'y2': res*(filtered[:, 4]+1+filtered[:, 2]/2).astype(int),
        'color': np.array(['0,255,0']*filtered.shape[0]),
        'comment': np.nan
    })
    df_juice.to_csv(file+'-track.brbq', index=None, sep='\t')
    
    