import warnings
warnings.simplefilter(action='ignore')


#import general math modules
import numpy as np
import pandas as pd
import h5py
from math import *

#import to add console arguments
import sys

from skimage import data
from skimage.feature import blob_dog, blob_log, blob_doh
from skimage.color import rgb2gray
import cv2

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
sns.set()
sns.set_context("notebook", font_scale=1.5, rc={"lines.linewidth": 2.5})

import cooler
import brbq



loopfile=sys.argv[1] # coolfile name
layoutfile=sys.argv[2] # layoutfile of None




c = cooler.Cooler(loopfile)
df=c.bins()[:]

boundaries=[]
for cn in c.chromnames:
    boundaries.append([cn, np.min(df[df['chrom']==cn].index.values), np.max(df[df['chrom']==cn].index.values)])


for cn, left, right in boundaries:
    print(cn+' WORKING:')
    print('Finding all loops (takes most time)...')
    brbq.BarbekuFinderWhole(loopfile=loopfile,
                            layoutfile=layoutfile,
                            left=left,
                            right=right,
                            chrom=cn.split('chr')[-1],
                            resultname='tmp.'+loopfile.split('.cool')[0])
    print('Merging intersecting loops...')
    filtered=brbq.BarbekuFiltrationFromFound(loopfile=loopfile,
                            layoutfile=layoutfile,
                            left=left,
                            right=right,
                            chrom=cn.split('chr')[-1],
                            foundname='brbq-tmp.'+loopfile.split('.cool')[0]+'.'+cn)
    print('Finding crosses...')
    cross=brbq.crossFinder(loopfile=loopfile,
                            layoutfile=layoutfile,
                            left=left,
                            right=right,
                            chrom=cn.split('chr')[-1])
    print('Matching with layout (or just passing by, if no layout)...')
    chk=brbq.BarbekuMarkup(loopfile=loopfile,
                            layoutfile=layoutfile,
                            left=left,
                            right=right,
                            chrom=cn.split('chr')[-1],
                            cross=cross,
                            filtered=filtered)
    print('FDR obtaining...')
    fdrs=brbq.BarbekuFDR(filtered=filtered)
    print('Writing dataset...')
    brbq.BarbekuDfParser(filtered=filtered,
                        chk=chk,
                        fdrs=fdrs,
                        filename=loopfile.split('.cool')[0]+'.'+cn,
                        isLayout=False)
    print('Writing Jucier track...')
    brbq.BarbekuJucierParser(chrom=cn.split('chr')[-1],
                            filtered=filtered,
                            filename=loopfile.split('.cool')[0]+'.'+cn,
                            res=2000)
    print('DONE!\n')
