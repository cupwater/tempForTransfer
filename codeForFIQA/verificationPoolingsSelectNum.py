

from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  
import random 
import scipy.io as sio
import numpy as np 
import math


res = sio.loadmat('YTFres_spherefaceProposed_poolingsSelectNum.mat')
res = res['res']
print res.shape 
dev_files = ['random', 'RQS', 'SVM', 'ROQ-CNN']
modelName = ['Centerloss', 'Sphereface', 'Insightface', 'Normface']

gt = np.ones(500)
gt[250:500] = -1
gt = gt.astype(int)

tprRes = np.zeros((210,5))
finalTprRes = np.zeros((21,10))


for i in range(3):
    for j in range(7):

        for sidx in range(10):
            currentIndexInRes = sidx*21+i*7+j
            similarity = res[0,currentIndexInRes]['scores'][0,:]
            accuracy = res[0,currentIndexInRes]['accuracy'][0,:]
            tprRes[i*70+j*10+sidx, 4] = accuracy
            fpr, tpr, thresholds = roc_curve(gt, similarity) 

            # find the 0.1, 0.01, 0.001, 0.0001 position
            for x in range(len(fpr)-1,1,-1):
                if fpr[x]==0.1 or (fpr[x]>0.1 and fpr[x-1]<=0.1):
                    tprRes[i*70+j*10+sidx, 0] = tpr[x]

                if fpr[x]==0.01 or (fpr[x]>0.01 and fpr[x-1]<=0.01):
                    tprRes[i*70+j*10+sidx, 1] = tpr[x]

                if fpr[x]==0.001 or (fpr[x]>0.001 and fpr[x-1]<=0.001):
                    tprRes[i*70+j*10+sidx, 2] = tpr[x]

                if fpr[x]==0.0001 or (fpr[x]>0.0001 and fpr[x-1]<=0.0001):
                    tprRes[i*70+j*10+sidx, 3] = tpr[x]

        for k in range(4):
            finalTprRes[i*7+j,k] = np.mean(tprRes[i*70+j*10:i*70+(j+1)*10, k]) 
            finalTprRes[i*7+j,k+4] = finalTprRes[i*7+j,k] - np.max(tprRes[i*70+j*10:i*70+(j+1)*10, k]) 
        finalTprRes[i*7+j,8] = np.mean(tprRes[i*70+j*10:i*70+(j+1)*10, 4]) 
        finalTprRes[i*7+j,9] = finalTprRes[i*7+j,8] - np.min(tprRes[i*70+j*10:i*70+(j+1)*10, 4])






tprResOut = open('YTFtprPoolingsSelectNumRes.txt', 'w')
for i in range(3):
    for j in range(7):
        for k in range(10):
            for x in range(4):
                print >> tprResOut, tprRes[i*70+j*10+k, x],
                print >> tprResOut, '\t',
            print >> tprResOut, ' ' 
tprResOut.close()

finalTprResOut = open('YTFfinaltprPoolingsSelectNumRes.txt', 'w')
for i in range(3):
    for j in range(7):
        for x in range(4):
            print >> finalTprResOut,  str(round(finalTprRes[i*7+j, x],3)) + '+_' + str(round(finalTprRes[i*7+j, x+4],3)),
            print >> finalTprResOut, '\t',
        print >> finalTprResOut,  str(round(finalTprRes[i*7+j, 8],3)) + '+_' + str(round(finalTprRes[i*7+j, 9],3)),
        print >> finalTprResOut, ' ' 
finalTprResOut.close() 











from sklearn.metrics import roc_curve, auc  
import matplotlib.pyplot as plt  
import random 
import scipy.io as sio
import numpy as np 
import math


res = sio.loadmat('IJBA11res_spherefaceProposed_poolingsSelectNum.mat')
res = res['res']
print res.shape 
dev_files = ['random', 'RQS', 'SVM', 'ROQ-CNN']
modelName = ['Centerloss', 'Sphereface', 'Insightface', 'Normface']

# gt = np.ones(500)
# gt[250:500] = -1
# gt = gt.astype(int)

tprRes = np.zeros((210,5))
finalTprRes = np.zeros((21,10))


for i in range(3):
    for j in range(7):

        for sidx in range(10):
            currentIndexInRes = sidx*21+i*7+j
            similarity = res[0,currentIndexInRes]['similarity'][0,:]
            gt = res[0,currentIndexInRes]['gt'][0,:]
            accuracy = res[0,currentIndexInRes]['accuracy'][0,:]
            tprRes[i*70+j*10+sidx, 4] = accuracy
            fpr, tpr, thresholds = roc_curve(gt, similarity) 

            # find the 0.1, 0.01, 0.001, 0.0001 position
            for x in range(len(fpr)-1,1,-1):
                if fpr[x]==0.1 or (fpr[x]>0.1 and fpr[x-1]<=0.1):
                    tprRes[i*70+j*10+sidx, 0] = tpr[x]

                if fpr[x]==0.01 or (fpr[x]>0.01 and fpr[x-1]<=0.01):
                    tprRes[i*70+j*10+sidx, 1] = tpr[x]

                if fpr[x]==0.001 or (fpr[x]>0.001 and fpr[x-1]<=0.001):
                    tprRes[i*70+j*10+sidx, 2] = tpr[x]

                if fpr[x]==0.0001 or (fpr[x]>0.0001 and fpr[x-1]<=0.0001):
                    tprRes[i*70+j*10+sidx, 3] = tpr[x]

        for k in range(4):
            finalTprRes[i*7+j,k] = np.mean(tprRes[i*70+j*10:i*70+(j+1)*10, k]) 
            finalTprRes[i*7+j,k+4] = finalTprRes[i*7+j,k] - np.max(tprRes[i*70+j*10:i*70+(j+1)*10, k]) 
        finalTprRes[i*7+j,8] = np.mean(tprRes[i*70+j*10:i*70+(j+1)*10, 4]) 
        finalTprRes[i*7+j,9] = finalTprRes[i*7+j,8] - np.min(tprRes[i*70+j*10:i*70+(j+1)*10, 4])





tprResOut = open('IJBAtprPoolingsSelectNumRes.txt', 'w')
for i in range(3):
    for j in range(7):
        for k in range(10):
            for x in range(4):
                print >> tprResOut, tprRes[i*70+j*10+k, x],
                print >> tprResOut, '\t',
            print >> tprResOut, ' ' 
tprResOut.close()

finalTprResOut = open('IJBAfinaltprPoolingsSelectNumRes.txt', 'w')
for i in range(3):
    for j in range(7):
        for x in range(4):
            print >> finalTprResOut,  str(round(finalTprRes[i*7+j, x],3)) + '+_' + str(round(finalTprRes[i*7+j, x+4],3)),
            print >> finalTprResOut, '\t',
        print >> finalTprResOut,  str(round(finalTprRes[i*7+j, 8],3)) + '+_' + str(round(finalTprRes[i*7+j, 9],3)),
        print >> finalTprResOut, ' ' 
finalTprResOut.close() 
