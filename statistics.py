from sklearn.metrics import roc_curve
from sklearn.metrics import auc
import numpy as np
import matplotlib.pyplot as plt
import sys

def HTER(label, score, thred=0.5, EERtype ="hter"):
    scores = []
    FAR_SUM = []
    FRR_SUM = []
    TPR_SUM = []
    roc_EER = []
    for i in range(0, len(label)):
        tmp = []
        tmp1 = score[i]
        tmp2 = label[i]
        tmp.append(tmp1)
        tmp.append(tmp2)
        scores.append(tmp)
    #print score
    scores = sorted(scores);  # min->max
    #print scores
    sort_score = np.matrix(scores);
    #print sort_score
    alltrue = sort_score.sum(axis=0)[0,1];
    allfalse = len(scores) - alltrue;
    fa = allfalse;  #认为所有假样本都判成真的   判断阈值为0
    miss = 0;
    #print sort_score
    #print alltrue
    for i in range(0, len(scores)):
        # min -> max
        #小于等于当前score时为假  即判断阈值为当前score
        if sort_score[i, 1] == 1:   # 真的 判成 假的  
            miss += 1;
        else:  # 假的 判成 假的
            fa -= 1;

        FAR=float(fa)/allfalse;   #所有假样本中有多少被判成真的   FPR
        FRR=float(miss)/alltrue;  #所有真样本中有多少被判成假的   
        TPR=1-FRR   # 所有真样本中有多少被判成真的  
        FAR_SUM.append(FAR)
        FRR_SUM.append(FRR)
        TPR_SUM.append(TPR)
    cords = list(zip(FAR_SUM, FRR_SUM, sort_score[:,0]))
    #print cords
    for item in cords:
        item_fpr, item_fnr, item_thd = item
        roc_EER.append(abs(item_thd - thred)) # 计算阈值与0.5的差值  没懂为什么要取与0.5最接近的FAR和FRR去算HTER
    eer_index = np.argmin(roc_EER)  #获取到差值最小时的位置
    eer_fpr, eer_fnr, thd = cords[eer_index]
    hter = (eer_fpr + eer_fnr)/2
    # print (eer_fpr,eer_fnr,thd)
    print (EERtype + " " + 'HTER is :%f %%' % (hter*100))
    print (EERtype + " " + 'FAR is :%f' % eer_fpr)
    print (EERtype + " " + 'FRR is :%f' % eer_fnr)
    return hter

def EER(label, score, EERtype="eer"):
    scores = []
    FAR_SUM = []
    FRR_SUM = []
    TPR_SUM = []
    for i in range(0, len(label)):
        tmp = []
        tmp1 = score[i]
        tmp2 = label[i]
        tmp.append(tmp1)
        tmp.append(tmp2)
        scores.append(tmp)

    scores = sorted(scores);  # min->max
    sort_score = np.matrix(scores);
    minIndex = sys.maxsize;
    minDis = sys.maxsize;
    minTh = sys.maxsize;
    eer = sys.maxsize;
    alltrue = sort_score.sum(axis=0)[0,1];
    allfalse = len(scores) - alltrue;
    fa = allfalse;
    miss = 0;
    #print sort_score
    #print alltrue
    for i in range(0, len(scores)):
        # min -> max
        if sort_score[i, 1] == 1:
            miss += 1;
        else:
            fa -= 1;

        FAR=float(fa)/allfalse;
        FRR=float(miss)/alltrue;
        TPR=1-FRR
        FAR_SUM.append(FAR)
        FRR_SUM.append(FRR)
        TPR_SUM.append(TPR)
       
        if abs(FAR - FRR) < minDis:
            minDis = abs(FAR - FRR)
            eer = min(FAR,FRR) #FAR与FRR最接近时 二者较小的值
            minTh = sort_score[i, 0]  # 当前阈值
    roc_auc = auc(FAR_SUM, TPR_SUM)

    #plt.plot(FAR_SUM, TPR_SUM, lw=1, label='ROC(area = %f)'%(roc_auc))
    #plt.plot(FAR_SUM, TPR_SUM, lw=1)
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.savefig("test_result")
    #plt.show()

    #plt.plot(FAR_SUM, FRR_SUM, lw=1)
    #plt.plot([0, 1], [1, 0], '--', color=(0.6, 0.6, 0.6), label='Luck')
    #plt.savefig("test_result2")
    #plt.show()
    print (EERtype + " " + 'EER is :%f %%' % (eer*100))
    print (EERtype + " " + 'AUC is :%f' % roc_auc)
    print (EERtype + " " + 'thd is :%f' % minTh)

    return roc_auc, eer, minTh
