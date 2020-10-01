# Created by qli10 at 7/10/2019
import numpy as np
# import torch
from helper import *
#from helper_n0203 import evaluate_n
from sklearn.metrics import mean_squared_error
from timeit import default_timer as timer

def compute_r2(predFC, fc_input):
    diff = np.zeros((predFC.shape[0], predFC.shape[1], predFC.shape[2]))
    diff2 = np.zeros((predFC.shape[0], predFC.shape[1], predFC.shape[2]))

    for i in range(predFC.shape[0]):
        diff[i, :, :] = np.subtract(predFC[i, :, :], fc_input[i, :, :])
        diff[i, :, :] = np.power(diff[i, :, :], 2)

    nom = np.sum(diff)
    y_bar = np.mean(fc_input)
    for i in range(predFC.shape[0]):
        diff2[i, :, :] = np.subtract(fc_input[i, :, :], y_bar)
        diff2[i, :, :] = np.power(diff2[i, :, :], 2)

    denom = np.sum(diff2)
    return (1 - (nom / denom))

def compute_mse(predFC, fc_input):
    predFC_s = predFC.shape[0]
    sum_mse = 0
    for i in range(predFC_s):
        a1 = predFC[i, :, :]
        a2 = fc_input[i]
        sum_mse = mean_squared_error(a1, a2) + sum_mse
        # a1.append(predFC[i+1].flatten)
        # a2.append(fc_input[i])
    return (np.mean(sum_mse))

def evaluate(predFC,empiricalFC, normalize=False):
    testSize = predFC.shape[0]
    nodeSize = predFC.shape[1]
    # pearsonCorrelations = np.zeros(testSize)
    # diff = np.zeros((testSize, int(nodeSize * (nodeSize-1) / 2)))
    # for row in range(testSize):
    if normalize:
        predfc = getNormalizedMatrix_np(predFC)
        empiricalfc = getNormalizedMatrix_np(empiricalFC)
    else:
        predfc = predFC
        empiricalfc = empiricalFC
    predict_FC_vec = predfc[np.triu(np.ones(predfc.shape),k = 1)==1]#triu2vec(predfc.cpu(), diag=1)
    empirical_FC_vec = empiricalfc[np.triu(np.ones(empiricalfc.shape),k = 1)==1]
    # empirical_FC_vec = triu2vec(empiricalfc, diag=1).detach().cpu().numpy().reshape(1, predict_FC_vec.shape[0])
    predict_fc = predict_FC_vec.reshape((1, predict_FC_vec.shape[0]))
    # diff[row, :] = empirical_FC_vec - predict_fc
    (pearson, p_val) = pearsonr(empirical_FC_vec.flatten(), predict_fc.flatten())
    # pearsonCorrelations = pearson
    return pearson
# In[Xiaojie's implementation]
#
# paper 2014
# ''Abdelnour F, Voss HU, Raj A.
#     Network diffusion accurately models the relationship between structural and functional brain connectivity networks. Neuroimage.
#      2014 Apr 15;90:335-47.''
#
#   paper 2018
#   eigen model in paper:
#     ''Abdelnour F, Dayan M, Devinsky O, Thesen T, Raj A.
#     Functional brain connectivity is predictable from anatomic network's Laplacian eigen-structure. NeuroImage.
#      2018 May 15;172:728-39.''



def predict_fc_diff(sc, beta_t):
    # sc_lap = laplacian(sc)
    # a,b=np.linalg.eig(sc_lap)
    # a[0]=0
    # a[1]=0
    # sc_lap=np.dot(np.dot(a,b),np.transpose(b))
    fc = np.exp(-1 * beta_t * sc)
    return fc


def predict_fc_eigen(sc, a, beta_t, b):
    # sc_lap = laplacian(sc)
    # a,b=np.linalg.eig(sc_lap)
    # a[0]=0
    # a[1]=0
    # sc_lap=np.dot(np.dot(np.diag(a),b),np.transpose(b))
    fc = a * np.exp(-beta_t * sc) + b * np.eye(sc.shape[1])
    return fc


# fitting the paramters
def curve_fiting_eigen(sc_train, fc_train):
    def func(x, a, b, c):
        return a * np.exp(-b * x) + c

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, sc_train, fc_train,maxfev=10000)
    return popt


def curve_fiting_diff(sc_train, fc_train):
    def func(x, b):
        return np.exp(-b * x)

    from scipy.optimize import curve_fit
    popt, pcov = curve_fit(func, sc_train, fc_train)
    return popt

def preprocessNumpy(rawSC, rawFC, useLaplacian=True, normalized=False, useAbs=False):
    rows = rawSC.shape[0]
    sizeNode = rawSC.shape[1]
    SC = np.zeros((rows, sizeNode, sizeNode))
    SC_u = np.zeros((rows, sizeNode, sizeNode))
    FC_u = np.zeros((rows, sizeNode, sizeNode))
    SC_lamb = np.zeros((rows, sizeNode))
    FC = np.zeros((rows, sizeNode, sizeNode))
    FC_lamb = np.zeros((rows, sizeNode))

    for row in range(rows):
        if useLaplacian:
            sc = getLaplacian_np(rawSC[row], normalized=True)
        else:
            sc = rawSC[row]
        lamb_sc, u_sc = np.linalg.eigh(sc)
        SC[row] = sc
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        if useAbs:
            fc = np.abs(rawFC[row])
        else:
            fc = rawFC[row]
        if useLaplacian:
            fc = getLaplacian_np(fc, normalized=normalized)
        FC[row] = fc
        lamb_fc, u_fc = np.linalg.eigh(fc)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u
# In[evaluate paper 2014 paper 2018]
def paper20142018(raw_sc_train,raw_fc_train,raw_sc_test,raw_fc_test, file, useLaplacian = True,normalized = False,useAbs = False):

    # paper 2014
    start = timer()
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                    useLaplacian=useLaplacian,
                                                                                    normalized=normalized,
                                                                                    useAbs=useAbs)

    popt_diff = curve_fiting_diff(np.reshape(trainSC_lamb, [-1])[2:], np.reshape(trainFC_lamb, [-1])[2:]) # 2014
    end = timer()
    train_time_2014 = end - start

    #predict fc
    start = timer()
    testSC, testSC_lamb, testSC_u, empiricalFC, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test, raw_fc_test,
                                                                                             useLaplacian=useLaplacian,
                                                                                             normalized=normalized,
                                                                                             useAbs=useAbs)
    pred_fc_diff=np.zeros((empiricalFC.shape[0],empiricalFC.shape[1],empiricalFC.shape[2]))  # paper 2014
    #corr_diff=np.zeros(empiricalFC.shape[0])
    for i in range(len(testSC)):
        pred_fc_diff[i]=predict_fc_diff(testSC[i],popt_diff[0])
        #corr_diff[i] = evaluate(pred_fc_diff[i],empiricalFC[i]) #2014

    s_pred = "fc_pred_2014_%s" % file
    np.save(s_pred, pred_fc_diff)

    end = timer()
    test_time_2014 = end - start



        # paper 2018
    start = timer()
    trainSC, trainSC_lamb, trainSC_u, trainFC, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                         useLaplacian=useLaplacian,
                                                                                         normalized=normalized,
                                                                                         useAbs=useAbs)

    popt_eigen = curve_fiting_eigen(np.reshape(trainSC_lamb, [-1])[2:], np.reshape(trainFC_lamb, [-1])[2:])  # 2018
    end = timer()
    train_time_2018 = end - start

    # predict fc
    start = timer()
    testSC, testSC_lamb, testSC_u, empiricalFC, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test,
                                                                                                 raw_fc_test,
                                                                                                 useLaplacian=useLaplacian,
                                                                                                 normalized=normalized,
                                                                                                 useAbs=useAbs)

    pred_fc_eigen = np.zeros((empiricalFC.shape[0], empiricalFC.shape[1], empiricalFC.shape[2]))  # paper 2018
    #corr_eigen = np.zeros(empiricalFC.shape[0])

    for i in range(len(testSC)):
         pred_fc_eigen[i] = predict_fc_eigen(testSC[i], popt_eigen[0], popt_eigen[1], popt_eigen[2])
         #corr_eigen[i] = evaluate(pred_fc_eigen[i], empiricalFC[i])  # 2018

    s_pred = "fc_pred_2018_%s" % file
    np.save(s_pred, pred_fc_eigen)

    end = timer()
    test_time_2018 = end - start


    #paper_2014_r2 = compute_r2(pred_fc_diff, raw_fc_test)
    #paper_2018_r2 = compute_r2(pred_fc_eigen, raw_fc_test)


    #paper_2014_mse = compute_mse(pred_fc_diff, raw_fc_test)
    #paper_2018_mse = compute_mse(pred_fc_eigen, raw_fc_test)

    #print("paper 2014 mse : %.4f" % (paper_2014_mse))
    #print("paper 2018 mse : %.4f" % (paper_2018_mse))

    #print ("paper 2014 correlation: %.4f ± %.4f" % (np.nanmean(corr_diff),np.nanstd(corr_diff)))
    #print ("paper 2018 correlation: %.4f ± %.4f" % (np.nanmean(corr_eigen),np.nanstd(corr_eigen.std())))
    #return paper_2014_mse, paper_2018_mse, paper_2014_r2, paper_2018_r2, np.nanmean(corr_diff),np.nanmean(corr_eigen)
    return train_time_2014, test_time_2014, train_time_2018, test_time_2018


# In[paper 2016]
"""
Created on Wed Apr 17 16:50:25 2019
@author: gxjco

This model is on paper:
    "Meier J, Tewarie P, Hillebrand A, Douw L, van Dijk BW, Stufflebeam SM, Van Mieghem P.
    A mapping between structural and functional brain networks. Brain connectivity. 2016 May 1;6(4):298-311."
"""

def predict_fc_mapping(sc,k,c):
    W=np.zeros((sc.shape[0],sc.shape[1]))
    for i in range(1,k+1):
        W=W+c[i]*np.power(sc,i)
    return W+np.eye(sc.shape[0])*c[0]

def paramter(sc,fc,k):
    W=np.reshape(fc,[-1])
    A=np.zeros((k+1,W.shape[0]))
    for i in range(k+1):
        if i==0:
            A[i]=np.ones(W.shape[0])
        else:
            A[i]=np.reshape(np.power(sc,i),[-1])
    c=np.linalg.lstsq(np.transpose(A),W)[0]
    return c
def paper2016(raw_sc_train,raw_fc_train,raw_sc_test,raw_fc_test, file, useLaplacian = True,normalized = True,useAbs = False,trainSize = 700):

    start = timer()
    sc_train, trainSC_lamb, trainSC_u, fc_train, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
                                                                                         useLaplacian=useLaplacian,
                                                                                         normalized=normalized,
                                                                                         useAbs=useAbs)

    k = 3
    c = paramter(sc_train, fc_train, k)
    end = timer()
    train_time = end - start

    start = timer()

    sc_test, testSC_lamb, testSC_u, fc_test, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test,
                                                                                                 raw_fc_test,
                                                                                                 useLaplacian=useLaplacian,
                                                                                                 normalized=normalized,
                                                                                                 useAbs=useAbs)

    pred_fc_mapping=np.zeros((sc_test.shape[0],sc_test.shape[1],sc_test.shape[2]))
    corr_mapping=np.zeros(fc_test.shape[0])
    for i in range(sc_test.shape[0]):
       pred_fc_mapping[i]=predict_fc_mapping(sc_test[i],k,c)
       corr_mapping[i] = evaluate(pred_fc_mapping[i], fc_test[i])

    s_pred = "fc_pred_2016_%s" %file
    np.save(s_pred, pred_fc_mapping)

    end = timer()
    test_time = end - start
        
    #paper_2016_r2 = compute_r2(pred_fc_mapping, raw_fc_test)

    #paper_2016_mse = compute_mse(pred_fc_mapping, raw_fc_test)

    #print(np.nanmean(corr_mapping))
    #print(paper_2016_mse)
    
    #return train_time, test_time, corr_mapping.mean()
    return train_time, test_time
# In[paper 2008]
"""
Created on Wed Apr 17 15:28:53 2019
@author: gxjco

This model is on paper:
    ''Galán RF. On how network architecture determines the dominant patterns of spontaneous neural activity. PloS one.
    2008 May 14;3(5):e2148."

"""


def predict_fc_linear(sc,b,alpha,t):
    A=(1-alpha*t)*np.eye(sc.shape[0])+t*sc
    D,L=np.linalg.eig(A)
    L_inv=np.linalg.inv(L)
    Q=np.eye(sc.shape[0])*b
    Q_star=np.dot(np.dot(L_inv,Q),np.transpose(L_inv))
    P=np.zeros((sc.shape[0],sc.shape[1]))
    for i in range(len(D)):
        for j in range(len(D)):
            P[i][j]=Q_star[i][j]/(1-D[i]*D[j])
    C=np.dot(np.dot(L,P),np.transpose(L))
    return A
def paper2008(raw_sc_train,raw_fc_train,raw_sc_test,raw_fc_test, file, useLaplacian = True,normalized = False,useAbs = False,trainSize = 700):


    # sc_train, trainSC_lamb, trainSC_u, fc_train, trainFC_lamb, trainFC_u = preprocessNumpy(raw_sc_train, raw_fc_train,
    #                                                                                      useLaplacian=useLaplacian,
    #                                                                                      normalized=normalized,
    #                                                                                      useAbs=useAbs)
    start = timer()
    sc_test, testSC_lamb, testSC_u, fc_test, empricalFC_lamb, empiricalFC_u = preprocessNumpy(raw_sc_test,
                                                                                                 raw_fc_test,
                                                                                                 useLaplacian=useLaplacian,
                                                                                                 normalized=normalized,
                                                                                                 useAbs=useAbs)


    pred_fc_linear=np.zeros((sc_test.shape[0],sc_test.shape[1],sc_test.shape[2]))
    corr_linear=np.zeros(fc_test.shape[0])
    b = 1
    for i in range(sc_test.shape[0]):
       pred_fc_linear[i] = predict_fc_linear(sc_test[i], b, 2, 100)
       #corr_linear[i] = evaluate(pred_fc_linear[i], fc_test[i])

    s_pred = "fc_pred_2008_%s" % file
    np.save(s_pred, pred_fc_linear)

    end = timer()

    #paper_2008_r2 = compute_r2(pred_fc_linear, raw_fc_test)
    #print("paper 2008 r2: %.4f" % (paper_2008_r2))

    #paper_2008_mse = compute_mse(pred_fc_linear, raw_fc_test)
    #print("paper 2008 mse: %.4f" % (paper_2008_mse))
        
    #print ("paper 2008 correlation: %.4f ± %.4f" % (corr_linear.mean(),corr_linear.std()))
    #return paper_2008_r2, paper_2008_mse, corr_linear.mean()
    return (end-start)
