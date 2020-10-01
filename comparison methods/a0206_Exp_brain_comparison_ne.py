import numpy as np
from sklearn.model_selection import KFold
import itertools
from helper import *
from a0206_comparison_functions_ne import paper20142018,paper2016,paper2008

def loss_correlation(source, target):
    # print('loss_correlation shape:',source.shape,target.shape)
    row_idx, col_idx = torch.triu_indices(source.shape[1], source.shape[2],offset=1)
    x = source[:, row_idx, col_idx]
    y = target[:, row_idx, col_idx]
    vx = x - torch.mean(x, 1, keepdim=True)
    vy = y - torch.mean(y, 1, keepdim=True)
    xy_cov = torch.bmm(vx.view(vx.shape[0], 1, vx.shape[1]), vy.view(vy.shape[0], vy.shape[1], 1), ).view(vx.shape[0])
    cost = xy_cov / torch.mul(torch.sqrt(torch.sum(vx ** 2, dim=1)), torch.sqrt(torch.sum(vy ** 2, dim=1)))
    loss = 1 - torch.mean(cost)
    # print('loss:',loss)
    return loss

# In[]
# both SC and FC should be in laplacian format
def preprocess(SC, FC):
    rows = SC.shape[0]
    sizeNode = SC.shape[1]
    device=SC.device
    SC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_D = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
    FC_u = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    SC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)
    # FC = torch.zeros(rows, sizeNode, sizeNode, dtype=torch.float32, device=device)
    FC_lamb = torch.zeros(rows, sizeNode, dtype=torch.float32, device=device)

    for row in range(rows):
        # if useLaplacian:
        #     sc = getLaplacian(rawSC[row], normalized=normalized)
        # else:
        #     sc = rawSC[row]
        sc = SC[row]
        A = SC[row]
        diagIndices = torch.arange(0, A.shape[0])
        A[diagIndices, diagIndices] = 0
        D_vec = torch.sum(torch.abs(A), dim=0)
        SC_D[row] = D_vec

        lamb_sc, u_sc = torch.symeig(sc, eigenvectors=True)
        # SC[row] = sc
        SC_u[row] = u_sc
        SC_lamb[row, :] = lamb_sc
        # if useAbs:
        #     fc = torch.abs(rawFC[row])
        # else:
        #     fc = rawFC[row]
        # if useLaplacian:
        #     fc = getLaplacian(fc, normalized=normalized)
        # FC[row] = fc
        fc = FC[row]
        lamb_fc, u_fc = torch.symeig(fc, eigenvectors=True)
        FC_lamb[row, :] = lamb_fc
        FC_u[row] = u_fc
    SC_lamb = SC_lamb.reshape((SC_lamb.shape[0], SC_lamb.shape[1], 1))  # 700*68*1
    return SC,SC_lamb,SC_u,FC,FC_lamb,FC_u,SC_D

# In[]
folder = '../SC_FC_dataset_0905/'


#files = ['SC_RESTINGSTATE_correlationFC','SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_correlationFC', 'SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
#files = ['SC_RELATIONAL_CorrelationFC']#'SC_EMOTION_correlationFC', 'SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
#files = ['SC_RESTINGSTATE_correlationFC','SC_EMOTION_correlationFC','SC_GAMBLING_correlationFC', 'SC_LANGUAGE_correlationFC', 'SC_MOTOR_correlationFC','SC_RELATIONAL_correlationFC', 'SC_SOCIAL_correlationFC',  'SC_WM_correlationFC']

files = ['SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_partialCorrelationFC_L2',
         'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_partialCorrelationFC_L2',
         'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_partialCorrelationFC_L2']

#files = ['SC_EMOTION_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L1',
 #        'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L1',
  #       'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L1']

#files = ['SC_EMOTION_correlationFC']

results_train_time = np.zeros((4, 20))
results_test_time = np.zeros((4, 20))
k = 0


#files = ['SC_WM_partialCorrelationFC_L1']

datasetIdx = 0
for file in files:

  for itr in range(1):

    #print('######### Current dataset: ',datasetIdx,len(files),file)
    datasetIdx+=1
    data = np.load(folder+file+'.npz')
    try:
        rawSC = data['sc']
        rawFC = data['fc']
    except:
        print("rfMRI dataset!!")
        rawSC = data['rawSC']
        rawFC = data['rawFC']

    rawSC2 = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2]))
    rawFC2 = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2]))

    for i in range(rawFC.shape[0]):
        rawSC2[i] = getNormalizedMatrix_np(rawSC[i])
        rawFC2[i] = rawFC[i]

    allSC = rawSC2
    allFC = rawFC2


# In[train and evaluate using 5-folder cross-validation]
    entire_size = rawSC.shape[0]

    test_size = 100

    #validation_size = int(entire_size / 5)
    #validation_index = list(range(validation_size))
    dataset_SC = allSC[list(range(test_size,entire_size))]
    dataset_FC = allFC[list(range(test_size,entire_size))]

    sc_train = dataset_SC
    fc_train = dataset_FC

    sc_test = allSC[list(range(0, test_size))]
    fc_test = allFC[list(range(0, test_size))]

    s_fc = "fc_true_%s" %file
    #np.save(s_fc, fc_test)

    cvFolers = 1
    #result = np.zeros((4,cvFolers)) # [paper2008;2014;2016;2018]
    
    cv_idx = 0
    
    
    #paper_2014_mse, paper_2018_mse, paper_2014_r2, paper_2018_r2, result[1,cv_idx],result[3,cv_idx] = paper20142018(sc_train,fc_train,sc_test,fc_test,useLaplacian = True,normalized = False,useAbs = False)
    #paper_2016_r2, paper_2016_mse, result[2,cv_idx] = paper2016(sc_train,fc_train,sc_test,fc_test,useLaplacian = False,normalized = False,useAbs = False)
    
    #train_time_2014, test_time_2014, train_time_2018, test_time_2018 = paper20142018(sc_train,fc_train,sc_test,fc_test, file, useLaplacian = True,normalized = False,useAbs = False)
    #train_time_2016, test_time_2016 = paper2016(sc_train, fc_train, sc_test, fc_test, file, useLaplacian=False, normalized=False, useAbs=False)
    train_time_2008 = paper2008(sc_train,fc_train,sc_test,fc_test, file, useLaplacian = False,normalized = False,useAbs = False)
    
    '''results_train_time[0, itr] = train_time_2008
    results_train_time[1, itr] = train_time_2014
    results_train_time[2, itr] = train_time_2016
    results_train_time[3, itr] = train_time_2018
    
    results_test_time[0, itr] = train_time_2008
    results_test_time[1, itr] = test_time_2014
    results_test_time[2, itr] = test_time_2016
    results_test_time[3, itr] = test_time_2018'''
