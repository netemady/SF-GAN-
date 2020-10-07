import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from scipy.stats.stats import pearsonr

# In[ICLR20]
def evaluate_n(predFC_np,empiricalFC_np, normalize=False):
    predFC=torch.from_numpy(predFC_np)
    empiricalFC = torch.from_numpy(empiricalFC_np)
    testSize = predFC.shape[0]
    nodeSize = predFC.shape[1]
    pearsonCorrelations = np.zeros(testSize)
    diff = np.zeros((testSize, int(nodeSize * (nodeSize-1) / 2)))
    for row in range(testSize):
        if normalize:
            predfc = getNormalizedMatrix(predFC[row])
            empiricalfc = getNormalizedMatrix(empiricalFC[row])
        else:
            predfc = predFC[row]
            empiricalfc = empiricalFC[row]
        predict_FC_vec = triu2vec(predfc.cpu(), diag=1)
        empirical_FC_vec = triu2vec(empiricalfc, diag=1).detach().cpu().numpy().reshape(1, predict_FC_vec.shape[0])
        predict_fc = predict_FC_vec.detach().numpy().reshape(1, predict_FC_vec.shape[0])
        diff[row, :] = empirical_FC_vec - predict_fc
        (pearson, p_val) = pearsonr(empirical_FC_vec.flatten(), predict_fc.flatten())
        pearsonCorrelations[row] = pearson
    return pearsonCorrelations,diff

# In[ICLR20]
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



def load_SC_FC_from_file(fileName):
    trainSize = 700
    allData = np.loadtxt(fileName, delimiter=',')
    sizeNode = 68
    sizeConnectivity = sizeNode * sizeNode
    row_count = allData.shape[0]

    # compute laplacian
    sc_mat = np.zeros((row_count, sizeNode, sizeNode)) # subjects * n * n
    fc_mat = np.zeros((row_count, sizeNode, sizeNode)) # subjects * n * n
    for row in range(row_count):
        sc_mat[row, :, :] = allData[row, 1:sizeConnectivity + 1].reshape((sizeNode, sizeNode))
        fc_mat[row, :, :] = allData[row, sizeConnectivity + 1:2 * sizeConnectivity + 1].reshape((sizeNode, sizeNode))
    sc_train = sc_mat[:trainSize,:,:]; sc_test = sc_mat[trainSize:,:,:]
    fc_train = fc_mat[:trainSize,:,:]; fc_test = fc_mat[trainSize:,:,:]

    return sc_train,sc_test,fc_train,fc_test,sc_mat,fc_mat,sizeNode,sizeConnectivity,row_count

# def evaluateCorrelation(prediction,empirical):
#     if torch.is_tensor(prediction):
#         pred = prediction.numpy()
#     else:
#         pred = prediction
#     if torch.is_tensor(empirical):
#         empi = empirical.numpy()
#     else:
#         empi = empirical
#     (pearson, p_val) = pearsonr(triu2vec_np(empi, diag=1), triu2vec_np(pred,diag=1))
#     return pearson

def pearsonCorrelation(x,y):
    vx = x - torch.mean(x)
    vy = y - torch.mean(y)
    corr = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)))
    return corr
def fc_loss(predictFC_lamb, trainFC, trainSC_u):
    # return torch.mean(torch.pow((x - y), 2))
    sampleCount = trainSC_u.__len__()
    loss = 0
    for i in range(sampleCount):
        predictFC=torch.mm(torch.mm(trainSC_u[i],torch.diag(predictFC_lamb[i,:])),trainSC_u[i].t())
        predict_val = predictFC[torch.triu(torch.ones(predictFC_lamb.shape[1],predictFC_lamb.shape[1])==0)]
        train_val = trainFC[i][torch.triu(torch.ones(predictFC_lamb.shape[1],predictFC_lamb.shape[1])==0)]
        loss += (1- pearsonCorrelation(predict_val,train_val))
        # loss.gr
        print(i,loss)
    return loss
def mse_loss(x, y):
    return torch.mean(torch.pow((x - y), 2))
    # return torch.frobenius_norm(predictFC,trainFC)

def vec2triu(vec):
    n = int((-1+(1+8* vec.shape[0])**0.5)/2)
    # print(vec.shape,n)
    matrix = torch.zeros(n,n)
    matrix[torch.triu(torch.ones(n,n))==1] = vec
    return matrix
def triu2vec(matrix,diag = 0):
    vec = matrix[torch.triu(torch.ones(matrix.shape[0],matrix.shape[1]),diagonal=diag)==1]
    return vec
def triu2vec_np(matrix,diag = 0):
    vec = matrix[np.triu(np.ones(matrix.shape),k = diag)==1]
    return vec
def getLaplacian(A,normalized = False):
    D_vec = torch.sum(torch.abs(A),dim=0)
    D = torch.diag(D_vec)
    L = D - A
    if normalized:
        D_norm = torch.diag(D_vec ** (-0.5))
        L_norm = torch.mm(D_norm,torch.mm(L,D_norm))
        return L_norm
    else:
        return L
def getLaplacian_np(A,normalized = False):
    D_vec = np.sum(np.abs(A),axis=0)
    D = np.diag(D_vec)
    L = D - A
    if normalized:
        D_norm = np.diag(D_vec ** (-0.5))
        L_norm = np.dot(D_norm,np.dot(L,D_norm))
        return L_norm
    else:
        return L

def getNormalizedAdjcentMatrix(A):
    D_vec = torch.sum(torch.abs(A),dim=0)
    D_norm = torch.diag(D_vec ** (-0.5))
    A_norm = torch.mm(D_norm, torch.mm(A, D_norm))
    return  A_norm


# For both laplacian and adjacent Matrix
def getNormalizedMatrix(M):
    A = M
    diagIndices = torch.arange(0,M.shape[0])
    A[diagIndices,diagIndices]= 0
    D_vec = torch.sum(torch.abs(A),dim=0)
    D_norm = torch.diag(D_vec ** (-0.5))
    A_norm = torch.mm(D_norm, torch.mm(A, D_norm))
    A_norm[torch.isnan(A_norm)] = 0
    return  A_norm
# For both laplacian and adjacent Matrix
def getNormalizedMatrix_np(M):
    A = M
    # diagIndices = np.arange(0,M.shape[0])
    # A[diagIndices,diagIndices]= 0
    np.fill_diagonal(A,0)
    D_vec = np.sum(np.abs(A),axis=0)
    D_norm = np.diag(D_vec ** (-0.5))
    A_norm = np.dot(D_norm, np.dot(A, D_norm))
    A_norm[np.isnan(A_norm)] = 0
    return  A_norm
## In[train]
### Baseline network
def network8(activation = 1,nodes = [68, 136, 272, 272, 544, 272, 272, 136, 68]):

    if activation==1:
        net = nn.Sequential(
            nn.Linear(nodes[0], nodes[1]),nn.ReLU(),
            nn.Linear(nodes[1], nodes[2]),nn.ReLU(),
            nn.Linear(nodes[2], nodes[3]),nn.ReLU(),
            nn.Linear(nodes[3], nodes[4]),nn.ReLU(),
            nn.Linear(nodes[4], nodes[5]),nn.ReLU(),
            nn.Linear(nodes[5], nodes[6]),nn.ReLU(),
            nn.Linear(nodes[6], nodes[7]),nn.ReLU(),
            nn.Linear(nodes[7], nodes[8])
        )
    elif activation == 2:
        net = nn.Sequential(
            nn.Linear(nodes[0],nodes[1]),nn.Sigmoid(),
            nn.Linear(nodes[1], nodes[2]),nn.Sigmoid(),
            nn.Linear(nodes[2], nodes[3]),nn.Sigmoid(),
            nn.Linear(nodes[3], nodes[4]),nn.Sigmoid(),
            nn.Linear(nodes[4], nodes[5]),nn.Sigmoid(),
            nn.Linear(nodes[5], nodes[6]),nn.Sigmoid(),
            nn.Linear(nodes[6], nodes[7]),nn.Sigmoid(),
            nn.Linear(nodes[7], nodes[8])
        )
    elif activation == 3:
        net = nn.Sequential(
            nn.Linear(nodes[0],nodes[1]),nn.Tanh(),
            nn.Linear(nodes[1], nodes[2]),nn.Tanh(),
            nn.Linear(nodes[2], nodes[3]),nn.Tanh(),
            nn.Linear(nodes[3], nodes[4]),nn.Tanh(),
            nn.Linear(nodes[4], nodes[5]),nn.Tanh(),
            nn.Linear(nodes[5], nodes[6]),nn.Tanh(),
            nn.Linear(nodes[6], nodes[7]),nn.Tanh(),
            nn.Linear(nodes[7], nodes[8])
        )
    elif activation == 4:
        net = nn.Sequential(
            nn.Linear(nodes[0],nodes[1]),nn.Softplus(),
            nn.Linear(nodes[1], nodes[2]),nn.Softplus(),
            nn.Linear(nodes[2], nodes[3]),nn.Softplus(),
            nn.Linear(nodes[3], nodes[4]),nn.Softplus(),
            nn.Linear(nodes[4], nodes[5]),nn.Softplus(),
            nn.Linear(nodes[5], nodes[6]),nn.Softplus(),
            nn.Linear(nodes[6], nodes[7]),nn.Softplus(),
            nn.Linear(nodes[7], nodes[8])
        )
    else:
        net = nn.Sequential(
            nn.Linear(nodes[0], nodes[1]),
            nn.Linear(nodes[1], nodes[2]),
            nn.Linear(nodes[2], nodes[3]),
            nn.Linear(nodes[3], nodes[4]),
            nn.Linear(nodes[4], nodes[5]),
            nn.Linear(nodes[5], nodes[6]),
            nn.Linear(nodes[6], nodes[7]),
            nn.Linear(nodes[7], nodes[8])
        )

    return net