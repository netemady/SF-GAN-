
from sklearn.model_selection import KFold
import itertools
from helper_n0203 import evaluate_n, getNormalizedMatrix_np, getNormalizedMatrix, triu2vec

import argparse
import os
import scipy.misc
import scipy.io
import numpy as np

from model_day1day2_nofeat import graph2graph
import tensorflow as tf
import torch
from timeit import default_timer as timer

from sklearn import preprocessing


folder = 'SC_FC_dataset_0905/'

file = 'SC_GAMBLING_correlationFC'

data = np.load(folder + file + '.npz')
SC_ids = data['subjects']
SC_818 = data['sc']

lst1 = SC_ids

lst2 = scipy.io.loadmat('D:/Negar/sFC-reprocess/FC_rfMRI_D1/new2/Day1_subjects')
l2 = lst2['Day1_ids']

final_subjects_day1 = [value for value in lst1 if value in l2]

final_sc_day1 = SC_ids
final_fc_day1 = np.zeros((818,68,68))


for i in range(818):

    id = final_sc_day1[i]
    #a = scipy.io.loadmat('D:/Negar/sFC-reprocess/FC_rfMRI_D1/new2/'+str(id))
    a = scipy.io.loadmat('D:/Negar/sFC-reprocess/FC_rfMRI_D1/new2/' + str(id))
    final_fc_day1[i,:,:] = a['correlationFC']
    k=1

rawSC_ori = SC_818
rawFC_ori = final_fc_day1

rawSC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
rawFC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
#rawFC1 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
#rawFC2 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
rawFC2 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))

#data_4D_day1_batchofSubs = np.load('n4_tensor_saliency_batchOfSubs_day1.npy')
#np.save('data_4D_sub0-To-50_day1.npy', data_4D_day1_batchofSubs[0:50, :, :, :])


#data_4D_day2_batchofSubs = np.load('n4_tensor_saliency_batchOfSubs_day2-correctVersion.npy')
#np.save('data_4D_sub0-To-50_day2.npy', data_4D_day2_batchofSubs[0:50, :, :, :])


########## strategy 1: keep originalWeights and only upperTriangular values in sc, fc, and the gradients matrices.

########## strategy 3: keep originalWeights for intra layer (sc and fc), and put additional weight = 1.5, for inter layer values.


for i in range(rawFC_ori.shape[0]):
    # rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i]) * 100
    # rawFC[i] = getNormalizedMatrix_np(rawFC_ori[i]) * 100
    rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])
    rawFC[i] = rawFC_ori[i]


########## strategy 2: normalizedFC values and keep only upperT elements for all intra and inter layer.

#for i in range(rawFC_ori.shape[0]): #day1
'''for i in range(130, 818): #day2

    rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])

    rawFC2[i] = preprocessing.normalize(rawFC_ori[i])'''

#######################################################################

allSC = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2], 4))
allSC[:, :, :, 0] = rawSC
allFC = rawFC  # strategy1 and 3
#allFC = rawFC2  # strategy2

entire_size = allSC.shape[0]

# validation_size = int(entire_size / 5)
test_size = int(entire_size / 5)

# [train and hypertune using 3-folder cross-validation]
dataset_SC = allSC[list(range(test_size, entire_size))]
dataset_FC = allFC[list(range(test_size, entire_size))]

sc_train = dataset_SC
fc_train = dataset_FC


sc_test = allSC[list(range(0, test_size))]
fc_test = allFC[list(range(0, test_size))]

###################################################################

data_4D_batchofSubs = np.load('data_4D_sub0-To-50_day1.npy')
#data_4D_batchofSubs = np.load('data_4D_sub0-To-50_day2.npy')

s1 = 4*130  # start from this subj in train set, as grads are computed for the last batch of training data

for subject in range(s1, s1+50):

        f = open("input_day1_sub{}_strategy1.txt".format(subject), "w")
        f.write('*Vertices 68')

        for i in range(68):
            s = ""
            f.write("\n{} 'node {}' ".format(i + 1, i + 1))

        f.write('\n*Multilayer')

        layer1 = sc_train[subject, :, :, 0]
        layer2 = fc_train[subject, :, :]

        #f = open("input_day1_sub{}_upperT.txt".format(subject), "w")
        for i in range(68):
            for j in range(68):
                if j == i or j > i:
                    f.write("\n1 {} 1 {} {}".format(i + 1, j + 1, layer1[i, j]))

        #f.close()

        #f = open("input_day1_sub{}_upperT.txt".format(subject), "a")
        for i in range(68):
            for j in range(68):
                if j == i or j > i:
                    #if layer2[i,j]<0:
                        #f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, 0))
                       # continue
                    #else:
                        f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, layer2[i, j]))

        #data_4D_batchofSubs = np.load('data_4D_sub0-To-50_day2.npy')
        GAP_input = np.zeros((68, 68, 68, 68))

        data_4D = data_4D_batchofSubs[subject-s1, :, :, :]

        # zeroing out gradients before GAP
        # data_4D[data_4D<0] = 0

        m = 0

        for i in range(68):

            for k in range(68):
                GAP_input[i, :, k, :] = data_4D[m, :, :]
                m = m + 1

        GAP_input_tensor = tf.convert_to_tensor(GAP_input, dtype=tf.float32)

        gap = tf.keras.layers.GlobalAveragePooling2D()
        GAP_output = gap(GAP_input_tensor)

        init_op = tf.initialize_all_variables()

        # run the graph
        with tf.Session() as sess:
            sess.run(init_op)  # execute init_op
            GAP_output_value = sess.run(GAP_output)

        # zeroing out gradients after GAP
        GAP_output_value[GAP_output_value < 0] = 0

        # GAP_output_value_normalized = preprocessing.normalize(GAP_output_value)

        # 1 node1 2 node2 value
        #f = open("interlayer_day1_sub5_allLinks.txt", "w")
        for i in range(68):
            for j in range(68):
                if j == i or j > i:
                    f.write("\n1 {} 2 {} {}".format(i + 1, j + 1, GAP_output_value[i, j]))  #strategy 1, 2
                    #f.write("\n1 {} 2 {} {}".format(i + 1, j + 1, 1.3*GAP_output_value[i, j]))  #strategy 3


        f.close()


