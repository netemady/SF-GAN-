# Created by netemady on 041321

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

def hyperparams(lambda1_value):
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
    parser.add_argument('--epoch', dest='epoch', type=int, default=1, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=130, help='# images in batch')
    # parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--niter', dest='niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.01, help='initial learning rate for adam')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--flip', dest='flip', type=bool, default=True,
                        help='if flip the images for data argumentation')
    parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50,
                        help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
    parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000,
                        help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
    parser.add_argument('--print_freq', dest='print_freq', type=int, default=50,
                        help='print the debug information every print_freq iterations')
    parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False,
                        help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False,
                        help='f 1, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True,
                        help='iter into serial image list')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./', help='test sample are saved here')
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=lambda1_value, help='weight on L1 term in objective')
    parser.add_argument('--L1_lambda2', dest='L1_lambda2', type=float, default=0,
                        help='weight on L1 term in objective for deconv weights')
    # parser.add_argument('--L1_lambda3', dest='L1_lambda3', type=float, default=70, help='weight on L1 term in objective for diff of deconv weights')
    parser.add_argument('--train_dir', dest='train_dir', default='./', help='train sample are saved here')
    parser.add_argument('--image_size', dest='image_size', default=[20, 20], help='input graph size')
    parser.add_argument('--output_size', dest='output_size', default=[20, 20], help='output graph size')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint20',
                        help='models are saved here')
    return parser.parse_args()


#args = hyperparams()

epoch_values = [1, 3, 5, 10]
lr_values = [0.01, 0.005, 0.001, 0.0005, 0.0001]
lambda1_values = [0, 1e0, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8]

for lr_value in lr_values:
    for lambda1_value in lambda1_values:
        for epoch_value in epoch_values:

            args = hyperparams(lambda1_value)


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
                a = scipy.io.loadmat('D:/Negar/sFC-reprocess/FC_rfMRI_D1/new2/'+str(id))
                final_fc_day1[i,:,:] = a['correlationFC']
                k=1

            rawSC_ori = SC_818
            rawFC_ori = final_fc_day1

            rawSC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
            rawFC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
            #rawFC1 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
            #rawFC2 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
            #rawFC3 = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))


            for i in range(rawFC_ori.shape[0]):
                # rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i]) * 100
                # rawFC[i] = getNormalizedMatrix_np(rawFC_ori[i]) * 100
                rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])
                rawFC[i] = rawFC_ori[i]
                #rawFC2[i] = getNormalizedMatrix_np(rawFC_ori[i])
                #rawFC3[i] = preprocessing.normalize(rawFC_ori[i])


            allSC = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2], 4))
            allSC[:, :, :, 0] = rawSC
            allFC = rawFC

            # normalize FC
            #normalized_FC = preprocessing.normalize(rawFC)
            #allFC = normalized_FC

            #FC1_0 = rawFC1[0]
            #FC2_0 = rawFC2[0]
            #FC3_0 = rawFC3[0]
            #allFC = rawFC3

            entire_size = allSC.shape[0]
            #entire_size = 695

            # validation_size = int(entire_size / 5)
            test_size = int(entire_size / 5)

            # [train and hypertune using 3-folder cross-validation]
            dataset_SC = allSC[list(range(test_size, entire_size))]
            dataset_FC = allFC[list(range(test_size, entire_size))]

            sc_train = dataset_SC
            fc_train = dataset_FC

            sc_test = allSC[list(range(0, test_size))]
            fc_test = allFC[list(range(0, test_size))]

            s = fc_train[5, :, :]
            a = 9

            checkpoint_dir = 'brain_checkpoint_data_day1'
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)

            tf.reset_default_graph()
            args.image_size = [sc_train[0].shape[0], sc_train[0].shape[0]]
            args.output_size = [fc_train[0].shape[0], fc_train[0].shape[0]]

            args.checkpoint_dir = checkpoint_dir
            ################################################################
            with tf.Session() as sess:
                # print('Training the model for ', folder, file)
                model = graph2graph(sess, batch_size=args.batch_size, L1_lambda = args.L1_lambda, checkpoint_dir=args.checkpoint_dir,
                                    sample_dir=args.sample_dir, test_dir=args.test_dir, train_dir=args.train_dir,
                                    image_size=args.image_size, output_size=args.output_size)

                model.train(args, sc_train, fc_train)

                #np.save('n4_tensor_saliency_batchOfSubs_day2.npy', a)
                #np.save('n4_tensor_saliency_subject1_day2.npy', b)
                #np.save('n4_tensor_saliency_subject2_day2.npy', c)

                #####################  Test ########################

                Ra_t = model.test(args, sc_test, fc_test)
                #runtime_test, Ra_t = model.test(args, sc_test, fc_test)
                predFC = Ra_t[:, :, :, 0]
            # predFC_train = Ra_t_train[:, :, :, 0]
            #s1 = "prediction_nofeat_test_%s" % (file)#
            #np.save(s1, predFC)
            # np.save("pred_nofeat",Ra_t)
            # predFC = vec2Adjacent(Ra_t,fc_test.shape[1])


            correlations, diff = evaluate_n(predFC, fc_test)
            # correlations_train, diff_train = evaluate_n(predFC_train, fc_train)
            # predfc, empiricalfc = tonp(predFC, fc_test)
            # print('evaluation_input_file:', file)
            # np.savez_compressed(file, a=predfc, b=empiricalfc)

            pearsonCorrelation = np.nanmean(correlations)
            if pearsonCorrelation>0.4:
               print(pearsonCorrelation)
               print(epoch_value)
               print(lr_value)
               print(lambda1_value)



'''layer1 = sc_train[5, :, :, 0]
layer2 = fc_train[5, :, :]

f = open("intra_layer1_day1_sub5_upperT.txt", "w")
for i in range(68):
    for j in range(68):
        if j == i or j > i:
            f.write("\n1 {} 1 {} {}".format(i + 1, j + 1, layer1[i, j]))

f.close()

f = open("intra_layer2_day1_sub5_upperT.txt", "w")
for i in range(68):
    for j in range(68):
        if j == i or j > i:
            #if layer2[i,j]<0:
                #f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, 0))
               # continue
            #else:
                f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, layer2[i, j]))

f.close()

layer1 = sc_train[2, :, :, 0]
layer2 = fc_train[2, :, :]

f = open("intera_layer1_sc_subj2.txt", "w")
for i in range(68):
    for j in range(68):
        if j == i or j > i:
            f.write("\n1 {} 1 {} {}".format(i + 1, j + 1, layer1[i, j]))

f.close()

f = open("intera_layer2_fc_day2_subj2.txt", "w")
for i in range(68):
    for j in range(68):
        if j == i or j > i:
            f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, layer2[i, j]))

f.close()'''

