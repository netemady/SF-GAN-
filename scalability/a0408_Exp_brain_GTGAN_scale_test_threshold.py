# Created by netemady at 03/06/2020

# In[]

from sklearn.model_selection import KFold
import itertools
from helper_n0203 import evaluate_n, getNormalizedMatrix_np, getNormalizedMatrix, triu2vec
import argparse
import os
import scipy.misc
import numpy as np
from model_GTGAN_scale_test_threshold import graph2graph
import tensorflow as tf
import torch
from timeit import default_timer as timer



def hyperparams(lambda2, setting_value):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
    parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
    # parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=131, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--niter', dest='niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.00010, help='initial learning rate for adam')
    parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.00010, help='initial learning rate for adam')
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
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1e2, help='weight on L1 term in objective')
    parser.add_argument('--L1_lambda2', dest='L1_lambda2', type=float, default=lambda2,
                        help='weight on L1 term in obective for deconv weights')
    parser.add_argument('--Setting_value', dest='Setting_value', type=float, default=setting_value,
                        help='setting for regularization')
    parser.add_argument('--train_dir', dest='train_dir', default='./', help='train sample are saved here')
    parser.add_argument('--image_size', dest='image_size', default=[20, 20], help='input graph size')
    parser.add_argument('--output_size', dest='output_size', default=[20, 20], help='output graph size')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint20',
                        help='models are saved here')
    return parser.parse_args()

def remove_edge(m, threshold):
    
   m_final = m
   if threshold>0:
    s1 = m.shape[1]
    fc_new = np.triu(m, 0)
    for i in range(m.shape[1]):
        fc_new[i, i] = 0

    f_flatten = fc_new.flatten()
    sort_index = np.argsort(f_flatten)
    k = int((s1 ** 2 - s1) / 2)
    k = int(k)
    r = int(threshold * (k))
    for i in range(r):
        element = sort_index[k + i + s1] + 1
        remainder = element % s1
        divisor = int(element / s1)
        col = int((element % s1) - 1)
        if divisor == 0:
            row_ = 0
        elif remainder == 0:
            row_ = divisor - 1
        else:
            row_ = divisor

        m_final[row_, col] = 0
        m_final[col, row_] = 0

   return(m_final)


# default=0.0005  , help='initial learning rate for adam

folder = 'SC_FC_dataset_0905/'
# files = ['SC_RESTINGSTATE_correlationFC', 'SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_correlationFC','SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
# files = ['SC_RESTINGSTATE_correlationFC','SC_EMOTION_correlationFC','SC_GAMBLING_correlationFC', 'SC_LANGUAGE_correlationFC', 'SC_MOTOR_correlationFC','SC_RELATIONAL_correlationFC', 'SC_SOCIAL_correlationFC',  'SC_WM_correlationFC']
#files = ['SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_partialCorrelationFC_L2',
         #'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_partialCorrelationFC_L2',
        # 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_partialCorrelationFC_L2']

#file = 'SC_SOCIAL_partialCorrelationFC_L2'
file = 'SC_RESTINGSTATE_correlationFC'

#results_train_time = np.zeros((11, 20))
#results_test_time = np.zeros((11, 20))

results_train_time_mean = np.zeros((9, 3))
results_train_time_std = np.zeros((9, 3))
results_test_time_mean = np.zeros((9, 3))
results_test_time_std = np.zeros((9, 3))

itr_vals_train = np.zeros((1,20))
itr_vals_test = np.zeros((1,20))

#t = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
t = [0]

settings_vals = [1,2]
entire_size_vals = [823, 780, 695, 610, 525, 440, 355, 270, 185]
test_size_vals = [100, 96, 84, 72, 60, 48, 36, 24, 12]

k = 0

for threshold in t:
 for setting_value in settings_vals:
    for kk in range(9):
      for itr in range(20):

        #lambda2 = dataset_lm_sett[file_num, setting_value - 1]
        lambda2 = 1
        #if file_num == 5:
         #   args.L1_lambda = 1e5
        # file_num = file_num+1
        args = hyperparams(lambda2, setting_value)

        feat_new = np.load('feat_new.npy')
        nn = 161
        ff1 = np.reshape(feat_new, (1, 823, 161, 1))
        ff1 = ff1[0,:,0:nn,0]
        ff1_new = np.array(ff1).astype(np.float32)
        ff1_new = np.reshape(ff1_new, (1, 823, nn, 1))

        # for file in files:
        data = np.load(folder + file + '.npz')
        try:

            rawSC_ori = data['sc']
            rawFC_ori = data['fc']
        except:
            print("rfMRI dataset!!")
            rawSC_ori = data['rawSC']
            rawFC_ori = data['rawFC']

        rawSC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
        rawFC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))

        for i in range(rawFC_ori.shape[0]):
            # rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i]) * 100
            # rawFC[i] = getNormalizedMatrix_np(rawFC_ori[i]) * 100
            rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])
            rawFC[i] = rawFC_ori[i]

        allSC = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2], 4))
        allSC[:, :, :, 0] = rawSC
        allFC = rawFC

        #entire_size = allSC.shape[0]
        entire_size = entire_size_vals[kk]

        # validation_size = int(entire_size / 5)
        #test_size = 100
        test_size = test_size_vals[kk]

        # [train and hypertune using 3-folder cross-validation]
        dataset_SC = allSC[list(range(100, entire_size))]
        dataset_FC = allFC[list(range(100, entire_size))]

        sc_train = dataset_SC
        fc_train = dataset_FC
        ff1_new_train = ff1_new[0, list(range(100, entire_size)), :, 0]
        ff1_new_test = ff1_new[0, list(range(0, test_size)), :, 0]

        sc_test = allSC[list(range(0, test_size))]
        fc_test = allFC[list(range(0, test_size))]
        

       # for j in range(sc_train.shape[0]):
        #      sc_train[j,:,:,0] = remove_edge(sc_train[j,:,:,0], threshold)
         #     fc_train[j,:,:] = remove_edge(fc_train[j,:,:], threshold)

        #for j in range(sc_test.shape[0]):
         #     sc_test[j,:, :, 0] = remove_edge(sc_test[j,:,:,0], threshold)
          #    fc_test[j,:,:] = remove_edge(fc_test[j,:,:], threshold)

        #np.save('trueFC_SocialL2', fc_test) used for brainNetviewer
        #np.save('SC_SocialL2', sc_test)

        args.batch_size = fc_test.shape[0]

        checkpoint_dir = 'brain_checkpoint_' + file
        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        tf.reset_default_graph()
        args.image_size = [sc_train[0].shape[0], sc_train[0].shape[0]]
        args.output_size = [fc_train[0].shape[0], fc_train[0].shape[0]]

        args.checkpoint_dir = checkpoint_dir
        with tf.Session() as sess:
            print('Training the model for ', folder, file)

            model = graph2graph(sess, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
                                sample_dir=args.sample_dir, test_dir=args.test_dir, train_dir=args.train_dir,
                                image_size=args.image_size, output_size=args.output_size,
                                L1_lambda2=args.L1_lambda2, L1_lambda=args.L1_lambda, Setting_value=args.Setting_value)

            runtime_train = model.train(args, sc_train, fc_train, ff1_new_train, lambda2, setting_value,nn)
            #results_train_time[k,itr] = runtime_train
            ####################################################

            Ra_t, runtime_test = model.test(args, sc_test, fc_test, ff1_new_test,nn)
            #results_test_time[k,itr] = runtime_test

        itr_vals_train[0, itr] = runtime_train
        itr_vals_test[0, itr] = runtime_test

      results_train_time_mean[kk, setting_value - 1] = np.mean(itr_vals_train)
      results_train_time_std[kk, setting_value - 1] = np.std(itr_vals_train)
      results_test_time_mean[kk, setting_value - 1] = np.mean(itr_vals_test)
      results_test_time_std[kk, setting_value - 1] = np.std(itr_vals_test)
      q = 5

a=5




print('training_mean = ', np.transpose(results_train_time_mean))
print('training_std = ', np.transpose(results_train_time_std))
print('test_mean = ', np.transpose(results_test_time_mean))
print('test_std = ', np.transpose(results_test_time_std))
a = 3
 


        # print(file + ' ' + '{:.2f}'.format(pearsonCorrelation))
        # s = "setting %s, lambda2 %s" % (setting_value, lambda2)
        # print(s)
        # print('mse:', mse)
        # print('r2:', r2)
        # q=1
        # print('mse:', mse)
        # print('r2:', r2)