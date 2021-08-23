# Created by netemady - 03/2021

#gradient generation for GT-GAN with meta-Features

#last access 71021

import numpy as np
from sklearn.model_selection import KFold
import itertools
from helper_n0203 import evaluate_n, getNormalizedMatrix_np, getNormalizedMatrix, triu2vec
import argparse
import os
import scipy.misc
from model_Saliency import graph2graph
import tensorflow as tf
import torch
from matplotlib import pyplot as plt

#file1 = np.load('output20.npy')
#flag = 1

# Created by netemady at 03/06/2020

# In[]

def hyperparams(lambda1, lambda2, setting_value):
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
    parser.add_argument('--epoch', dest='epoch', type=int, default=3, help='# of epoch')
    # parser.add_argument('--batch_size', dest='batch_size', type=int, default=1, help='# images in batch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=131, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--niter', dest='niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.010, help='initial learning rate for adam')
    parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.010, help='initial learning rate for adam')
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
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=lambda1,
                        help='weight on L1 term in objective')
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


# default=0.0005  , help='initial learning rate for adam

# Learning rate for ADAM
# full: 0.0001
# L1: 0.0005
# L2: 0.0003

folder = 'SC_FC_dataset_0905/'
# files = ['SC_RESTINGSTATE_correlationFC', 'SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_correlationFC','SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']

#files = ['SC_RESTINGSTATE_correlationFC']

results_corr = np.zeros((2, 3))

# lambda1_values = [1e5, 1e4, 1e3]
lambda1_values = [1e8]

#l1 = range(0, 1001, 100)
#l2 = [1e4]

# lambda2_values = np.concatenate((l1,l2,), axis=0)
lambda2_values = [1e2]

settings_vals = [3]

files = ['SC_GAMBLING_correlationFC']

data = np.load(folder + 'SC_GAMBLING_correlationFC' + '.npz')
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


for file in files:

    for setting_value in settings_vals:
        # plt.figure()

        for lambda1 in lambda1_values:

            for lambda2 in lambda2_values:


                args = hyperparams(lambda1, lambda2, setting_value)

                feat_new = np.load('feat_new.npy')
                #file1 = np.load('output20.npy')
                nn = 161
                ff1 = np.reshape(feat_new[0:818,:], (1, 818, 161, 1))
                ff1_new = np.array(ff1).astype(np.float32)

                '''data = np.load(folder + file + '.npz')
                try:

                    rawSC_ori = data['sc']
                    rawFC_ori = data['fc']
                except:
                    print("rfMRI dataset!!")
                    rawSC_ori = data['rawSC']
                    rawFC_ori = data['rawFC']'''

                rawSC_ori = SC_818
                rawFC_ori = final_fc_day1

                rawSC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))
                rawFC = np.zeros((rawSC_ori.shape[0], rawSC_ori.shape[1], rawSC_ori.shape[2]))

                for i in range(rawFC_ori.shape[0]):
                    rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])
                    rawFC[i] = rawFC_ori[i]

                allSC = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2], 4))
                allSC[:, :, :, 0] = rawSC
                allFC = rawFC

                fc1 = allFC[0, :, :]

                entire_size = allSC.shape[0]

                # validation_size = int(entire_size / 5)
                test_size = 100

                # [train and hypertune using 3-folder cross-validation]
                dataset_SC = allSC[list(range(test_size, entire_size))]
                dataset_FC = allFC[list(range(test_size, entire_size))]

                sc_train = dataset_SC
                fc_train = dataset_FC
                ff1_new_train = ff1_new[0, list(range(test_size, entire_size)), :, 0]
                ff1_new_test = ff1_new[0, list(range(0, test_size)), :, 0]

                sc_test = allSC[list(range(0, test_size))]
                fc_test = allFC[list(range(0, test_size))]

                '''layer1 = sc_train[0, :, :, 0]
                layer2 = fc_train[0, :, :]

                f = open("intera_layer1_sc.txt", "w")
                for i in range(68):
                    for j in range(68):
                        if j == i or j > i:
                            f.write("\n1 {} 1 {} {}".format(i + 1, j + 1, layer1[i, j]))

                f.close()

                f = open("intera_layer2_fc.txt", "w")
                for i in range(68):
                    for j in range(68):
                        if j == i or j > i:
                            f.write("\n2 {} 2 {} {}".format(i + 1, j + 1, layer2[i, j]))

                f.close()'''


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
                                        L1_lambda2=args.L1_lambda2, L1_lambda=args.L1_lambda,
                                        Setting_value=args.Setting_value)

                    model.train(args, sc_train, fc_train, sc_test, fc_test, ff1_new_train, ff1_new_test,
                                           lambda2,
                                           setting_value)

                    #grad_new = a[2,:,:]

                    #v_list = a[1, :, :]
                    #np.save('n4_tensor_saliency.npy', a)

                    '''v_s0s0_ch0 = v_list[0, :, :, 0]
                    v_s0s0_ch1 = v_list[0, :, :, 1]
                    v_s0s0_ch2 = v_list[0, :, :, 2]
                    v_s0s0_ch3 = v_list[0, :, :, 3]

                    v_s0s1_ch0 = v_list[1, :, :, 0]
                    v_s0s1_ch1 = v_list[1, :, :, 1]
                    v_s0s1_ch2 = v_list[1, :, :, 2]
                    v_s0s1_ch3 = v_list[1, :, :, 3]'''

                    ####################################################

                    # access and use self.real_A, self.fake_B: model.real_A, model.fake_B ?

                    # var = tf.Variable(model.real_A[0, :, :, 0])  # Must be a tf.float32 or tf.float64 variable.

                    # var_grad = tf.gradients(model.fake_B[0,5,6,0], [var])

                    # E = var_grad.eval({var: sc_train[0, :, :, 0]})

                    mse, r2, Ra_t = model.test(args, sc_test, fc_test, ff1_new_test)
                    predFC = Ra_t[:, :, :, 0]
                    # s1 = "prediction_test_%s_set%s_lm%s" % (file, setting_value, lambda2)
                    # np.save(s1, Ra_t)

                '''v_s0s0_ch0[v_s0s0_ch0 < 0] = 0
                v_s0s0_ch1[v_s0s0_ch1 < 0] = 0
                v_s0s0_ch2[v_s0s0_ch2 < 0] = 0
                v_s0s0_ch3[v_s0s0_ch3 < 0] = 0

                v_s0s1_ch0[v_s0s1_ch0 < 0] = 0
                v_s0s1_ch1[v_s0s1_ch1 < 0] = 0
                v_s0s1_ch2[v_s0s1_ch2 < 0] = 0
                v_s0s1_ch3[v_s0s1_ch3 < 0] = 0

                plt.figure(figsize=(10, 4))
                plt.subplot(131)
                plt.xlabel('Brain ROI index')
                plt.ylabel('Brain ROI index')
                plt.title('SC00')
                plt.imshow(v_s0s0_ch0, cmap='hot', interpolation='nearest')
                # plt.clim(-0.1, 0.3)
                plt.colorbar()

                plt.subplot(132)
                plt.imshow(v_s0s0_ch1, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC01')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.subplot(133)
                plt.imshow(v_s0s0_ch2, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC02')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.subplot(134)
                plt.imshow(v_s0s0_ch3, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC03')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.savefig('case020221_grad_GTGAN_RestL2_s0s0' + '.pdf')
                plt.show()
                a = 1

                plt.figure(figsize=(10, 4))
                plt.subplot(131)
                plt.xlabel('Brain ROI index')
                plt.ylabel('Brain ROI index')
                plt.title('SC10')
                plt.imshow(v_s0s1_ch0, cmap='hot', interpolation='nearest')
                # plt.clim(-0.1, 0.3)
                plt.colorbar()

                plt.subplot(132)
                plt.imshow(v_s0s1_ch1, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC11')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.subplot(133)
                plt.imshow(v_s0s1_ch2, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC12')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.subplot(134)
                plt.imshow(v_s0s1_ch3, cmap='hot', interpolation='nearest')
                plt.xlabel('Brain ROI index')
                # plt.ylabel('Brain ROI index')
                plt.title('SC13')
                # plt.clim(-0.2, 0.2)
                plt.colorbar()

                plt.savefig('case020221_grad_GTGAN_RestL2_s0s1' + '.pdf')
                plt.show()'''

                correlations, diff = evaluate_n(predFC, fc_test)

                pearsonCorrelation = np.nanmean(correlations)
                # results_corr[lm_num, setting_value - 1] = '{:.2f}'.format(pearsonCorrelation)
                lm_num = lm_num + 1
                some_v = 0

