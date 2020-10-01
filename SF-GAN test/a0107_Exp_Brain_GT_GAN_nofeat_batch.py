# Created by qli10 at 9/16/2019

# Created by qli10 at 9/14/2019

# Created by qli10 at 9/10/2019

# In[]

from sklearn.model_selection import KFold
import itertools
from helper_n0203 import evaluate_n,getNormalizedMatrix_np,getNormalizedMatrix,triu2vec

#,getNormalizedMatrix,triu2vec

import argparse
import os
import scipy.misc
import numpy as np
# from model_classify import graph2graph  # original Xiaojie's code
# from model import graph2graph  #
# from model_classify_qz import graph2graph
from model_qz_fromOriginal_nofeat_batch import graph2graph
import tensorflow as tf
import torch

def hyperparams():
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    parser = argparse.ArgumentParser(description='')
    # parser.add_argument('--epoch', dest='epoch', type=int, default=1000, help='# of epoch')
    parser.add_argument('--epoch', dest='epoch', type=int, default=30, help='# of epoch')
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=130, help='# images in batch')
    # parser.add_argument('--batch_size', dest='batch_size', type=int, default=20, help='# images in batch')
    parser.add_argument('--train_size', dest='train_size', type=int, default=1e8, help='# images used to train')
    parser.add_argument('--ngf', dest='ngf', type=int, default=5, help='# of gen filters in first conv layer')
    parser.add_argument('--ndf', dest='ndf', type=int, default=5, help='# of discri filters in first conv layer')
    parser.add_argument('--input_nc', dest='input_nc', type=int, default=1, help='# of input image channels')
    parser.add_argument('--output_nc', dest='output_nc', type=int, default=1, help='# of output image channels')
    parser.add_argument('--niter', dest='niter', type=int, default=100, help='# of iter at starting learning rate')
    parser.add_argument('--lr_d', dest='lr_d', type=float, default=0.00010, help='initial learning rate for adam')
    parser.add_argument('--lr_g', dest='lr_g', type=float, default=0.00010, help='initial learning rate for adam')
    parser.add_argument('--beta1', dest='beta1', type=float, default=0.5, help='momentum term of adam')
    parser.add_argument('--flip', dest='flip', type=bool, default=True, help='if flip the images for data argumentation')
    parser.add_argument('--save_epoch_freq', dest='save_epoch_freq', type=int, default=50, help='save a model every save_epoch_freq epochs (does not overwrite previously saved models)')
    parser.add_argument('--save_latest_freq', dest='save_latest_freq', type=int, default=5000, help='save the latest model every latest_freq sgd iterations (overwrites the previous latest model)')
    parser.add_argument('--print_freq', dest='print_freq', type=int, default=50, help='print the debug information every print_freq iterations')
    parser.add_argument('--continue_train', dest='continue_train', type=bool, default=False, help='if continue training, load the latest model: 1: true, 0: false')
    parser.add_argument('--serial_batches', dest='serial_batches', type=bool, default=False, help='f 1, takes images in order to make batches, otherwise takes them randomly')
    parser.add_argument('--serial_batch_iter', dest='serial_batch_iter', type=bool, default=True, help='iter into serial image list')
    parser.add_argument('--sample_dir', dest='sample_dir', default='./sample', help='sample are saved here')
    parser.add_argument('--test_dir', dest='test_dir', default='./', help='test sample are saved here')
    parser.add_argument('--L1_lambda', dest='L1_lambda', type=float, default=1e2, help='weight on L1 term in objective')
    parser.add_argument('--L1_lambda2', dest='L1_lambda2', type=float, default=0, help='weight on L1 term in objective for deconv weights')
    #parser.add_argument('--L1_lambda3', dest='L1_lambda3', type=float, default=70, help='weight on L1 term in objective for diff of deconv weights')
    parser.add_argument('--train_dir', dest='train_dir', default='./', help='train sample are saved here')
    parser.add_argument('--image_size', dest='image_size', default=[20, 20], help='input graph size')
    parser.add_argument('--output_size', dest='output_size', default=[20, 20], help='output graph size')
    parser.add_argument('--checkpoint_dir', dest='checkpoint_dir', default='./checkpoint20', help='models are saved here')
    return parser.parse_args()


folder = 'SC_FC_dataset_0905/'
#files = ['SC_RESTINGSTATE_correlationFC', 'SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_correlationFC','SC_EMOTION_partialCorrelationFC_L1', 'SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_correlationFC', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L2', 'SC_LANGUAGE_correlationFC', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_correlationFC', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L2', 'SC_RELATIONAL_correlationFC', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_correlationFC', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_correlationFC', 'SC_WM_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L2']
# files = ['SC_RESTINGSTATE_correlationFC', 'SC_RESTINGSTATE_partialcorrelationFC_L2']
#files = ['SC_EMOTION_partialCorrelationFC_L2']
files = ['SC_RESTINGSTATE_correlationFC','SC_EMOTION_correlationFC','SC_GAMBLING_correlationFC', 'SC_LANGUAGE_correlationFC', 'SC_MOTOR_correlationFC','SC_RELATIONAL_correlationFC', 'SC_SOCIAL_correlationFC',  'SC_WM_correlationFC']


args = hyperparams()

for file in files:
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

    entire_size = allSC.shape[0]

    # validation_size = int(entire_size / 5)
    test_size = 100

    # [train and hypertune using 3-folder cross-validation]
    dataset_SC = allSC[list(range(test_size, entire_size))]
    dataset_FC = allFC[list(range(test_size, entire_size))]

    sc_train = dataset_SC
    fc_train = dataset_FC

    sc_test = allSC[list(range(0, test_size))]
    fc_test = allFC[list(range(0, test_size))]

    args.batch_size = fc_test.shape[0]

    checkpoint_dir = 'brain_checkpoint_' + file
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    tf.reset_default_graph()
    args.image_size = [sc_train[0].shape[0], sc_train[0].shape[0]]
    args.output_size = [fc_train[0].shape[0], fc_train[0].shape[0]]

    args.checkpoint_dir = checkpoint_dir
    ################################################################
    with tf.Session() as sess:
        #print('Training the model for ', folder, file)
        model = graph2graph(sess, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
                            sample_dir=args.sample_dir, test_dir=args.test_dir, train_dir=args.train_dir,
                            image_size=args.image_size, output_size=args.output_size)

        model.train(args, sc_train, fc_train, sc_test, fc_test)

        #####################  Test ########################

        mse, r2, Ra_t = model.test(args, sc_test, fc_test)
        predFC = Ra_t[:, :, :, 0]
        # predFC_train = Ra_t_train[:, :, :, 0]
        s1 = "prediction_nofeat_test_%s" % (file)
        np.save(s1, predFC)
        #np.save("pred_nofeat",Ra_t)
        # predFC = vec2Adjacent(Ra_t,fc_test.shape[1])
        #print('predFC:', predFC.shape)
        print(file)
        print('mse', mse)
        print('r2', r2)

    correlations, diff = evaluate_n(predFC, fc_test)
    # correlations_train, diff_train = evaluate_n(predFC_train, fc_train)
    # predfc, empiricalfc = tonp(predFC, fc_test)
    #print('evaluation_input_file:', file)
    # np.savez_compressed(file, a=predfc, b=empiricalfc)
    pearsonCorrelation = np.nanmean(correlations)
    # pearsonCorrelation_train[cv_idx] = np.nanmean(correlations_train)
    print('{:.2f}'.format(pearsonCorrelation))

    break
outfile = folder + 'result0917_GT_GAN_' + file + '_' + str(args.epoch)
outtxt = outfile + ' ' + '{:.4f}'.format(np.nanmean(pearsonCorrelation)) + ' {:.4f}'.format(np.std(pearsonCorrelation))
#print(outtxt)



'''for file in files:
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
          #rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i]) * 100
          #rawFC[i] = getNormalizedMatrix_np(rawFC_ori[i]) * 100
          rawSC[i] = getNormalizedMatrix_np(rawSC_ori[i])
          rawFC[i] = rawFC_ori[i]

      allSC = np.zeros((rawSC.shape[0], rawSC.shape[1], rawSC.shape[2], 4))
      allSC[:, :, :, 0] = rawSC
      allFC = rawFC

    #   find the best hyperparameters
    #print('training bestnet on validation set')
      entire_size = allSC.shape[0]

      validation_size = int(entire_size / 5)


# [train and evaluate using 5-folder cross-validation]
      dataset_SC = allSC[list(range(validation_size, entire_size))]
      dataset_FC = allFC[list(range(validation_size, entire_size))]

      raw_dataset_SC = rawSC_ori[list(range(validation_size, entire_size))]
      raw_dataset_FC = rawFC_ori[list(range(validation_size, entire_size))]
      
      cvFolers = 5
      kf = KFold(n_splits=cvFolers)
      pearsonCorrelation = np.zeros(cvFolers)
      pearsonCorrelation_train = np.zeros(cvFolers)
      cv_idx = 0


      for train_index, test_index in kf.split(dataset_SC):
        # print("TRAIN:", train_index, "\n TEST:", test_index)
          sc_train, sc_test = dataset_SC[train_index], dataset_SC[test_index]
          fc_train, fc_test = dataset_FC[train_index], dataset_FC[test_index]
        #ff1_new_train, ff1_new_test = ff1_new[0, train_index, :, 0], ff1_new[0, test_index, :, 0]

          raw_fc_train, raw_fc_test = raw_dataset_FC[train_index], raw_dataset_FC[test_index]
          raw_sc_train, raw_sc_test = raw_dataset_SC[train_index], raw_dataset_SC[test_index]

          sc_train_min = np.amin(sc_train)
          sc_test_min = np.amin(sc_test)
          fc_train_min = np.amin(fc_train)
          fc_test_min = np.amin(fc_test)

          sc_train_max = np.amax(sc_train)
          sc_test_max = np.amax(sc_test)
          fc_train_max = np.amax(fc_train)
          fc_test_max = np.amax(fc_test)

          args.batch_size = fc_test.shape[0]

      #features_train = tf.reshape(ff1_new_train,[1,527,3,1])

        # kf_validation = KFold(n_splits=cvFolers)

          checkpoint_dir = 'brain_checkpoint_'+file
          if not os.path.exists(checkpoint_dir):
              os.makedirs(checkpoint_dir)
        #tf.reset_default_graph()
          tf.reset_default_graph()
          args.image_size = [sc_train[0].shape[0], sc_train[0].shape[0]]
          args.output_size = [fc_train[0].shape[0], fc_train[0].shape[0]]

          args.checkpoint_dir = checkpoint_dir
          with tf.Session() as sess:
              print('Training the model for ', folder, file)
            # model = graph2graph(sess, Ds=Ds, No=sc_train.shape[1], Nr=int(sc_train.shape[1]*(sc_train.shape[1]-1)), Dr=Dr, Dx=Dx, De_o=De_o,
            #                     De_r=De_r, Dmap=Dmap, Mini_batch=sc_test.shape[0],
            #                     checkpoint_dir=checkpoint_dir, epoch=epoch)
            # model = graph2graph(sess, batch_size=1,
            #                     checkpoint_dir=checkpoint_dir, sample_dir='./sample', test_dir='./',
            #                     train_dir='./sample', image_size=[sc_train[0].shape[0],sc_train[0].shape[0]], output_size=[sc_train[0].shape[0],sc_train[0].shape[0]])
              model = graph2graph(sess, batch_size=args.batch_size, checkpoint_dir=args.checkpoint_dir,
                                sample_dir=args.sample_dir, test_dir=args.test_dir, train_dir=args.train_dir,
                                image_size=args.image_size, output_size=args.output_size)



              model.train(args, sc_train, fc_train, sc_test, fc_test)

        #####################  Test ########################

              Ra_t = model.test(args, sc_test, fc_test)
              Ra_t_train = model.test(args, sc_train, fc_train)
              predFC = Ra_t[:, :, :, 0]
              #predFC_train = Ra_t_train[:, :, :, 0]
              #np.save("pred_nofeat",Ra_t)
            # predFC = vec2Adjacent(Ra_t,fc_test.shape[1])
              print('predFC:', predFC.shape)

          correlations, diff = evaluate_n(predFC, fc_test)
          #correlations_train, diff_train = evaluate_n(predFC_train, fc_train)
        #predfc, empiricalfc = tonp(predFC, fc_test)
          print('evaluation_input_file:', file)
        #np.savez_compressed(file, a=predfc, b=empiricalfc)
          pearsonCorrelation[cv_idx] = np.nanmean(correlations)
          #pearsonCorrelation_train[cv_idx] = np.nanmean(correlations_train)
          print(file+' '+'{:.2f}'.format(pearsonCorrelation[cv_idx]))
          cv_idx = cv_idx+1

          break
      outfile = folder+'result0917_GT_GAN_' + file+'_'+str(args.epoch)
    # print(outfile,np.mean(pearsonCorrelation))
      outtxt = outfile + ' ' + '{:.4f}'.format(np.nanmean(pearsonCorrelation))+' {:.4f}'.format(np.std(pearsonCorrelation))
      print(outtxt)
    #file1 = open(folder + 'result0917_GT_GAN_'+str(args.epoch)+file+'.txt', "a")
    #file1.write(outtxt + '\n')
    #file1.close()
    #np.save(outfile, pearsonCorrelation)'''
