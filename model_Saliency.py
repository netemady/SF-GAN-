# -*- coding: utf-8 -*-
"""
Created on Tue Jun 26 09:19:08 2018

@author: gxjco
"""
from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
from six.moves import xrange
import csv
from ops_ import *
from utils_ import *
from sklearn.metrics import mean_squared_error
from math import sqrt
from scipy.spatial.distance import pdist
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
from helper_n0203 import evaluate_n, getNormalizedMatrix_np, getNormalizedMatrix, triu2vec
import scipy.io as sio


def load_data_diag_qz(x, y):
    size = x.shape[1]
    a = np.ones((size, size)) - np.eye(size)
    # x = np.load('real/data_x_' + str(size) + '.npy')
    # y = np.load('real/data_y_' + str(size) + '.npy')
    data = np.zeros((x.shape[0], x.shape[1], x.shape[2], 5))
    print('f=load_data')
    # print('this is size of sc')
    # print(x.shape[0], x.shape[1], x.shape[2])
    for i in range(x.shape[0]):
        data[i, :, :, 0:4] = x[i, :, :, :]
        data[i, :, :, 4] = y[i, :, :]
    node = np.zeros((x.shape[0], x.shape[1]))
    for i in range(x.shape[0]):
        node[i] = sum(data[i, :, :, 0] * np.eye(size))
    for i in range(x.shape[0]):
        data[i, :, :, 0] = data[i, :, :, 0] * a  # diagnal is removed
        data[i, :, :, 4] = data[i, :, :, 4] * a
    node = node.reshape(x.shape[0], size, 1, 1)
    # print(size(node))
    # feat = np.ones((x.shape[0],1,1,1))
    # node = np.concatenate((node,feat),axis=1)
    return data, node
    # if type_ == 'train':  return data[:200], node[:200]
    # if type_ == 'test':   return data[200:343], node[200:343]


class graph2graph(object):
    def __init__(self, sess, test_dir, train_dir, image_size, output_size,
                 batch_size, L1_lambda2, L1_lambda, Setting_value, sample_size=1,
                 gf_dim=5, df_dim=5,
                 input_c_dim=4, output_c_dim=1,
                 checkpoint_dir=None, sample_dir=None, g_train_num=6, d_train_num=3):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            output_size: (optional) The resolution in pixels of the images. [256]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            input_c_dim: (optional) Dimension of input image color. For grayscale input, set to 1. [3]
            output_c_dim: (optional) Dimension of output image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.is_grayscale = (input_c_dim == 1)
        self.batch_size = batch_size
        self.image_size = image_size
        self.sample_size = sample_size
        self.output_size = output_size
        self.g_train_num = g_train_num
        self.d_train_num = d_train_num

        self.gf_dim = gf_dim
        self.df_dim = df_dim

        self.input_c_dim = input_c_dim
        self.output_c_dim = output_c_dim

        self.L1_lambda = L1_lambda
        self.L1_lambda2 = L1_lambda2

        self.Setting_value = Setting_value

        # batch normalization : deals with poor initialization helps gradient flow
        self.d_bn1 = batch_norm(name='d_bn1')
        self.d_bn2 = batch_norm(name='d_bn2')
        self.d_bn3 = batch_norm(name='d_bn3')
        self.d_bn4 = batch_norm(name='d_bn4')
        # d.bn4/g_bn_e4/g_bn_d1 : unused

        self.g_bn_e1 = batch_norm(name='g_bn_e1')
        self.g_bn_e2 = batch_norm(name='g_bn_e2')
        self.g_bn_e3 = batch_norm(name='g_bn_e3')
        self.g_bn_e4 = batch_norm(name='g_bn_e4')

        self.g_bn_d1 = batch_norm(name='g_bn_d1')
        self.g_bn_d2 = batch_norm(name='g_bn_d2')
        self.g_bn_d3 = batch_norm(name='g_bn_d3')

        self.checkpoint_dir = checkpoint_dir
        self.build_model()
        print('flag=g-init')

    def build_model(self):
        nn = 161
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size[0], self.image_size[1],
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        # self.node = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], 1, 1])
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], 1, 1])
        self.features = tf.placeholder(tf.float32, [self.batch_size, 1, nn, 1])
        self.row = tf.placeholder(tf.int32, shape = ())
        self.column = tf.placeholder(tf.int32, shape = ())
        # self.d2_w_diff = tf.zeros([1, 1, 1, 163], dtype=tf.dtypes.float32)

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]


        print('flag=build model,before gen')
        self.grad, self.fake_B, self.d3, self.d2_w = self.generator(self.real_A, self.node, self.features, self.row, self.column)
        # self.fake_B, self.d3 = self.generator(self.real_A, self.node)
        print('flag=build model,after gen')

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, self.node, reuse=False)  # define the input from image
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, self.node, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        if self.Setting_value == 3:

            # Setting3
            '''layer3 = 0
            for i in xrange(10):
                layer2 = tf.zeros((1, 1))
                for j in xrange(nn):
                    # print(j)
                    layer1 = self.d2_w[0, :, i, 11:]
                    norm_1 = tf.norm(layer1[:, j], ord=1)
                    norm_1 = tf.reshape(norm_1, [1, 1])
                    layer2 = tf.concat((layer2, norm_1), axis=0)
                layer3 = layer3 + tf.norm(layer2, ord=2)

            loss_term = layer3
        else:
            loss_term = tf.reduce_mean(tf.abs(self.d2_w[:, :, :, 11:]))'''

            layer3 = 0
            for i in xrange(10):
                layer1 = self.d2_w[0, :, i, 11:]
                norm_1_set3 = tf.norm(layer1, ord=1, axis=0)
                norm_1_set3 = tf.reshape(norm_1_set3, [1, nn])
                layer3 = layer3 + tf.norm(norm_1_set3, ord=2)

            # loss_term = tf.reduce_mean(tf.abs(layer3))
            loss_term = layer3
        else:
            loss_term = tf.reduce_mean(tf.abs(self.d2_w[:, :, :, 11:]))

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_AB - self.fake_AB)) \
                      + self.L1_lambda2 * loss_term

        # + self.L1_lambda2 * layer3
        # + self.L1_lambda2 * tf.reduce_mean(tf.abs(self.d2_w[:,:,:,11:]))

        self.d_loss_real_sum = tf.summary.scalar("d_loss_real", self.d_loss_real)
        self.d_loss_fake_sum = tf.summary.scalar("d_loss_fake", self.d_loss_fake)

        self.d_loss = self.d_loss_real + self.d_loss_fake

        self.g_loss_sum = tf.summary.scalar("g_loss", self.g_loss)
        self.d_loss_sum = tf.summary.scalar("d_loss", self.d_loss)

        t_vars = tf.trainable_variables()

        self.d_vars = [var for var in t_vars if 'd_' in var.name]
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()
        print('flag=end of build model')

    def train(self, args, sc, fc, sc_test, fc_test, ff1_new_train, ff1_new_test, lambda2, setting_value):
        nn = 161

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

        """Train pix2pix"""

        d_optim = tf.train.AdamOptimizer(args.lr_d, beta1=args.beta1) \
            .minimize(self.d_loss, var_list=self.d_vars)

        g_optim = tf.train.AdamOptimizer(args.lr_g, beta1=args.beta1) \
            .minimize(self.g_loss, var_list=self.g_vars)

        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        self.g_sum = tf.summary.merge([self.d__sum,
                                       self.fake_B_sum, self.d_loss_fake_sum, self.g_loss_sum])
        self.d_sum = tf.summary.merge([self.d_sum, self.d_loss_real_sum, self.d_loss_sum])
        self.writer = tf.summary.FileWriter("./logs", self.sess.graph)

        counter = 1
        start_time = time.time()

        data, node = load_data_diag_qz(sc, fc, )
        data0 = data[0,:,:,0]
        data1 = data[0, :, :, 1]
        data2 = data[0, :, :, 2]
        data3 = data[0, :, :, 3]
        data4 = data[0, :, :, 4]
        # print('flag=train,after load')

        errD_fake = 0
        errD_real = 0
        errG = 0
        # best = 5

        pearsonCorrelation = np.zeros(args.epoch)
        pearsonCorrelation_train = np.zeros(args.epoch)

        v1_final = np.zeros((args.epoch, 161))

        for epoch in xrange(args.epoch):

            print('############## epoch =', epoch)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            loss_train = 0
            for idx in xrange(0, batch_idxs):
                batch = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_ = batch[0,:,:,0]
                batch_images = np.array(batch).astype(np.float32)

                '''a = np.ones((batch_images.shape[0], self.image_size[0], self.image_size[0], 1))
                for i in range(batch_images.shape[0]):
                    a[i, :, :, 0] = np.triu(a[i, :, :, 0]) - np.eye(self.image_size[0])
                # image is (n*300 x 300 x 1)
                batch_images_new = batch_images * a

                batch_images_new_0 = batch_images_new[0, :, :, 0]
                batch_images_new_1 = batch_images_new[0, :, :, 1]
                batch_images_new_2 = batch_images_new[0, :, :, 2]
                batch_images_new_3 = batch_images_new[0, :, :, 3]
                batch_images_new_4 = batch_images_new[0, :, :, 4]'''


                batch_images_values = batch_images[0,:,:,0]
                # print(node)
                node_ = node[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_nodes = np.array(node_).astype(np.float32)
                feat_train = ff1_new_train[idx * self.batch_size: (idx + 1) * self.batch_size]
                feat_train2 = np.array(feat_train).astype(np.float32)
                # feat_train3 = np.reshape(feat_train2, [1, 1, nn, 1])
                feat_train3 = np.reshape(feat_train2, [self.batch_size, 1, nn, 1])
                print('idx,errD_fake + errD_real', idx, errD_fake + errD_real)

                if errD_fake + errD_real > 0.5:
                    for i in range(self.d_train_num):
                        # Update D network
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                       feed_dict={self.real_data: batch_images,
                                                                  self.node: batch_nodes,
                                                                  self.features: feat_train3})

                        self.writer.add_summary(summary_str, counter)

                for i in range(self.g_train_num):
                    # Update G network
                    # print(tf.shape(self.node))
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.real_data: batch_images,
                                                              self.node: batch_nodes,
                                                              self.features: feat_train3})

                    self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)

                if setting_value == 1:

                    vars = tf.trainable_variables()
                    vals = self.sess.run(vars)
                    value_ = vals[16]
                    v1 = value_[0, :, 0, 11:]

                    t = value_[0, :, 0, 11:]
                    t_mean = np.mean(t, axis=0)
                    t_mean = np.reshape(t_mean, [1, nn])
                    t_mean2 = np.concatenate((t_mean, t_mean), axis=0)
                    for i in xrange(66):
                        t_mean2 = np.concatenate((t_mean2, t_mean), axis=0)

                    t1 = np.reshape(t_mean2, [1, 68, 1, nn])

                    for k in xrange(1, 10):
                        t = value_[0, :, k, 11:]
                        t_mean = np.mean(t, axis=0)
                        t_mean = np.reshape(t_mean, [1, nn])
                        t_mean2 = np.concatenate((t_mean, t_mean), axis=0)
                        for i in xrange(66):
                            t_mean2 = np.concatenate((t_mean2, t_mean), axis=0)

                        t2 = np.reshape(t_mean2, [1, 68, 1, nn])
                        t1 = np.concatenate((t1, t2), axis=2)

                    t0 = value_[0, :, :, 0:11]
                    t0 = np.reshape(t0, [1, 68, 10, 11])
                    vals[16] = np.concatenate((t0, t1), axis=3)
                    variable_0 = vals[16]
                    variable_1 = variable_0[0, :, 0, :]

                samples_train = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: batch_images, self.node: batch_nodes, self.features: feat_train3})

                errD_fake = self.d_loss_fake.eval(
                    {self.real_data: batch_images, self.node: batch_nodes, self.features: feat_train3})
                errD_real = self.d_loss_real.eval(
                    {self.real_data: batch_images, self.node: batch_nodes, self.features: feat_train3})
                errG = self.g_loss.eval(
                    {self.real_data: batch_images, self.node: batch_nodes, self.features: feat_train3})

                loss_train = loss_train + errD_fake + errD_real + errG

                if idx == 0:
                    gen_data_train = samples_train

                if idx > 0:
                    gen_data_train = np.concatenate((gen_data_train, samples_train), axis=0)

            predFC = gen_data_train[:, :, :, 0]
            correlations, diff = evaluate_n(predFC, fc)
            # pearsonCorrelation_train[corr_idx] = np.nanmean(correlations)
            # corr_idx += 1



            counter += 1

            self.save(args.checkpoint_dir, counter)

            ####### Compute test loss
            '''sample_images_all, sample_nodes_all = load_data_diag_qz(sc_test, fc_test)

            batch_idxs_test = len(sample_images_all) // self.batch_size

            sample_images = [sample_images_all[i:i + self.batch_size]
                             for i in xrange(0, len(sample_images_all), self.batch_size)]
            sample_images = np.array(sample_images)

            sample_nodes = [sample_nodes_all[i:i + self.batch_size]
                            for i in xrange(0, len(sample_nodes_all), self.batch_size)]
            sample_nodes = np.array(sample_nodes)

            feat_test = [ff1_new_test[i:i + self.batch_size]
                         for i in xrange(0, len(ff1_new_test), self.batch_size)]
            feat_test2 = np.array(feat_test)

            if self.load(self.checkpoint_dir):
                print(" [*] Load SUCCESS")
            else:
                print(" [!] Load failed...")

            loss = 0
            for i, sample_image in enumerate(sample_images):
                idx = i + 1

                if i < batch_idxs_test:
                    feat_test3 = np.reshape(feat_test2[i], [self.batch_size, 1, nn, 1])

                    samples = self.sess.run(
                        self.fake_B,
                        feed_dict={self.real_data: sample_images[i], self.node: sample_nodes[i],
                                   self.features: feat_test3})

                    errD_fake_test = self.d_loss_fake.eval(
                        {self.real_data: sample_images[i], self.node: sample_nodes[i], self.features: feat_test3})
                    errD_real_test = self.d_loss_real.eval(
                        {self.real_data: sample_images[i], self.node: sample_nodes[i], self.features: feat_test3})
                    errG_test = self.g_loss.eval(
                        {self.real_data: sample_images[i], self.node: sample_nodes[i], self.features: feat_test3})

                    loss = loss + errD_fake_test + errD_real_test + errG_test

                if i == 0:
                    gen_data = samples

                if i > 0:
                    if i < batch_idxs_test:
                        gen_data = np.concatenate((gen_data, samples), axis=0)

            predFC = gen_data[:, :, :, 0]
            correlations, diff = evaluate_n(predFC, fc_test)

            if epoch == 0:
                total_loss_test = [loss / (batch_idxs_test * self.batch_size)]
                pearsonCorrelation = [np.nanmean(correlations)]
                r2_score_value_test = [compute_r2(predFC, fc_test)]
                mse_value_test = [compute_mse(predFC, fc_test)]

            else:
                total_loss_test.append(loss / (batch_idxs_test * self.batch_size))
                pearsonCorrelation.append(np.nanmean(correlations))
                r2_score_value_test.append(compute_r2(predFC, fc_test))
                mse_value_test.append(compute_mse(predFC, fc_test))



            # s = "finalWeights_set%s_lm%s" % (setting_value, lambda2)
            vars = tf.trainable_variables()
            vals = self.sess.run(vars)
            value_ = vals[16]
            # v1 = value_[0, :, 0, 11:]
            v1 = value_[0, :, :, 11:]

            v1_abs = np.absolute(v1)
            v1_mean = np.mean(v1_abs, axis=1)
            v1_mean = np.reshape(v1_mean, (68, 161))
            v1_mean = np.mean(v1_mean, axis=0)

            v1_final[epoch, :] = v1_mean

            if epoch % 200 == 0 or epoch == args.epoch - 1:

                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, batch_idxs, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                self.save(args.checkpoint_dir, counter)'''

        #r_value = 5
        #c_value = 7
        '''a = np.zeros((68**2, 68, 68))
        k = 0
        for i in range(2):
            r_value = i
            for j in range(4):
                  c_value = j
                  var_grad_val = self.sess.run(self.grad, feed_dict={self.real_A: batch_images[:,:,:,0:4], self.node: batch_nodes, self.features: feat_train3, self.row: r_value, self.column: c_value})
                  a1 = var_grad_val[0]
                  a[k,:,:] = a1[0, :, :, 0]
                  k = k+1
        return a'''

    def discriminator(self, image2, node, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is n* 20 x 20 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = lrelu(e2e(image2, self.df_dim, self.image_size[0], self.image_size[0], name='d_h0_conv'))
            # h0 is (n*300 x 300 x d)
            h1 = lrelu(self.d_bn1(e2e(h0, self.df_dim * 2, self.image_size[0], self.image_size[0], name='d_h1_conv')))
            # h1 is (n*300 x 300 x d)
            h2 = lrelu(self.d_bn2(e2n(h1, self.df_dim * 2, self.image_size[0], self.image_size[0], name='d_h2_conv')))
            # h2 is (n*300x 1 x d)
            h2_ = tf.concat((h2, node), axis=3)
            h3 = lrelu(self.d_bn3(n2g(h2_, self.df_dim * 4, self.image_size[0], self.image_size[0], name='d_h3_conv')))
            # h3 is (n*1x1xd)
            # h3_=tf.concat((tf.reshape(h3, [self.batch_size, -1]),image[0:self.batch_size,0,0,1:4]),axis=1)
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 50, 'd_h4_lin')
            h5 = linear(h4, 1, 'd_h5_lin')
            # h4 is (n*d)
            return tf.nn.sigmoid(h5), h5

    def generator(self, image1, node, features, row, column, y=None):
        nn = 161
        #image_input = image

        #X1 = tf.Variable(image_input[0, :, :, 0])
        with tf.variable_scope("generator") as scope:
            #image = tf.Variable(image1)
            image = image1
            a = np.ones((image.shape[0], self.image_size[0], self.image_size[0], 1))
            for i in range(image.shape[0]):
                a[i, :, :, 0] = np.triu(a[i, :, :, 0]) - np.eye(self.image_size[0])
            # image is (n*300 x 300 x 1)
            image = image * a
            e1 = self.g_bn_e1(e2e(lrelu(image), self.gf_dim, self.image_size[0], self.image_size[0], name='g_e1_conv'))
            # e1 is (n*300 x 300*d )
            e2 = self.g_bn_e2(e2e(lrelu(e1), self.gf_dim * 2, self.image_size[0], self.image_size[0], name='g_e2_conv'))
            e2_ = tf.nn.dropout(e2, 0.5)
            # e2 is (n*300 x 300*d )
            print('flag=gen,before e3')
            e3 = self.g_bn_e3(
                e2n(lrelu(e2_), self.gf_dim * 2, self.image_size[0], self.image_size[0], name='g_e3_conv'))
            print('flag=gen,after e3')
            # print(tf.shape(e3))
            # e3 is (n*300 x 1*d )
            # e4 = self.g_bn_e4(n2g(lrelu(e3), self.gf_dim*2, name='g_e4_conv'))
            # e4 is (n*1 x 1*d )
            e3_ = tf.concat((e3, node), axis=3)

            '''
            '''
            b0 = features[0, :, :, :]
            b0 = tf.reshape(b0, [1, 1, nn, 1])
            b1 = tf.concat([b0, b0], axis=1)
            c0 = features[0, :, :, :]
            c0 = tf.reshape(c0, [1, 1, nn, 1])
            for k in range(66):
                b1 = tf.concat([b1, c0], axis=1)

            b1 = tf.reshape(b1, [1, 68, 1, nn])
            # b1 = tf.reshape(b1, [self.batch_size, 68, 1, nn])
            b2 = tf.concat([b1, b1], axis=0)
            for k in range(self.batch_size - 2):
                b2 = tf.concat([b2, b1], axis=0)
            e3_new = tf.concat((e3_, b2), axis=3)

            self.d2, self.d2_w, self.d2_b = de_e2n(tf.nn.relu(e3_new),
                                                   [self.batch_size, self.image_size[0], self.image_size[0],
                                                    self.gf_dim * 2], self.image_size[0], self.image_size[0],
                                                   name='g_d2', with_w=True)
            w_var_flat = self.d2_w[0, :, 0, :]
            d2 = tf.nn.dropout(self.g_bn_d2(self.d2), 0.5)
            d2_ = tf.concat([d2, e2], 3)
            # d2 is (20 x 20 )
            self.d3, self.d3_w, self.d3_b = de_e2e(tf.nn.relu(d2_),
                                                   [self.batch_size, self.image_size[0], self.image_size[0],
                                                    int(self.gf_dim)], self.image_size[0], self.image_size[0],
                                                   name='g_d3', with_w=True)
            d3 = self.g_bn_d3(self.d3)
            d3_ = tf.concat([d3, e1], 3)
            # d3 is (20 x 20 )

            self.d4, self.d4_w, self.d4_b = de_e2e(tf.nn.relu(d3_),
                                                   [self.batch_size, self.image_size[0], self.image_size[0],
                                                    self.output_c_dim], self.image_size[0], self.image_size[0],
                                                   name='g_d4', with_w=True)
            # out = tf.nn.relu(self.d4) * a
            out = (tf.nn.tanh(self.d4)) * a
            out_list = []
            for i in range(out.shape[0]):
                out_list.append(tf.subtract(tf.add(out[i, :, :, 0], tf.transpose(out[i, :, :, 0])),
                                            tf.diag(tf.diag_part(out[i, :, :, 0]))))

            out_final  = tf.reshape(tf.stack(out_list), [len(out_list), self.image_size[0], self.image_size[0], 1])

            #var = tf.Variable(image_input[0, :, :, 0])  # Must be a tf.float32 or tf.float64 variable.

            #var_grad = tf.zeros([68,68], dtype = tf.dtypes.float32)

            var_grad = tf.gradients(out_final[0, row, column, 0], image)



            '''X = tf.Variable(tf.random_normal([3, 3]))
            X_sum = tf.reduce_sum(X)
            y = X_sum ** 2 + X_sum - 1

            #X1 = tf.Variable(image_input[0,:,:,0])

            #grad = tf.gradients(y, X)'''

            return var_grad, tf.reshape(tf.stack(out_list),
                              [len(out_list), self.image_size[0], self.image_size[0], 1]), self.d3, self.d2_w

    def save(self, checkpoint_dir, step):
        model_name = "g2g.model"
        # model_dir = "%s" % ('flu')
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoint...")

        # model_dir = "%s" % ('flu')
        # checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False

            np.save('gen20.npy', gen_data[:, :, :, 0])
            np.save('input20.npy', sample_images_all[:, :, :, 0])
            np.save('output20.npy', sample_images_all[:, :, :, 4])
            return gen_data

    def test(self, args, sc, fc, ff1_new_test):
        nn = 161

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

        gen_data = []
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load testing input
        print("Loading testing images ...")

        sample_images_all, sample_nodes_all = load_data_diag_qz(sc, fc)
        batch_idxs_test = len(sample_images_all) // self.batch_size

        sample_images = [sample_images_all[i:i + self.batch_size]
                         for i in xrange(0, len(sample_images_all), self.batch_size)]
        sample_images = np.array(sample_images)

        sample_nodes = [sample_nodes_all[i:i + self.batch_size]
                        for i in xrange(0, len(sample_nodes_all), self.batch_size)]
        sample_nodes = np.array(sample_nodes)

        feat_test = [ff1_new_test[i:i + self.batch_size]
                     for i in xrange(0, len(ff1_new_test), self.batch_size)]
        feat_test2 = np.array(feat_test)
        # feat_test3 = np.reshape(feat_test2, [self.batch_size, 1, nn, 1])

        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        for i, sample_image in enumerate(sample_images):
            idx = i + 1

            feat_size = feat_test2.shape[0]

            if i < batch_idxs_test:
                feat_test3 = np.reshape(feat_test2[i], [self.batch_size, 1, nn, 1])

                samples = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: sample_images[i], self.node: sample_nodes[i], self.features: feat_test3}
                )

            if i == 0:
                gen_data = samples
                d3 = self.sess.run(
                    self.d3,
                    feed_dict={self.real_data: sample_images[i], self.node: sample_nodes[i], self.features: feat_test3}
                )

            if i > 0:
                if i < batch_idxs_test:
                    gen_data = np.concatenate((gen_data, samples), axis=0)

        '''for i in range(gen_data.shape[0]):
            for m in range(20):
                for n in range(20):
                    if gen_data[i, m, n, 0] < 1: gen_data[i, m, n, 0] = 0'''

        np.save('gen20.npy', gen_data[:, :, :, 0])
        np.save('input20.npy', sample_images_all[:, :, :, 0])
        np.save('output20.npy', sample_images_all[:, :, :, 4])

        mse = compute_mse(gen_data[:, :, :, 0], fc)
        r2 = compute_r2(gen_data[:, :, :, 0], fc)
        return mse, r2, gen_data