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
from helper import evaluate

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
                 batch_size, sample_size=1,
                 gf_dim=5, df_dim=5, L1_lambda=100,
                 L1_lambda2=(1e6),
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
        # self.L1_lambda3 = L1_lambda3

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
        nn = 163
        self.real_data = tf.placeholder(tf.float32,
                                        [self.batch_size, self.image_size[0], self.image_size[1],
                                         self.input_c_dim + self.output_c_dim],
                                        name='real_A_and_B_images')
        # self.node = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], 1, 1])
        self.node = tf.placeholder(tf.float32, [self.batch_size, self.image_size[0], 1, 1])
        # self.features = tf.placeholder(tf.float32, [self.batch_size, 1, nn, 1])
        # self.d2_w_diff = tf.zeros([1, 1, 1, 163], dtype=tf.dtypes.float32)

        self.real_A = self.real_data[:, :, :, :self.input_c_dim]
        self.real_B = self.real_data[:, :, :, self.input_c_dim:self.input_c_dim + self.output_c_dim]

        print('flag=build model,before gen')
        self.fake_B, self.d3, self.d2_w = self.generator(self.real_A, self.node)
        # self.fake_B, self.d3 = self.generator(self.real_A, self.node)
        print('flag=build model,after gen')

        self.real_AB = tf.concat([self.real_A, self.real_B], 3)
        self.fake_AB = tf.concat([self.real_A, self.fake_B], 3)
        self.D, self.D_logits = self.discriminator(self.real_AB, self.node, reuse=False)  # define the input from image
        self.D_, self.D_logits_ = self.discriminator(self.fake_AB, self.node, reuse=True)

        self.d_sum = tf.summary.histogram("d", self.D)
        self.d__sum = tf.summary.histogram("d_", self.D_)
        self.fake_B_sum = tf.summary.image("fake_B", self.fake_B)

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_))) \
                      + self.L1_lambda * tf.reduce_mean(tf.abs(self.real_AB - self.fake_AB)) \
            # + self.L1_lambda2 * tf.reduce_mean(tf.abs(self.d2_w[:,:,:,11:]))

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

    def train(self, args, sc, fc, sc_test, fc_test):
        nn = 163
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
        data_test, node_test = load_data_diag_qz(sc_test, fc_test, )
        print('flag=train,after load')

        errD_fake = 0
        errD_real = 0
        errG = 0
        best = 5

        for epoch in xrange(args.epoch):

            print('############## epoch =', epoch)
            batch_idxs = min(len(data), args.train_size) // self.batch_size
            loss_train = 0
            for idx in xrange(0, batch_idxs):
                batch = data[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_images = np.array(batch).astype(np.float32)
                # print(node)
                node_ = node[idx * self.batch_size:(idx + 1) * self.batch_size]
                batch_nodes = np.array(node_).astype(np.float32)
                print('idx,errD_fake + errD_real', idx, errD_fake + errD_real)

                if errD_fake + errD_real > 0.5:
                    for i in range(self.d_train_num):
                        # Update D network
                        _, summary_str = self.sess.run([d_optim, self.d_sum],
                                                       feed_dict={self.real_data: batch_images,
                                                                  self.node: batch_nodes, })

                        self.writer.add_summary(summary_str, counter)
                # print(tf.size(self.node))

                for i in range(self.g_train_num):
                    # Update G network
                    # print(tf.shape(self.node))
                    _, summary_str = self.sess.run([g_optim, self.g_sum],
                                                   feed_dict={self.real_data: batch_images,
                                                              self.node: batch_nodes})

                    self.writer.add_summary(summary_str, counter)

                # Run g_optim twice to make sure that d_loss does not go to zero (different from paper)

                errD_fake = self.d_loss_fake.eval({self.real_data: batch_images, self.node: batch_nodes})
                errD_real = self.d_loss_real.eval({self.real_data: batch_images, self.node: batch_nodes})
                errG = self.g_loss.eval({self.real_data: batch_images, self.node: batch_nodes})

                loss_train = loss_train + errD_fake + errD_real + errG
                test_batch_nodes = np.array(node_).astype(np.float32)
                # print('idx,errD_fake + errD_real', idx, errD_fake + errD_real)
                predict_train = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: batch_images, self.node: batch_nodes}
                )
                if idx==0:
                    train_gen_data = predict_train
                if idx>0:
                    train_gen_data = np.concatenate((train_gen_data,predict_train),axis=0)
            predFC_train = train_gen_data[:,:,:,0]
            correlations,diff = evaluate(predFC_train,fc)

            actual_train_size = batch_idxs*self.batch_size
            if epoch == 0:
                total_loss_train = [loss_train / actual_train_size ]
                correlation_train = [np.nanmean(correlations)]
            else:
                total_loss_train.append(loss_train / actual_train_size )
                correlation_train.append(np.nanmean(correlations))


            counter += 1

            # # self.save(args.checkpoint_dir, counter)
            #
            ####### Compute test loss
            loss = 0
            batch_idxs_test = len(data_test)// self.batch_size
            for idx in xrange(0, batch_idxs_test):
                batch = data_test[idx * self.batch_size:(idx + 1) * self.batch_size]
                test_batch_images = np.array(batch).astype(np.float32)
                # print(node)
                node_ = node_test[idx * self.batch_size:(idx + 1) * self.batch_size]
                test_batch_nodes = np.array(node_).astype(np.float32)
                # print('idx,errD_fake + errD_real', idx, errD_fake + errD_real)
                predict_test = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: test_batch_images, self.node: test_batch_nodes}
                )
                errD_fake_test = self.d_loss_fake.eval({self.real_data: test_batch_images, self.node: test_batch_nodes})
                errD_real_test = self.d_loss_real.eval({self.real_data: test_batch_images, self.node: test_batch_nodes})
                errG_test = self.g_loss.eval({self.real_data: test_batch_images, self.node: test_batch_nodes})
                loss = loss + errD_fake_test + errD_real_test + errG_test
                if idx==0:
                    test_gen_data = predict_test
                if idx>0:
                    test_gen_data = np.concatenate((test_gen_data,predict_test),axis=0)
            predFC_test = test_gen_data[:,:,:,0]
            correlations,diff = evaluate(predFC_test,fc_test)

            test_count = batch_idxs_test*self.batch_size
            if epoch == 0:
                total_loss_test = [loss / test_count]
                correlation_test = [np.nanmean(correlations)]

            else:
                total_loss_test.append(loss / test_count)
                correlation_test.append(np.nanmean(correlations))
            if epoch%10==0 or epoch == args.epoch - 1:
                plt.figure()

                plt.subplot(211)
                plt.title('L1_lambda = {}'.format(self.L1_lambda))
                plt.plot(range(len(total_loss_train)), total_loss_train, 'r--', range(len(total_loss_test)), total_loss_test, 'bs')
                plt.legend(['train','test'])
                plt.ylabel('Average loss per sample')
                plt.xlabel('Epoch')
                plt.subplot(212)
                plt.title('L1_lambda = {}'.format(self.L1_lambda))
                plt.plot(range(len(correlation_train)), correlation_train, 'r--', range(len(correlation_test)),
                         correlation_test, 'bs')
                plt.legend(['train', 'test'])
                plt.ylabel('Average pearson correlation')
                plt.xlabel('Epoch')
                plt.show()

            if epoch % 200 == 0 or epoch == args.epoch - 1:
                print("Epoch: [%2d] [%4d/%4d] time: %4.4f, d_loss: %.8f, g_loss: %.8f" \
                      % (epoch, batch_idxs, batch_idxs,
                         time.time() - start_time, errD_fake + errD_real, errG))

                self.save(args.checkpoint_dir, counter)

                # print('avg value for weights: ', tf.reduce_mean(tf.abs(self.d2_w[:,:,:,11:])))

    def discriminator(self, image, node, y=None, reuse=False):

        with tf.variable_scope("discriminator") as scope:

            # image is n* 20 x 20 x (input_c_dim + output_c_dim)
            if reuse:
                tf.get_variable_scope().reuse_variables()
            else:
                assert tf.get_variable_scope().reuse == False
            h0 = lrelu(e2e(image, self.df_dim, self.image_size[0], self.image_size[0], name='d_h0_conv'))
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

    def generator(self, image, node, y=None):
        nn = 163
        with tf.variable_scope("generator") as scope:
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
            # b0 = features[0,:,:,:]
            # b0 = tf.reshape(b0,[1,1,nn,1])
            # b1 = tf.concat([b0,b0], axis=1)
            # c0 = features[0, :, :, :]
            # c0 = tf.reshape(c0, [1, 1, nn, 1])
            # for k in range(66):
            # b1 = tf.concat([b1, c0], axis=1)

            # b1 = tf.reshape(b1, [1, 68, 1, nn])
            # b1 = tf.reshape(b1, [self.batch_size, 68, 1, nn])
            # b2 = tf.concat([b1,b1], axis=0)
            # for k in range(self.batch_size-2):
            #   b2 = tf.concat([b2,b1], axis=0)
            # e3_new = tf.concat((e3_, b2), axis=3)

            # c = features[:, :, 0, :]
            # c1 = np.reshape(, (1, 68, 1, 1))

            # e3_new_1 = tf.concat((e3_, features), axis=3)
            # e3_new_2 = tf.concat((e3_new_1, features), axis=3)
            # e3_new = tf.concat((e3_new_2, features), axis=3)
            # self.d1, self.d1_w, self.d1_b = de_n2g(tf.nn.relu(e4),
            #   [self.batch_size, 300, 1, self.gf_dim*2], name='g_d1', with_w=True)
            # d1 = tf.nn.dropout(self.g_bn_d1(self.d1), 0.5)
            # d1 = tf.concat([d1, e3], 3)
            # d1 is (20 x 1 )

            self.d2, self.d2_w, self.d2_b = de_e2n(tf.nn.relu(e3_),
                                                   [self.batch_size, self.image_size[0], self.image_size[0],
                                                    self.gf_dim * 2], self.image_size[0], self.image_size[0],
                                                   name='g_d2', with_w=True)
            # w_var_flat = self.d2_w[0,:,0,:]
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
            out = tf.nn.relu(self.d4) * a
            out_list = []
            for i in range(out.shape[0]):
                out_list.append(tf.subtract(tf.add(out[i, :, :, 0], tf.transpose(out[i, :, :, 0])),
                                            tf.diag(tf.diag_part(out[i, :, :, 0]))))
            return tf.reshape(tf.stack(out_list),
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

    def test(self, args, sc, fc):
        def c_rmse1(a1, a2):
            return r2_score(a1, a2)  # np.sqrt(mean_squared_error(a1, a2))

        def c_rs2(a1, a2, a3):
            a = np.zeros((len(a1), len(a1)))
            for i in range(len(a1)):
                for j in range(len(a1)):
                    if a3[i, j] != 0 and a2[i, j] == 0:
                        a[i, j] = 1
            return r2_score(a1 * a, a2 * a)  # np.sqrt(mean_squared_error(a1*a, a2*a))

        def c_rmse2(a1, a2, a3):
            a = np.zeros((len(a1), len(a1)))
            for i in range(len(a1)):
                for j in range(len(a1)):
                    if a3[i, j] != 0 and a2[i, j] == 0:
                        a[i, j] = 1
            return sqrt(mean_squared_error(a1 * a, a2 * a))  # np.sqrt(mean_squared_error(a1*a, a2*a))

        gen_data = []
        init_op = tf.global_variables_initializer()
        self.sess.run(init_op)

        # load testing input
        print("Loading testing images ...")
        # sample_images_all, sample_nodes_all = load_data_diag(args.test_dir, 'test', self.image_size[0])
        sample_images_all, sample_nodes_all = load_data_diag_qz(sc, fc)
        sample_images = [sample_images_all[i:i + self.batch_size]
                         for i in xrange(0, len(sample_images_all), self.batch_size)]
        sample_images = np.array(sample_images)
        sample_nodes = [sample_nodes_all[i:i + self.batch_size]
                        for i in xrange(0, len(sample_nodes_all), self.batch_size)]
        sample_nodes = np.array(sample_nodes)
        if self.load(self.checkpoint_dir):
            print(" [*] Load SUCCESS")
        else:
            print(" [!] Load failed...")
        batch_loss_test = []
        for i, sample_image in enumerate(sample_images):
            idx = i + 1
            s = sample_images[i]
            if s.shape[0] == self.batch_size:
            # if True:
                # print("sampling image ", idx)
                samples = self.sess.run(
                    self.fake_B,
                    feed_dict={self.real_data: sample_images[i], self.node: sample_nodes[i]}
                )

                if i == 0:
                    gen_data = samples
                    d3 = self.sess.run(
                        self.d3,
                        feed_dict={self.real_data: sample_images[i], self.node: sample_nodes[i]}
                    )
                if i > 0: gen_data = np.concatenate((gen_data, samples), axis=0)

                '''errD_fake = self.d_loss_fake.eval({self.real_data: sample_images[i], self.node: sample_nodes[i]})
                errD_real = self.d_loss_real.eval({self.real_data: sample_images[i], self.node: sample_nodes[i]})
                errG = self.g_loss.eval({self.real_data: sample_images[i], self.node: sample_nodes[i]})
  
                loss = errD_fake+errD_real+errG
                if i==0:
                    batch_loss_test = loss
                    batch_loss_test = np.reshape(batch_loss_test, [1,1])
                else:
                    batch_loss_test = np.concatenate((batch_loss_test, loss), axis=0)
  
  
          for i in range(gen_data.shape[0]):
              for m in range(20):
                  for n in range(20):
                      if gen_data[i, m, n, 0] < 1: gen_data[i, m, n, 0] = 0
          np.save('gen20.npy', gen_data[:, :, :, 0])
          np.save('input20.npy', sample_images_all[:, :, :, 0])
          np.save('output20.npy', sample_images_all[:, :, :, 4])'''
        return gen_data