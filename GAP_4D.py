import numpy as np
import tensorflow as tf
from sklearn import preprocessing

a_d2 = np.load('average_batchOfSubs_day2.npy')

#data_4D_batchofSubs = np.load('n4_tensor_saliency_batchOfSubs_day2.npy')
#n4_tensor_saliency_batchOfSubs_day1
#n4_tensor_saliency_batchOfSubs_day2-correctVersion
data_4D_batchofSubs = np.load('n4_tensor_saliency_batchOfSubs_day2-correctVersion.npy')

GAP_input = np.zeros((68, 68, 68, 68))

GAP_output_all_subjects = np.zeros((68,68))

for subject in range(20):

    data_4D = data_4D_batchofSubs[subject, :, :, :]

    #zeroing out gradients before GAP
    #data_4D[data_4D<0] = 0

    m = 0

    for i in range(68):

        for k in range(68):

            GAP_input[i, :, k, :] = data_4D[m, :, :]
            m = m+1

    GAP_input_tensor = tf.convert_to_tensor(GAP_input, dtype=tf.float32)

    gap = tf.keras.layers.GlobalAveragePooling2D()
    GAP_output = gap(GAP_input_tensor)

    init_op = tf.initialize_all_variables()

    #run the graph
    with tf.Session() as sess:
        sess.run(init_op) #execute init_op
        GAP_output_value = sess.run(GAP_output)

    #zeroing out gradients after GAP
    GAP_output_value[GAP_output_value<0] = 0
    GAP_output_all_subjects = GAP_output_value + GAP_output_all_subjects

np.save('average_batchOfSubs_day2', (1/20)* GAP_output_all_subjects)

#GAP_output_value_normalized = preprocessing.normalize(GAP_output_value)


########################Infomap Input###########################

# 1 node1 2 node2 value
f = open("interlayer_day1_sub5_allLinks.txt", "w")
for i in range(68):
    for j in range(68):
        if j==i or j>i:
           f.write("\n1 {} 2 {} {}".format(i+1, j+1, GAP_output_value[i,j]))

f.close()

f = open("node_list.txt", "w")
for i in range(68):
           s = ""
           f.write("\n{} 'node {}' ".format(i+1, i+1))
f.close()

#################################Infomap output##############################3

#layer1 = sc_train[0, :, :, 0]
#layer2 = fc_train[0, :, :]

nodes_m1_layer1 = []
nodes_m1_layer2 = []
nodes_m2_layer1 = []
nodes_m2_layer2 = []
nodes_m3_layer1 = []
nodes_m3_layer2 = []

f = open("SC_FC_community.txt", "r")
lines = f.readlines()
for line in lines:
    line_strip = line.strip()
    line_st_sp = line_strip.split(" ")
    if len(line) > 1:
     if line_st_sp[1] == '2': #this is module 2
        if line_st_sp[-1] == '1': # this is layer 1
            nodes_m2_layer1.append(int(line_st_sp[-2]))
        elif line_st_sp[-1] == '2': # this is layer 2
            nodes_m2_layer2.append(int(line_st_sp[-2]))

nodes_m2_layer1.sort()
nodes_m2_layer2.sort()
nodes_m1_layer1 == nodes_m1_layer2

f.close()


f = 1

