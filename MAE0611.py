import numpy as np
test_size = 100
N = 68
#set = [1,2,3]

folder = 'true_fc/'
#folder2 = 'CV-test\SF-GAN/'
folder_ = 'prediction_2008/'

#files = ['SC_RESTINGSTATE_correlationFC','SC_EMOTION_correlationFC','SC_GAMBLING_correlationFC', 'SC_LANGUAGE_correlationFC', 'SC_MOTOR_correlationFC','SC_RELATIONAL_correlationFC', 'SC_SOCIAL_correlationFC',  'SC_WM_correlationFC']
files = ['SC_EMOTION_partialCorrelationFC_L1', 'SC_GAMBLING_partialCorrelationFC_L1', 'SC_LANGUAGE_partialCorrelationFC_L1', 'SC_MOTOR_partialCorrelationFC_L1', 'SC_RELATIONAL_partialCorrelationFC_L1', 'SC_SOCIAL_partialCorrelationFC_L1', 'SC_WM_partialCorrelationFC_L1']
files = ['SC_RESTINGSTATE_partialcorrelationFC_L2','SC_EMOTION_partialCorrelationFC_L2', 'SC_GAMBLING_partialCorrelationFC_L2',
         'SC_LANGUAGE_partialCorrelationFC_L2', 'SC_MOTOR_partialCorrelationFC_L2',
         'SC_RELATIONAL_partialCorrelationFC_L2', 'SC_SOCIAL_partialCorrelationFC_L2', 'SC_WM_partialCorrelationFC_L2']

#prediction = ['prediction_test_SC_EMOTION_correlationFC_set2_lm1']

#files = ['SC_EMOTION_correlationFC']
#s_nofeat = 'prediction_nofeat_test_'
#prediction_set = np.load('prediction_test_SC_EMOTION_correlationFC_set3_lm1.npy')
#fc_pred_2016 = np.load('fc_pred_2016_Emotion_full_0610.npy')

for file in files:
  print(file)
  s_true = "fc_true_%s" % file
  #s_pred_2016 = 'fc_pred_2016_%s' % file
  s_pred_2014 = 'fc_pred_2008_%s' % file

  # data_nofeat = np.load(folder2 + s_nofeat + file + '.npy')
  data_true = np.load(folder + s_true + '.npy')
  fc_pred_2014 = np.load(folder_ + s_pred_2014 + '.npy')

  #for setting in set:

    #s_true = "fc_true_%s_full" %file

    #s_SFGAN = "prediction_test_%s_set%d_lm" % (file, setting)

    #data_SFGAN = np.load(folder2 + s_SFGAN + '.npy')

  var_ = 0
    #var_nofeat = 0
    #var_set = 0

  for i in range(test_size):

       fc_true = data_true[i]
       #pred_nofeat = data_nofeat[i]
       pred_2014 = fc_pred_2014[i]
       #pred_set = data_SFGAN[i,:,:,0]

       diff_abs = abs(pred_2014 - fc_true)
       diff_abs_triu_flatten = diff_abs[np.triu_indices(N)]
       var_ = var_ + np.mean(diff_abs_triu_flatten)

       '''diff_abs = abs(pred_nofeat - fc_true)
       diff_abs_triu_flatten = diff_abs[np.triu_indices(N)]
       var_nofeat = var_nofeat + np.mean(diff_abs_triu_flatten)

       diff_abs = abs(pred_set - fc_true)
       diff_abs_triu_flatten = diff_abs[np.triu_indices(N)]
       var_set = var_set + np.mean(diff_abs_triu_flatten)'''

    #print('mean_set:', var_set/test_size)
  print('mean_:', var_ / test_size)
  # print('mean_nofeat:', var_nofeat/test_size)
