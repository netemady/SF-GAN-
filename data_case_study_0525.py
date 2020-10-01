import numpy as np
from helper_n0203 import evaluate_n, getNormalizedMatrix_np
import matplotlib.pyplot as plt

folder_pred = 'prediction_CV_test/'

#predFC = np.load('prediction_test_SC_EMOTION_partialCorrelationFC_L2_set1_lm100.0.npy')
#predFC = np.load(folder_pred+'prediction_test_SC_RESTINGSTATE_correlationFC_set3_lm1.npy')
predFC = np.load('prediction_test_SC_RESTINGSTATE_partialcorrelationFC_L2_set1_lm10000.0.npy')

folder = 'SC_FC_dataset_0905/'

#file = 'SC_RESTINGSTATE_correlationFC'
file = 'SC_RESTINGSTATE_partialcorrelationFC_L2'
#file = 'SC_EMOTION_partialcorrelationFC_L2'



for fc_i in range(1):

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
        dataset_SC = allSC[list(range(100, entire_size))]
        dataset_FC = allFC[list(range(100, entire_size))]

        sc_train = dataset_SC
        fc_train = dataset_FC

        sc_test = allSC[list(range(0, test_size))]
        fc_test = allFC[list(range(0, test_size))]


#correlations, diff = evaluate_n(predFC, fc_test)
#list = []
#correlations_sort = np.argsort(correlations)

#Rest_l2_images = [4,23,25,30,31,32,43,46,65,73,74,83,96]

Rest_l2_images = [7,21,34,51,57,69,91]

#Rest_l2_images = [95,35,48,22,80,70,60,50]

'''for i in range(correlations.shape[0]):
    if correlations[i]>0.63:
        list.append(i)
        fc_true = fc_test[i, :, :]
        fc_pred = predFC[i, :, :, 0]
        #tfc_s = 'fc_true_Rest_0711 %s' %i
        #pfc_s = 'fc_pred_Rest_set3_0711 %s' %i
        #sc_s = 'sc_Rest_L2 %s' %i'''

for i in Rest_l2_images:

        sc = sc_test[i,:,:,0]
        fc_true = fc_test[i, :, :]
        fc_pred = predFC[i, :, :, 0]

        #sc_s = 'sc_RestL2 %s .edge' %i

        #np.save(tfc_s, fc_true)
        #np.save(pfc_s, fc_pred)
        #np.save(sc_s, sc)
        #np.savetxt(sc_s, sc, delimiter='     ')
        plt.figure(figsize=(10, 3))
        plt.subplot(131)
        plt.xlabel('Brain ROI index')
        plt.ylabel('Brain ROI index')
        plt.title('SC')
        plt.imshow(sc, cmap='hot', interpolation='nearest')
        #plt.clim(-0.1, 0.3)
        plt.colorbar()

        plt.subplot(132)
        plt.imshow(fc_true, cmap='hot', interpolation='nearest')
        plt.xlabel('Brain ROI index')
        # plt.ylabel('Brain ROI index')
        plt.title('Empirical FC')
        plt.clim(-0.2, 0.2)
        plt.colorbar()

        plt.subplot(133)
        plt.imshow(fc_pred, cmap='hot', interpolation='nearest')
        plt.xlabel('Brain ROI index')
        # plt.ylabel('Brain ROI index')
        plt.title('Predicted FC')
        plt.clim(-0.2, 0.2)
        plt.colorbar()

        plt.savefig('case0715_GTGAN_RestL2_'+str(i)+'.pdf')
        plt.show()
        a = 1



t=0