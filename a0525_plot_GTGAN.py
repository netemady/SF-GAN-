import numpy as np
from matplotlib import pyplot as plt
from helper import getNormalizedMatrix_np
from helper import evaluate

# In[L2]


'''allSC = np.load('pred0128/raw_sc_test_RESTING_L2.npy')
trueFC = np.load('pred0128/raw_fc_test_RESTING_L2.npy')
predFC = np.load('pred0128/prediction_SC_RESTINGSTATE_partialcorrelationFC_L2_set3_lm1000.0.npy')[:,:,:,0]
pearson,diff = evaluate(predFC,trueFC,normalize=True)'''
#print(np.mean(pearson),np.mean(diff))

k=0
idx = 10

sc = np.load('sc_0525 10.npy')
tfc = np.load('fc_true_0525 10.npy')
pfc = np.load('fc_pred_0525 10.npy')


if k==0:
        plt.figure(figsize=(10, 3))
        plt.subplot(131)
        plt.xlabel('Brain ROI index')
        plt.ylabel('Brain ROI index')
        plt.title('SC')
        plt.imshow(sc, cmap='hot', interpolation='nearest')
        #plt.clim(-0.1, 0.3)
        plt.colorbar()
        plt.subplot(132)
        plt.imshow(tfc, cmap='hot', interpolation='nearest')
        plt.xlabel('Brain ROI index')
        # plt.ylabel('Brain ROI index')
        plt.title('Empirical FC')
        plt.clim(-0.2, 0.2)
        plt.colorbar()
        plt.subplot(133)
        plt.imshow(pfc, cmap='hot', interpolation='nearest')
        plt.xlabel('Brain ROI index')
        # plt.ylabel('Brain ROI index')
        plt.title('Predicted FC')
        plt.clim(-0.2, 0.2)
        plt.colorbar()

        plt.savefig('case0525_GTGAN_emotionL2_'+str(idx)+'.pdf')
        plt.show()

a=0