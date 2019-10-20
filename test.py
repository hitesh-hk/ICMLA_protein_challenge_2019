################################################################################

import numpy as np
#import tensorflow as tf
#from keras.models import *
#from keras.layers import *
# from keras.callbacks import *
# from keras.models import load_model
import datetime
# import keras.backend as K
# epsilon = K.epsilon()
epsilon = 1e-7
from io import BytesIO, StringIO
# from tensorflow.python.lib.io import file_io
import argparse

################################################################################
flag_show_plots = False # True for Notebooks, False otherwise
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

################################################################################
dirlocal = './dataset/'
dirgcp = 'gs://protein-distance/'
dataset = 'full' # 'sample' or 'full'
stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
modelfile = 'model-' + str(stamp) + '.h5'
max_epochs = 64
es_patience = 32
if dataset == 'sample':
    max_epochs = 8
    es_patience = 1

################################################################################
def determine_number_of_channels(input_features, pdb_list, length_dict):
    F = 0
    x = input_features[pdb_list[0]]
    l = length_dict[pdb_list[0]]
    for feature in x:
        if len(feature) == l:
            F += 2
        elif len(feature) == l * l:
            F += 1
        else:
            print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
            sys.exit(1)
    return F

################################################################################
def print_max_avg_sum_of_each_channel(x):
    print(' Channel        Avg        Max        Sum')
    for i in range(len(x[0, 0, :])):
        (m, s, a) = (x[:, :, i].flatten().max(), x[:, :, i].flatten().sum(), x[:, :, i].flatten().mean())
        print(' %7s %10.4f %10.4f %10.1f' % (i, a, m, s))


################################################################################
# Roll out 1D features to two 2D features, all to 256 x 256 (because many are smaller)
def prepare_input_features_2D(pdbs, input_features, distance_maps_cb, length_dict, F):
    X = np.full((len(pdbs), 256, 256, F), 0.0)
    Y = np.full((len(pdbs), 256, 256, 1), 100.0)
    for i, pdb in enumerate(pdbs):
        x = input_features[pdb]
        y = distance_maps_cb[pdb]
        l = length_dict[pdb]
        newi = 0
        xmini = np.zeros((l, l, F))
        for feature in x:
            feature = np.array(feature)
            feature = feature.astype(np.float)
            if len(feature) == l:
                for k in range(0, l):
                    xmini[k, :, newi] = feature
                    xmini[:, k, newi + 1] = feature
                newi += 2
            elif len(feature) == l * l:
                xmini[:, :, newi] = feature.reshape(l, l)
                newi += 1
            else:
                print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
                sys.exit(1)
        if l > 256:
            l = 256
        X[i, 0:l, 0:l, :] = xmini[:l, :l, :]
        Y[i, 0:l, 0:l, 0] = y[:l, :l]
    return X, Y
################################################################################
def plot_input_output_of_this_protein(X, Y):
    figure(num=None, figsize=(16, 16), dpi=80, facecolor='w', frameon=True, edgecolor='k')
    for i in range(13):
        plt.subplot(7, 7, i + 1)
        plt.grid(None)
        plt.imshow(X[:, :, i], cmap='RdYlBu', interpolation='nearest')
    # Last plot is the true distance map
    plt.subplot(7, 7, 14)
    plt.grid(None)
    plt.imshow(Y[:, :], cmap='Spectral', interpolation='nearest')
    plt.show()

################################################################################
def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    plot_count = 0
    if flag_show_plots:
        plot_count = 4
    avg_mae = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_mae = 0.0
        for pair in top_pairs:
            abs_dist = abs(y_dict[pair] - p_dict[pair])
            sum_mae += abs_dist
        sum_mae /= L
        avg_mae += sum_mae
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + ' = %.2f' % sum_mae)
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(0, L):
                    if not (j, k) in top_pairs:
                        P[j, k] = np.inf
                        Y[j, k] = np.inf
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average MAE = %.2f' % (avg_mae / len(PRED[:, 0, 0, 0])))


def make_new_XTRAIN(X):
    print(X.shape)
    X_new= np.full((X.shape[0], 256, 256, 8), 0.0)
    for i in range(X.shape[0]):
        k=0;
        for j in range(0,10,2):
            X_new[i,:,:,k]=(X[i,:,:,j]/2)+(X[i,:,:,j+1]/2)
            k+=1
        for j in range(10,13):
            X_new[i,:,:,k]=X[i,:,:,j]
            k+=1
    print(k)
    return X_new

x = dirlocal + 'testset-input-features.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-input-features.npy', binary_mode=True))
(pdb_list, length_dict, sequence_dict, input_features)  = np.load(x)
x = dirlocal + 'testset-distance-maps-cb.npy'
if not os.path.isfile(x):
    x = BytesIO(file_io.read_file_to_string(dirgcp + 'testset-distance-maps-cb.npy', binary_mode=True))
(pdb_list_y, distance_maps_cb) = np.load(x)
F = determine_number_of_channels(input_features, pdb_list, length_dict)
XTEST, YTEST = prepare_input_features_2D(pdb_list, input_features, distance_maps_cb, length_dict, F)


for pdb in length_dict:
    if length_dict[pdb] > 256:
        length_dict[pdb] = 256


length_dict[pdb_list[1]]
len(pdb_list)==len(pdb_list_y)==len(length_dict)
print(XTEST.shape)
XTEST=make_new_XTRAIN(XTEST)
print(XTEST.shape)


import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.autograd import Variable
# from build_model import *
import os
from tqdm import tqdm
#from tensorboardX import SummaryWriter
from torch.optim.lr_scheduler import MultiStepLR
from dataloader_un import npy_train,npy_valid,npy_test
# from resnet_unet18_1 import ResNetUNet
from model import UNet11
# import keras.backend as K
# epsilon = K.epsilon()
epsilon = 1e-7
import numpy as np
from io import BytesIO, StringIO
# from tensorflow.python.lib.io import file_io
#from Transforms import *
from build_unetmodel import UNet
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
# #--------------------------

class Average(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.sum += val
        self.count += n

    @property
    def avg(self):
        return self.sum / self.count
#------------------------------                                      

class mse(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(mse, self).__init__()
        
    def forward(self, PRED, YTRUE):
        # print('MSE started')
        train_loss1 = F.mse_loss(PRED, YTRUE, True)
        # print('MSE calculated')
        return train_loss1
        
       
class mae(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(mae, self).__init__()
        
    def forward(self, PRED, YTRUE):
        # print('MAE started')
        train_loss1 = F.smooth_l1_loss(PRED, YTRUE, True)
        # print('MAE calculated')
        return train_loss1

class symmetry_mse(nn.Module):
    '''
    custom loss
    '''
    def __init__(self, weight=None, size_average=True):
        super(symmetry_mse, self).__init__()
        
    def forward(self, y_pred):
    
        train_loss1 = F.mse_loss(y_pred, y_pred.permute(0,1,3,2), True)
        return train_loss1
#--------------
DATA_DIR='/home/SSD/protien_ICMLA/'

LOGGING = False
BATCH_SIZE = 1  
NUM_EPOCHS = 1
LEARNING_RATE = 10E-4
LEARNING_RATE_DECAY = 0.8
WEIGHT_TENSOR = torch.Tensor([2,2,2])
WEIGHT_DECAY = 0.01
MODEL = 'unet'
phase = 'test'
SUMMARY_NAME = 'PROTIEN'+'_'+str(BATCH_SIZE)+'_'+str(LEARNING_RATE)+'_'+str(LEARNING_RATE_DECAY)+'_'+str(WEIGHT_DECAY)+'_'+str(MODEL)+'_'+str(phase)+"_restart_497_N"
# writer = SummaryWriter('./runs/'+SUMMARY_NAME)

#######################LOGGING######################
if LOGGING:
    writer = SummaryWriter('./runs/'+SUMMARY_NAME)
    print("**********LOGGING IS ON***************\n"*10)
else:
    print("**********LOGging IS Off***************\n"*10)



flag_show_plots = False
#########################################
def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    plot_count = 0
    if flag_show_plots:
        plot_count = 4
    avg_mae = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        print(i)
        L = length_dict[pdb_list[i]]
        if(L>256):
            L=256
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                #print(i,j,k)
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_mae = 0.0
        for pair in top_pairs:
            abs_dist = abs(y_dict[pair] - p_dict[pair])
            sum_mae += abs_dist
        sum_mae /= L
        avg_mae += sum_mae
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + ' = %.2f' % sum_mae)
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(0, L):
                    if not (j, k) in top_pairs:
                        P[j, k] = np.inf
                        Y[j, k] = np.inf
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average MAE = %.2f' % (avg_mae / len(PRED[:, 0, 0, 0])))

#--------------
DATA_DIR='/home/SSD/protien_ICMLA/'
LOGGING = False
BATCH_SIZE = 1  
NUM_EPOCHS = 1
LEARNING_RATE = 10E-4
LEARNING_RATE_DECAY = 0.8
WEIGHT_TENSOR = torch.Tensor([2,2,2])
WEIGHT_DECAY = 0.01
MODEL = 'unet'
phase = 'test'
SUMMARY_NAME = 'PROTIEN'+'_'+str(BATCH_SIZE)+'_'+str(LEARNING_RATE)+'_'+str(LEARNING_RATE_DECAY)+'_'+str(WEIGHT_DECAY)+'_'+str(MODEL)+'_'+str(phase)+"_check_restart_497_N"
# writer = SummaryWriter('./runs/'+SUMMARY_NAME)

#######################LOGGING######################
if LOGGING:
    writer = SummaryWriter('./runs/'+SUMMARY_NAME)
    print("**********LOGGING IS ON***************\n"*10)
else:
    print("**********LOGging IS Off***************\n"*10)



flag_show_plots = False
#########################################
def calculate_mae(PRED, YTRUE, pdb_list, length_dict):
    plot_count = 0
    if flag_show_plots:
        plot_count = 4
    avg_mae = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        print(i)
        L = length_dict[pdb_list[i]]
        if(L>256):
            L=256
        P = np.zeros((L, L))
        # Average the predictions from both triangles (optional)
        # This can improve MAE by upto 6% reduction
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = np.inf
                    Y[j, k] = np.inf
        p_dict = {}
        y_dict = {}
        for j in range(0, L):
            for k in range(0, L):
                p_dict[(j,k)] = P[j, k]
                y_dict[(j,k)] = Y[j, k]
        top_pairs = []
        x = L
        for pair in sorted(p_dict.items(), key=lambda x: x[1]):
            (k, v) = pair
            top_pairs.append(k)
            x -= 1
            if x == 0:
                break
        sum_mae = 0.0
        for pair in top_pairs:
            abs_dist = abs(y_dict[pair] - p_dict[pair])
            sum_mae += abs_dist
        sum_mae /= L
        avg_mae += sum_mae
        print('MAE for ' + str(i) + ' - ' + str(pdb_list[i]) + ' = %.2f' % sum_mae)
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(0, L):
                    if not (j, k) in top_pairs:
                        P[j, k] = np.inf
                        Y[j, k] = np.inf
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average MAE = %.2f' % (avg_mae / len(PRED[:, 0, 0, 0])))

################################################################################
def calculate_longrange_contact_precision(PRED, YTRUE, pdb_list, length_dict):
    if flag_show_plots:
        plot_count = 4
    avg_precision = 0.0
    for i in range(0, len(PRED[:, 0, 0, 0])):
        L = length_dict[pdb_list[i]]
        if(L>256):
            L=256
        P = np.zeros((L, L))
        # Average the predictions from both triangles
        for j in range(0, L):
            for k in range(0, L):
                P[k, j] = (PRED[i, k, j, 0] + PRED[i, j, k, 0]) / 2.0
        Y = np.copy(YTRUE[i, 0:L, 0:L, 0])
        for j in range(0, L):
            for k in range(0, L):
                if k - j < 24:
                    P[j, k] = 0
                    Y[j, k] = 0
        for j in range(0, L):
            for k in range(0, L):
                if P[j, k] < 8.0 and P[j, k] > 0.001:
                    P[j, k] = 1
                else:
                    P[j, k] = 0
                if Y[j, k] < 8.0 and Y[j, k] > 0.001:
                    Y[j, k] = 1
                else:
                    Y[j, k] = 0
        matches = np.logical_and(P, Y).sum()
        precision = matches / (Y.sum() + epsilon)
        avg_precision += precision
        print('Precision for ' + str(i) + ' - ' + str(pdb_list[i]) +  ' ' + str(L) + ' [' + str(matches) + '/' + str(Y.sum()) + '] = %.2f ' % precision)
        plot_count = 0
        # Contact maps visualization of prediction against truth
        # Legend: lower triangle = true, upper triangle = prediction
        if plot_count > 0:
            plot_count -= 1
            for j in range(0, L):
                for k in range(j, L):
                    P[k, j] = Y[j, k]
            plt.grid(None)
            plt.imshow(P, cmap='RdYlBu', interpolation='nearest')
            plt.show()
    print('Average Precision = %.2f' % (avg_precision / len(PRED[:, 0, 0, 0])))

########################################3

XTEST = np.load('./XTEST_channels_8_torch.npy')
YTEST = np.load('./YTEST_channels_8_torch.npy')
print(XTEST.shape)
print(YTEST.shape)
print(XTEST.shape)
print(YTEST.shape)
print(XTEST[0].shape)
print(YTEST[0].shape)
print(XTEST[0].reshape(1,256,256,8).shape)

length_dict[pdb_list[1]]

def test():
    cuda = torch.cuda.is_available()
    predict_test_y_mae=[]
    predict_test_y_pre=[]
    ytrue_test_y=[]
    # net = ResNetUNet(1)
    net = UNet11(1,32,8)
    # net = UNet(8,1)
    # print(net)
    if cuda:
        net = net.cuda()
    #-------------------------------------
    net.load_state_dict(torch.load('./PROTIEN_16_0.001_0.8_0.01_vgg11_dilation_normalized_UNorm_CEloss_96.pth'))
    #-------------------------
    # criterion1 = mse().cuda()
    # criterion2 = mae().cuda()
    # criterion3 = symmetry_mse().cuda()
    criterion1 = mse()
    criterion2 = mae()
    criterion3 = symmetry_mse()
    #-------------------------------
    optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE,weight_decay = WEIGHT_DECAY)
    scheduler = MultiStepLR(optimizer, milestones=[5,25,75,125,200], gamma=LEARNING_RATE_DECAY)
    #----------------------------------------------
    print("preparing testing data ...")


    test_iter_count = 150 #length of test samples    

    # ------ Test -------
    for epoch in tqdm(range(NUM_EPOCHS)):
         
        # ######################Testing###########
        
        val_loss1 = Average()
        val_loss2 = Average()
        val_loss3 = Average()
        val_loss = Average()
        net.eval()
        print("starting Testing")
        for i in range(150):
            images_val = Variable(torch.Tensor(np.expand_dims(XTEST[i],axis=0)).float())
            masks_val = Variable(torch.Tensor(np.expand_dims(YTEST[i],axis=0)).float())
            if cuda:
                images_val = images_val.cuda()
                masks_val = masks_val.cuda()
            
            # print(images_val.size())

            outputs = net(images_val)

            if epoch == NUM_EPOCHS-1:
                predict_test_y_mae.extend(outputs[0].data.cpu().numpy())
                predict_test_y_pre.extend(outputs[1].data.cpu().numpy())
                ytrue_test_y.extend(masks_val.data.cpu().numpy())


    # print('Length of predict_test_y',len(predict_test_y_mae))
    np.save("upload_pred_test_mae",np.array(predict_test_y_mae))
    np.save("upload_pred_test_pre",np.array(predict_test_y_pre))
    # print('Length of true_test_y',len(ytrue_test_y))
    np.save("upload_true_test",np.array(ytrue_test_y))
    return net

if __name__ == "__main__":
    print("I am in test")
    test()
    print('')
    Q_pre = np.load('upload_pred_test_pre.npy')
    
    P_pre = torch.Tensor(Q_pre).permute(0,2,3,1)

    Q_mae=np.load('upload_pred_test_mae.npy')
    P_mae=torch.Tensor(Q_mae).permute(0,2,3,1)
    YTEST=np.load('upload_true_test.npy')
    YTEST_GT=torch.Tensor(YTEST).permute(0,2,3,1)
    # print('YTEST_last',YTEST_GT.size())
    # print('PredTEST_last_mae',P_mae.size())
    # print('Pred_pre',P_pre.size())
    print('MAE of top L long-range distance predictions on the test set..')
    calculate_mae(P_mae, YTEST_GT, pdb_list, length_dict)
    print('')
    print('Precision of top L long-range distance predictions on the test set..')
    calculate_longrange_contact_precision(P_pre, YTEST_GT, pdb_list, length_dict)

