import numpy as np
import tensorflow as tf
from keras.models import *
from keras.layers import *
from keras.callbacks import *
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy.linalg as la
import datetime
import keras.backend as K
epsilon = K.epsilon()
from io import BytesIO, StringIO
from tensorflow.python.lib.io import file_io
import torch

################################################################################
flag_show_plots = False # True for Notebooks, False otherwise
if flag_show_plots:
    import matplotlib.pyplot as plt
    from matplotlib.pyplot import figure

################################################################################
dirlocal = './dataset/'
dirgcp = 'gs://protein-distance/'
dirpredictions = './predictions/' # only if building 3D models

dataset = 'full' # 'sample' or 'full'

stamp = datetime.datetime.now().strftime('%m_%d_%Y_%H_%M_%S_%f')
modelfile = 'model-' + str(stamp) + '.h5'

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
#################################################################################################
def make_new_XTRAIN(X):
    print(X.shape)
    X_new= np.full((X.shape[0], 256, 256, 8), 0.0)
    for i in range(X.shape[0]):
        k=0;
        for j in range(0,10,2):
            C=X[i,:,:,j]+X[i,:,:,j+1]
            print("C.shape",C.shape)
            #print(X_new[i,:,].shape)
            X_new[i,:,:,k]=(X[i,:,:,j]/2)+(X[i,:,:,j+1]/2)
            print(X_new[i,:,:,k].shape)
            print(checkTranspose(X_new[i,:,:,k],X_new[i,:,:,k].T))
            k+=1
        for j in range(10,13):
            X_new[i,:,:,k]=X[i,:,:,j]
            k+=1
    print(k)
    return X_new

###################################################################################
def make_features():
    print("prepare test data features")
    print('')
    print('Load distance maps..')
##################Test data Preparation############################################
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
    XTEST_new_tensorflow = make_new_XTRAIN(XTEST)
    XTEST_new_pytorch=np.array(torch.tensor(XTEST_new_tensorflow).permute(0,3,1,2))
    YTEST_new_pytorch=np.array(torch.tensor(YTEST).permute(0,3,1,2))
    np.save("XTEST_channels_8_torch__1sub",XTEST_new_pytorch)
    np.save("YTEST_channels_8_torch_1sub",YTEST_new_pytorch)
    #####################Train and Valid Data Preparation##############################
    print ('Prepare the training input and outputs')
    print('')
    print('Load input features..')

    x = dirlocal + dataset + '-input-featuresnpy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-input-features.npy', binary_mode=True))
    (pdb_list, length_dict, input_features) = np.load(x,allow_pickle=True, encoding='latin1')

    print('')
    print('Load distance maps..')
    x = dirlocal + dataset + '-distance-maps-cb.npy'
    if not os.path.isfile(x):
        x = BytesIO(file_io.read_file_to_string(dirgcp + dataset + '-distance-maps-cb.npy', binary_mode=True))
    (pdb_list_y, distance_maps_cb) = np.load(x,allow_pickle=True, encoding='latin1')

    #########################################################################################################
    print('')
    print ('Some cross checks on data loading..')
    for pdb in pdb_list:
        if not pdb in pdb_list_y:
            print ('I/O mismatch ', pdb)
            sys.exit(1)
    ################################################################################
    print('')
    print('Find the number of input channels..')
    F = determine_number_of_channels(input_features, pdb_list, length_dict)
    ################################################################################
    print('')
    print('Split into training and validation set (20%)..')
    split = int(0.2 * len(pdb_list))
    valid_pdbs = pdb_list[:split]
    train_pdbs = pdb_list[split:]
    print('Total validation proteins = ', len(valid_pdbs))
    print('Total training proteins = ', len(train_pdbs))

    ################################################################################
    print('')
    print ('Prepare the validation input and outputs..')
    XVALID, YVALID = prepare_input_features_2D(valid_pdbs, input_features, distance_maps_cb, length_dict, F)
    XVALID_new_tensorflow = make_new_XTRAIN(XVALID)
    XVALID_new_pytorch=np.array(torch.tensor(XVALID_new_tensorflow).permute(0,3,1,2))
    YVALID_new_pytorch=np.array(torch.tensor(YVALID).permute(0,3,1,2))
    np.save("XVALID_channels_8_torch__1sub",XVALID_new_pytorch)
    np.save("YVALID_channels_8_torch__1sub",YVALID_new_pytorch)

    #################################################################################
    print ('Prepare the training input and outputs..')
    XTRAIN, YTRAIN = prepare_input_features_2D(train_pdbs, input_features, distance_maps_cb, length_dict, F)
    XTRAIN_new_tensorflow=make_new_XTRAIN(XTRAIN)
    XTRAIN_new_pytorch=np.array(torch.tensor(XTRAIN_new_tensorflow).permute(0,3,1,2))
    YTRAIN_new_pytorch=np.array(torch.tensor(YTRAIN).permute(0,3,1,2))
    np.save("XTRAIN_channels_8_torch",XTRAIN_new_pytorch)
    np.save("YTRAIN_channels_8_torch",YTRAIN_new_pytorch)
    print(YTRAIN.shape)
    ##################################################################################


