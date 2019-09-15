# 3-28-2019
# Badri Adhikari
# https://badriadhikari.github.io/

# This script loads a trained model and predicts (writes) topL long-range distances in the CASP RR format
# See RR format description at http://predictioncenter.org/casp8/index.cgi?page=format#RR

################################################################################
from keras.models import load_model
import numpy as np

################################################################################
dirlocal = './dataset/'
diroutput = './predictions/'
modelfile = 'model-03_29_2019_12_29_00_659706.h5'

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
# Roll out 1D features to two 2D features, all to 256 x 256 (because many are smaller)
def prepare_input_features_2D(this_input_features, L, F):
    x = np.zeros((L, L, F))
    newi = 0
    for feature in this_input_features:
        feature = np.array(feature)
        feature = feature.astype(np.float)
        if len(feature) == L:
            for k in range(0, L):
                x[k, :, newi] = feature
                x[:, k, newi + 1] = feature
            newi += 2
        elif len(feature) == L * L:
            x[:, :, newi] = feature.reshape(L, L)
            newi += 1
        else:
            print('Expecting features to be either L or L*L !! Something went wrong!!', l, len(feature))
            sys.exit(1)
    X = np.zeros((1, 256, 256, F))
    if L > 256:
        X[0, :256, :256, :] = x[:256, 0:256, :]
    else:
        X[0, :L, :L, :] = x
    return X

################################################################################
print('Load the model..')
model = load_model(modelfile, compile = False)

x = dirlocal + 'testset-input-features.npy'
(pdb_list, length_dict, sequences, input_features)  = np.load(x)
x = dirlocal + 'testset-distance-maps-cb.npy'
(pdb_list_y, distance_maps_cb) = np.load(x)

F = determine_number_of_channels(input_features, pdb_list, length_dict)

################################################################################
for pdb in sorted(pdb_list):
    print (pdb)
    x = prepare_input_features_2D(input_features[pdb], length_dict[pdb], F)
    PRED = model.predict(x)
    L = length_dict[pdb]
    if L > 256:
        L = 256
    PRED = PRED[0, :L, :L, 0]
    # Average the predictions from both triangles (optional)
    # This can improve MAE by upto 6% reduction
    P = np.zeros((L, L))
    for j in range(0, L):
        for k in range(0, L):
            P[j, k] = (PRED[k, j] + PRED[j, k]) / 2.0
    f = open(diroutput + pdb + '.dmap', 'w')
    for j in range(0, L):
        for k in range(0, L):
            f.write("%.2f " % P[j, k])
        f.write('\n')
    f.close()
    # Write top L predicted long-range distances to a separate file
    SEQSEP = 24
    for j in range(0, L):
        for k in range(0, L):
            if k - j < SEQSEP:
                P[j, k] = np.inf
    p_dict = {}
    for j in range(0, L):
        for k in range(0, L):
            p_dict[(j,k)] = P[j, k]
    x = L
    DEVIATIONPOSNEG = 0.1
    f = open(diroutput + pdb + '.rr', 'w')
    f.write(sequences[pdb] + '\n')
    for pair in sorted(p_dict.items(), key=lambda x: x[1]):
        (k, v) = pair
        (i, j) = k
        f.write("%d %d %.2f %.2f %.5f\n" % (i, j, p_dict[k] - DEVIATIONPOSNEG , p_dict[k] + DEVIATIONPOSNEG, 1 / p_dict[k]))
        x -= 1
        if x == 0:
            break
    f.close()

################################################################################
