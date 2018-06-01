#!/usr/bin/env python

import sys
import time
import numpy as np
import scipy.io as sio

from keras.optimizers import SGD, Adagrad
from keras import regularizers
from keras.constraints import max_norm
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Flatten
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D, UpSampling1D
from keras.callbacks import EarlyStopping

from Bio import SeqIO
import obonet
import networkx as nx
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pickle

np.random.seed(0)

FILTER_LEN = 10
NB_FILTER = 500
NB_HIDDEN = 200
POOL_FACTOR = 2
DROPOUT_CNN = 0.25
BATCH_SIZE = 32
NB_EPOCH = 80
LR = 0.1


def build_model(seq_len, channel_num, output_dim, nb_filter=NB_FILTER, filter_len=FILTER_LEN, pool_len=POOL_FACTOR, dropout_cnn=DROPOUT_CNN, nb_hidden=NB_HIDDEN):
    input_layer = Input(name='input', shape=(seq_len, channel_num))

    # convolution layer
    conv_layer_1 = Conv1D(filters=nb_filter,
                          kernel_size=filter_len,
                          activation='relu',
                          name='conv_layer_1')(input_layer)

    pooling_layer_1 = MaxPooling1D(pool_size=pool_len,
                                   strides=pool_len)(conv_layer_1)

    dropout_layer_1 = Dropout(dropout_cnn)(pooling_layer_1)
    conv_layer_2 = Conv1D(filters=int(2*nb_filter),
                          kernel_size=int(filter_len/2),
                          activation='relu',
                          name='conv_layer_2')(dropout_layer_1)

    pooling_layer_2 = MaxPooling1D(pool_size=pool_len,
                                   strides=pool_len)(conv_layer_2)

    dropout_layer_2 = Dropout(2*dropout_cnn)(pooling_layer_2)

    flat_layer = Flatten()(dropout_layer_2)
    # flat_layer = GlobalMaxPooling1D()(dropout_layer_2)

    hidden_layer = Dense(nb_hidden,
                         activation='sigmoid')(flat_layer)

    hidden_layer = Dropout(dropout_cnn)(hidden_layer)

    # Multi-task learning
    """
    hidden_layers = []
    for ii in range(0, output_dim):
        hidden_layers.append(Dense(50, activation='sigmoid')(hidden_layer))

    output_layers = []
    for ii in range(0, output_dim):
        output_layers.append(Dense(1, activation='sigmoid',
                                   kernel_regularizer=regularizers.l2(0.01))(hidden_layers[ii]))
    """
    output_layer = Dense(output_dim, activation='sigmoid', name='output_layer')(hidden_layer)

    # Multi-task learning
    # model = Model(inputs=input_layer, outputs=output_layers)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def build_deep_model(seq_len, channel_num, output_dim, depth=2, nb_filter=200, filter_len=8, nb_hidden=NB_HIDDEN):
    input_layer = Input(name='input', shape=(seq_len, channel_num))

    # convolution layer
    current_layer = input_layer
    for ii in range(0, depth):
        conv_layer = Conv1D(filters=nb_filter,
                            kernel_size=filter_len,
                            activation='relu',
                            name='conv_layer_' + str(ii+1))(current_layer)
        pooling_layer = MaxPooling1D(pool_size=2)(conv_layer)
        dropout_layer = Dropout(0.25)(pooling_layer)
        current_layer = dropout_layer

    # flat_layer = Flatten()(current_layer)
    flat_layer = GlobalMaxPooling1D()(current_layer)

    # hidden_layer = Dense(nb_hidden,
    #                     activation='sigmoid')(flat_layer)

    hidden_layer = Dropout(0.3)(flat_layer)
    output_layer = Dense(output_dim, activation='sigmoid', name='output_layer')(hidden_layer)

    model = Model(inputs=input_layer, outputs=output_layer)

    return model


def build_conv_ae(seq_len, channel_num, nb_filter=200, filter_len=8):
    input_layer = Input(name='input', shape=(seq_len, channel_num))  # adapt this if using `channels_first` image data format

    x = Conv1D(filters=nb_filter, kernel_size=filter_len, activation='relu')(input_layer)
    x = MaxPooling1D(pool_size=2)(x)
    x = Conv1D(filters=nb_filter, kernel_size=filter_len, activation='relu')(x)
    encoded = MaxPooling1D(pool_size=2, name='encoded')(x)

    # at this point the representation is ( 8) i.e. 128-dimensional

    x = Conv1D(filters=nb_filter, kernel_size=filter_len, activation='relu')(encoded)
    x = UpSampling1D(size=2)(x)
    x = Conv1D(filters=nb_filter, kernel_size=filter_len, activation='relu')(x)
    x = UpSampling1D(size=2)(x)
    decoded = Conv1D(filters=nb_filter, kernel_size=filter_len, activation='sigmoid', name='decoded')(x)

    model = Model(inputs=input_layer, outputs=decoded)

    return model


def convert_labels(y):
    if isinstance(y, list):
        new_y = np.zeros((len(y[0]), len(y)))
        for ii in range(0, len(y)):
            new_y[:, ii] = y[ii].reshape(-1)
    else:
        new_y = []
        for ii in range(0, y.shape[1]):
            new_y.append(y[:, ii])
    return new_y


def get_go_net(test_func_indx, all_goterms, obo_fname):
    """Creates a subnetwork from GO hierarchy."""
    graph = obonet.read_obo(open(obo_fname, 'r'))
    superterms = []
    for go in all_goterms[test_func_indx]:
        superterms.append(go)
    for go in all_goterms[test_func_indx]:
        for go_ in nx.descendants(graph, go):
            if go_ not in superterms:
                superterms.append(go_)
    superterms.remove('GO:0003674')
    graph = graph.subgraph(superterms)
    A = nx.adjacency_matrix(graph, nodelist=superterms)
    return superterms, A


def macro_aupr(y_test, y_score):
    # Compute macro-averaged AUPR
    perf = 0.0
    n = 0
    for i in range(y_test.shape[1]):
        if sum(y_test[:, i]) > 1:
            perf += aupr(y_test[:, i], y_score[:, i])
            n += 1
    perf /= n
    return perf


def load_annotations(mat_fname):
    Data = sio.loadmat(mat_fname, squeeze_me=True)
    Data['prot_IDs'] = np.array([str(prot) for prot in Data['prot_IDs']])
    Data['GO_IDs'] = np.array([str(goterm) for goterm in Data['GO_IDs']])
    return Data


def get_seq_vecs(sequences, char_indices):
    print('### Generating seq vectors...')
    one_hot_seqs = np.zeros((sequences.shape[0], sequences.shape[1], len(char_indices)))
    for i in range(0, len(sequences)):
        for j in range(0, len(sequences[i])):
            if sequences[i][j] in char_indices:
                one_hot_seqs[i][j][char_indices[sequences[i][j]]] = 1
    return one_hot_seqs


def process_sequences(entries, maxlen):
    sequences = []
    for entry_idx in range(0, len(entries)):
        entry = entries[entry_idx]
        entry_chars = list(entry)
        entry_chars = [char for char in entry_chars if char != '\n']
        sequences.append(entry_chars)

    nb_samples = len(sequences)
    x = np.zeros((nb_samples, maxlen), dtype=np.str)
    for idx, s in enumerate(sequences):
        trunc = np.asarray(s, dtype=np.str)
        if maxlen < len(trunc):
            x[idx] = trunc[:maxlen]
        elif(maxlen > len(trunc)):
            x[idx, 0:len(trunc)] = trunc
        else:
            x[idx] = trunc
    return x


def get_char_indices():
    # Amino-acid letters
    chars = ['A', 'R', 'N', 'D', 'B', 'C', 'Q', 'E', 'Z', 'G', 'H', 'I', 'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']
    char_indices = dict()
    for idx, char in enumerate(chars):
        char_indices[char] = idx
    return char_indices


def load_FASTA(filename):
    # Loads fasta file and returns a list of the Bio SeqIO records
    infile = open(filename, 'rU')
    entries = [str(entry.seq) for entry in SeqIO.parse(infile, 'fasta')]
    if(len(entries) == 0):
        return False
    return entries


def aupr(label, score):
    """Computing real AUPR"""
    label = label.flatten()
    score = score.flatten()

    order = np.argsort(score)[::-1]
    label = label[order]

    P = np.count_nonzero(label)
    # N = len(label) - P

    TP = np.cumsum(label, dtype=float)
    PP = np.arange(1, len(label)+1, dtype=float)  # python

    x = np.divide(TP, P)  # recall
    y = np.divide(TP, PP)  # precision

    pr = np.trapz(y, x)

    return pr


def main():
    sequences = load_FASTA('5.21.2018.all_reviewed_prots_experimental_protein_sequences.fasta')
    Annot = pickle.load(open('5.21.2018.all_reviewed_prots_experimental_MFannotation_data.pkl', 'rb'))
    seq_vec = process_sequences(sequences, maxlen=600)
    X = get_seq_vecs(seq_vec, get_char_indices())
    test_funcs = np.where(np.logical_and(np.logical_and(Annot['train_annots'].sum(axis=0) > 30,
                                                        Annot['train_annots'].sum(axis=0) <= 300),
                                         Annot['test_annots'].sum(axis=0) >= 20))[1]
    print (Annot['GO_IDs'][test_funcs])
    # goterms, A = get_go_net(test_funcs, Annot['GO_IDs'], 'go-basic.obo')
    # train_funcs = [np.where(np.isin(Annot['GO_IDs'], go_term))[0][0] for go_term in goterms]

    Y_tr = np.asarray(Annot['train_annots'][:, test_funcs])
    Y_te = np.asarray(Annot['test_annots'][:, test_funcs])
    Y_va = np.asarray(Annot['valid_annots'][:, test_funcs])
    train_ids = Annot['train_prot_inds']
    test_ids = Annot['test_prot_inds']
    valid_ids = Annot['valid_prot_inds']
    print ("Train=%d; Valid=%d; Test=%d" % (Y_tr.shape[0], Y_va.shape[0], Y_te.shape[0]))
    X_tr = X[train_ids]
    X_te = X[test_ids]
    X_va = X[valid_ids]

    X_tr = np.concatenate((X_tr, X_va), axis=0)
    Y_tr = np.concatenate((Y_tr, Y_va), axis=0)

    # save_name = sys.argv[1]
    # nb_filter = int(sys.argv[2])
    # filter_len = int(sys.argv[3])
    # nb_hidden = int(sys.argv[4])
    # dropout_cnn = float(sys.argv[5])

    save_name = "reviewed_cnn"

    print ('loading data...')
    sys.stdout.flush()

    __, seq_len, channel_num = X_tr.shape
    output_dim = Y_tr.shape[1]

    # model = build_model(seq_len, channel_num, output_dim)
    model = build_deep_model(seq_len, channel_num, output_dim)

    print ('model compiling...')
    sys.stdout.flush()

    adagrad = SGD(lr=LR, momentum=0.9)
    # adagrad = Adagrad()
    model.compile(optimizer=adagrad, loss='binary_crossentropy')
    print (model.summary())

    # save model
    outmodel = open(save_name + '.json', 'w')
    outmodel.write(model.to_json())
    outmodel.close()

    print ('training...')
    sys.stdout.flush()

    time_start = time.time()

    earlystopper = EarlyStopping(monitor='val_loss', patience=10, verbose=1)
    history = model.fit(X_tr, Y_tr, batch_size=BATCH_SIZE, epochs=NB_EPOCH,
                        shuffle=True, verbose=1, validation_split=0.2,
                        callbacks=[earlystopper])
    # history = model.fit(X_tr, convert_labels(Y_tr), batch_size=BATCH_SIZE, epochs=NB_EPOCH,
    #                    shuffle=True, verbose=1, validation_split=0.1,
    #                    callbacks=[earlystopper])

    # Export figure: loss vs epochs (history)
    plt.figure()
    plt.plot(history.history['loss'], '.-')
    plt.plot(history.history['val_loss'], '.-')
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper left')
    plt.savefig(save_name + '_loss.png', bbox_inches='tight')

    time_end = time.time()

    Y_va_hat = model.predict(X_va)
    Y_te_hat = model.predict(X_te)

    # Y_va_hat = convert_labels(model.predict(X_va))
    # Y_te_hat = convert_labels(model.predict(X_te))

    # print ('*'*100)
    # print ('### TEST SAMPLES:')
    # print ('*'*100)
    # print ("%s m-AUPR [valid]: %.4f" % (save_name, aupr(Y_va[:, 0:18], Y_va_hat[:, 0:18])))
    # print ("%s m-AUPR [test]: %.5f" % (save_name, aupr(Y_te[:, 0:18], Y_te_hat[:, 0:18])))

    # print ('*'*100)
    # print ("%s M-AUPR [valid]: %.4f" % (save_name, macro_aupr(Y_va[:, 0:18], Y_va_hat[:, 0:18])))
    # print ("%s M-AUPR [test]: %.5f" % (save_name, macro_aupr(Y_te[:, 0:18], Y_te_hat[:, 0:18])))
    # print ('*'*100)

    print ('\n')
    print ('### TRAINING SAMPLES:')
    print ('*'*100)
    print ("%s m-AUPR [valid]: %.4f" % (save_name, aupr(Y_va, Y_va_hat)))
    print ("%s m-AUPR [test]: %.5f" % (save_name, aupr(Y_te, Y_te_hat)))

    print ('*'*100)
    print ("%s M-AUPR [valid]: %.4f" % (save_name, macro_aupr(Y_va, Y_va_hat)))
    print ("%s M-AUPR [test]: %.5f" % (save_name, macro_aupr(Y_te, Y_te_hat)))

    print ('%s training time : %d sec' % (save_name, time_end-time_start))

if __name__ == '__main__':
    main()
