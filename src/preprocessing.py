import numpy as np
import scipy.io as sio
import pickle


def _adj_rownorm(A):
    """
    Row-wise normalization of ajacency matrix.
    """
    np.fill_diagonal(A, 1.0)
    if (A.T == A).all():
        pass
    else:
        A = A + A.T
        print "### Matrix converted to symmetric."
    col = A.sum(axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        A = A.astype(np.float)/col[:, None]
        A[np.isnan(A)] = 0

    return A


def _adj_normalize(X):
    """
    Normalizing networks according to node degrees.
    """
    X = X - np.diag(np.diag(X))
    if (X.T == X).all():
        pass
    else:
        X = X + X.T
        print "### Matrix converted to symmetric."

    # normalizing the matrix
    deg = X.sum(axis=1).flatten()
    with np.errstate(divide='ignore', invalid='ignore'):
        deg = np.divide(1., np.sqrt(deg))
    deg[np.isinf(deg)] = 0
    D = np.diag(deg)
    X = D.dot(X.dot(D))

    return X


def net_preprocessing(A, max_steps=3):
    """Computing high-dimensional network embedding from high-order proximities
    computed from its adjacency matrix"""
    # computing high-order proximities
    A = _adj_rownorm(A)
    n = A.shape[0]
    tmp = np.eye(n, dtype=np.float)
    M = np.zeros((n, n), dtype=np.float)
    for i in range(0, max_steps):
        tmp = np.dot(tmp, A)
        M += tmp
    M = M - np.diag(np.diag(M))

    # computing PPMI matrix
    PPMI = computePPMI(M)

    return PPMI


def load_networks(fname):
    """Load string networks (with node ids already mapped: string->uniprot->id)"""
    A = {}
    f = open(fname, 'rb')
    rw1 = f.readline()
    rw2 = f.readline()
    N = int(rw1.split(":")[1])  # number of nodes
    net_names = rw2.split()[2:]  # net names
    for net_name in net_names:
        A[net_name] = np.zeros((N, N), dtype=float)
    for line in f:
        splitted = line.strip().split()
        i = int(splitted[0])
        j = int(splitted[1])
        w = splitted[2:]
        w = [float(wi) for wi in w]
        for ii in range(0, len(w)):
            if w[ii] > 0:
                A[net_names[ii]][j][i] = w[ii]
    f.close()

    return A


def load_th_annot(mat_fname):
    """Load th_annot *.mat file"""
    Data = sio.loadmat(mat_fname, squeeze_me=True)
    onts = ['MF', 'BP', 'CC']
    GO = {}
    GO['other_orgs_inds'] = Data['other_orgs_inds'] - 1
    for ont in onts:
        GO[ont] = {}
        goterms = Data['GO'][ont].tolist()['GOterms'].tolist()
        goterms = [str(goterm) for goterm in goterms]
        GO[ont]['goterms'] = goterms
        genes = Data['GO'][ont].tolist()['genes'].tolist()
        genes = [str(gene) for gene in genes]
        GO[ont]['genes'] = genes
        GO[ont]['y_train'] = Data['GO'][ont].tolist()['y_train'].tolist().todense()
        GO[ont]['y_test'] = Data['GO'][ont].tolist()['y_test'].tolist().todense()
        GO[ont]['y_valid'] = Data['GO'][ont].tolist()['y_valid'].tolist().todense()
        GO[ont]['train_idx'] = Data['GO'][ont].tolist()['train_idx'].tolist()
        GO[ont]['test_idx'] = Data['GO'][ont].tolist()['test_idx'].tolist()
        GO[ont]['valid_idx'] = Data['GO'][ont].tolist()['valid_idx'].tolist()

    return GO


def load_seq_features(mat_fname):
    """Load features from *.mat file"""
    names = ['ipr', 'key', 'kmer_full', 'sig']
    Data = sio.loadmat(mat_fname, squeeze_me=True)
    Features = {}
    for name in names:
        # Features['protFeat_' + name] = np.asarray(Data['protFeat_' + name].todense(), dtype=float)
        Features['protFeat_' + name] = np.asarray(Data['protFeat_' + name + '_norm'].todense(), dtype=float)
        # Features['regionFeat_' + name] = np.asarray(Data['regionFeat_' + name].todense(), dtype=float)
        Features['regionFeat_' + name] = np.asarray(Data['regionFeat_' + name + '_norm'].todense(), dtype=float)

    return Features


def load_net_features(pickle_fname):
    """Load string network features from pickle file"""
    Nets = pickle.load(open(pickle_fname, 'rb'))

    return Nets


def computePPMI(mat):
    """
     Compute the PPMI values for the raw co-occurrence matrix.
    """
    (nrows, ncols) = mat.shape
    colTotals = mat.sum(axis=0)
    rowTotals = mat.sum(axis=1)
    N = np.sum(rowTotals)
    rowMat = np.ones((nrows, ncols), dtype=np.float)
    for i in range(nrows):
        rowMat[i, :] = 0 if rowTotals[i] == 0 else rowMat[i, :] * (1.0 / rowTotals[i])
    colMat = np.ones((nrows, ncols), dtype=np.float)
    for j in range(ncols):
        colMat[:, j] = 0 if colTotals[j] == 0 else (1.0 / colTotals[j])
    P = N * np.multiply(mat, np.multiply(rowMat, colMat))
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.fmax(np.zeros((nrows, ncols), dtype=np.float), np.log2(P))

    return P


if __name__ == "__main__":
    # test 1
    # Features = load_seq_features('../data/seq_features/180411-motility_1_final.mat')
    # Annot = load_th_annot('../data/annot/180411-motility_1_th_annot.mat')
    # x = minmax_scale(computePPMI(Features['protFeat_key']))
    # perf, optimal_params = temporal_holdout(x, Annot['MF'], "mf_yeast_test.txt")
    # print Features['protFeat_kmer_full'].max()
    # print Features.keys()
    # print Annot['BP']['goterms'][12]
    # other_spec_ids =  Annot['other_orgs_inds']
    # spec_ids = [ii for ii in range(90142) if ii not in other_spec_ids]
    # print Annot['BP']['goterms'].index("GO:0001539")

    # test 2
    """
    print "### Loading string networks..."
    Nets = load_string_networks('../data/string/559292_string_networks.txt')
    for name in Nets:
        print "### Computing high-dim network embedding..."
        Nets[name] = net_preprocessing(Nets[name], max_steps=4, ppmi_type=None)
        # Nets[name] = newman_similarity(Nets[name])

    print
    print "### Writing output to file..."
    fWrite = open('../data/string/559292_string_networks_steps4_ppmi.pckl', 'wb')
    pickle.dump(Nets, fWrite)
    fWrite.close()
    """
    # test 3
    """
    A = np.array([[0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 1, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 1, 0, 0, 1, 1],
                  [0, 0, 0, 1, 0, 1],
                  [0, 0, 0, 1, 1, 0]])

    normA = net_preprocessing(A, ppmi_type=None)
    print A
    print normA
    """

    # test 4
    """
    print "### Loading networks..."
    Nets = load_networks('../data/string/9606_string_networks.txt')
    for name in Nets:
        print "### Computing high-dim network embedding..."
        Nets[name] = Nets[name]/float(Nets[name].max())
        Nets[name] = net_preprocessing(Nets[name], max_steps=3)
        print "### [%s] Number of missing genes: %d" % (name, len(np.where(Nets[name].sum(axis=1) == 0)[0]))

    print
    print "### Writing output to file..."
    fWrite = open('../data/string/9606_string_networks_steps3_ppmi.pckl', 'wb')
    pickle.dump(Nets, fWrite)
    fWrite.close()
    """
    # test 5
    """
    print "### Loading networks..."
    A = pickle.load(open('../data/gi_boone/559292_boone_gi_network.pckl', 'rb'))
    Nets = {}
    print "### Computing high-dim network embedding..."
    Nets['gi-boone'] = net_preprocessing(A, max_steps=3)
    print "### [%s] Number of missing genes: %d" % ('gi-boone', len(np.where(Nets['gi-boone'].sum(axis=1) == 0)[0]))

    print
    print "### Writing output to file..."
    fWrite = open('../data/gi_boone/559292_boone_gi_network_steps3_ppmi.pckl', 'wb')
    pickle.dump(Nets, fWrite)
    fWrite.close()
    """
