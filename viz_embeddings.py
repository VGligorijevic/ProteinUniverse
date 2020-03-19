import sys
import pickle
import seaborn as sns

from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np


def load_names(fn="./data/cath-names.txt"):
    num2name = {}
    fRead = open(fn, 'r')
    for line in fRead:
        if not line.startswith("#"):
            splitted = line.strip().split()
            num = splitted[0]
            name = "_".join(splitted[2:])
            if name != ':':
                num2name[num] = name
    fRead.close()

    return num2name


def load_domains(fn="./data/cath-domain-list.txt"):
    domain2num = {}
    fRead = open(fn, 'r')
    for line in fRead:
        if not line.startswith("#"):
            splitted = line.strip().split()
            domain = splitted[0]
            c = splitted[1]
            a = splitted[2]
            t = splitted[3]
            domain2num[domain] = (c, ".".join([c, a]), ".".join([c, a, t]))
    fRead.close()

    return domain2num


def export_tsv(data, domain2num, num2name, fname_prefix):
    domains = data['domains']
    X = data['Feat']

    fW1 = open(fname_prefix + "_vectors.tsv", 'w')
    fW2 = open(fname_prefix + "_metadata.tsv", 'w')
    fW2.write("Class\tArchitecture\tTopology\n")
    for i, d in enumerate(domains):
        for j in range(X.shape[1]):
            fW1.write("%0.3f\t" % (X[i, j]))
        fW1.write('\n')
        clas = num2name[domain2num[d][0]]
        arch = num2name[domain2num[d][1]]
        topo = num2name[domain2num[d][2]]
        fW2.write("%s\t%s\t%s\n" % (clas, arch, topo))
    fW1.close()
    fW2.close()


if __name__ == "__main__":
    num2name = load_names()
    domain2num = load_domains()

    data = pickle.load(open(str(sys.argv[1]), 'rb'))

    # exprot files in a format suitalbe for Embedding Projector https://projector.tensorflow.org/
    export_tsv(data, domain2num, num2name, "./results/cath_embedding")

    domains = data['domains']
    X = data['Feat']

    print ("### Number of domains=", len(domains))
    print ("### Number of features=", X.shape[1])

    # labels (show only 3 most frequent folds)
    labels = [num2name[domain2num[d][-1]] for d in domains]
    unique_labels, counts_labels = np.unique(np.asarray(labels),
                                             return_counts=True)
    sorted_label_idx = np.argsort(counts_labels)[::-1][:3]

    indices = []
    for label in unique_labels[sorted_label_idx]:
        indices.append(np.where(np.asarray(labels) == label)[0])
    indices = np.concatenate(indices)

    # TSNE
    X = StandardScaler().fit_transform(X)
    X_tsne = TSNE(n_components=2, perplexity=20).fit_transform(X)

    # subset for most frequent labels
    X_tsne = X_tsne[indices]
    labels = np.asarray(labels)[indices]
    domains = np.asarray(domains)[indices]

    print ("### Number of presented domains = ", len(labels))

    # tSNE viz
    plt.figure()
    g = sns.scatterplot(x=X_tsne[:, 0], y=X_tsne[:, 1], hue=labels, s=40)
    plt.xlabel('tSNE-1', fontsize=16, fontweight="bold")
    plt.ylabel('tSNE-2', fontsize=16, fontweight="bold")
    plt.title('GAE', fontsize=16, fontweight="bold")
    plt.grid(linestyle=':', alpha=0.5)
    g.legend(loc='lower left', fontsize=8, ncol=1)
    plt.tight_layout()
    plt.legend(fontsize=14)
    plt.savefig('./results/gae_embedding.pdf', dpi=300)
    plt.show()
