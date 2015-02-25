import numpy as np
import tarfile
import os
import fnmatch
import matplotlib.pyplot as plt
from scipy.linalg import svd


def unpickle(f):
    import cPickle
    fo = open(f, 'rb')
    d = cPickle.load(fo)
    fo.close()
    return d

try:
    import urllib
    urllib.urlretrieve('http://google.com')
except AttributeError:
    import urllib.request as urllib
url = 'http://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
data_file = '../cifar-10-python.tar.gz'
if not os.path.exists(data_file):
    print("Downloading cifar10")
    urllib.urlretrieve(url, data_file)
    tar = tarfile.open(data_file)
    os.chdir('..')
    tar.extractall()
    tar.close()
    print("Download complete")

data_dir = '../cifar-10-batches-py/'
train_files = []
for filepath in fnmatch.filter(os.listdir(data_dir), 'data*'):
    train_files.append(os.path.join(data_dir, filepath))

name2label = {k:v for v,k in enumerate(
    unpickle(os.path.join(data_dir, 'batches.meta'))['label_names'])}
label2name = {v:k for k,v in name2label.items()}

train_files = sorted(train_files, key=lambda x: x.split("_")[-1])
train_x = []
train_y = []
for f in train_files:
    d = unpickle(f)
    t = d['data'].reshape(d['data'].shape[0], 3, 32 * 32)
    t = t.transpose(0, 2, 1)
    train_x.append(t)
    train_y.append(d['labels'])

train_x = np.array(train_x)
train_y = np.array(train_y)
train_x -= train_x.mean(axis=1, keepdims=True)
train_x /= train_x.std(axis=1, keepdims=True)

random_state = np.random.RandomState(1999)
n_classes = 10
X = train_x
y = train_y
minibatch_size = 1000
idx = np.arange(0, X.shape[0], minibatch_size)
mb_indices = zip(idx[:-1], idx[1:])
W = 0.1 * random_state.randn(n_classes, X.shape[1])
b = np.zeros(W.shape[0],)
onehot = np.zeros((W.shape[0], X.shape[0]))
onehot[y, np.arange(len(y))] = 1.

lr = 0.001
r = 0.1

epochs = 100

def forward(X, W, b):
    inner = np.dot(X, W.T) + b
    xmax = inner.max(axis=-1)
    normalization = xmax + np.log(np.sum(np.exp(inner - xmax[..., None]),
                                         axis=-1))
    probs = np.array([np.exp(np.dot(X, W[k]) + b[k] - normalization)
                             for k in range(W.shape[0])])
    return probs

def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=0)

for e in range(epochs):
    print("Starting epoch %e" % e)
    y_pred = predict(X, W, b)
    print(np.mean(y == y_pred))
    for i, j in mb_indices:
        X_n = X[i:j]
        y_n = y[i:j]
        probs = forward(X_n, W, b)
        class_probs = probs[y_n, np.arange(len(y_n))]
        l2_reg = r * np.sum(W ** 2)
        cost = -np.mean(np.log(class_probs)) + r *  l2_reg
        grad_cost_W = np.dot(probs - onehot, X) + r * np.sum(W)
        grad_cost_b = np.sum(probs - onehot, axis=1)
        W -= lr * grad_cost_W
        b -= lr * grad_cost_b

"""
U, S, V = svd(train_x)

f1, pcaxarr = plt.subplots(10, 10)
plt.suptitle("Color patch PCA basis functions")
f2, zcaxarr = plt.subplots(10, 10)
plt.suptitle("Color patch ZCA basis functions")
for n, b in enumerate(V[:100]):
    pcaxarr.flat[n].imshow(b.reshape((10, 10, 3)))
    pcaxarr.flat[n].axis('off')

Z = np.dot(V.T * np.sqrt(1.0 / ( S ** 2 / len(train_x) + .1)), V)
for n, b in enumerate(Z[:100]):
    zcaxarr.flat[n].imshow(b.reshape((10, 10, 3)))
    zcaxarr.flat[n].axis('off')

plt.show()
"""
