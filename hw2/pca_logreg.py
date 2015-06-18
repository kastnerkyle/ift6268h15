import numpy as np
import tarfile
import os
import fnmatch
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
    pushd = os.getcwd()
    os.chdir('..')
    tar.extractall()
    tar.close()
    os.chdir(pushd)
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
    train_x.append(t.ravel())
    train_y.append(d['labels'])

random_state = np.random.RandomState(1999)
train_x = np.array(train_x).reshape(50000, -1)
train_y = np.array(train_y).reshape(50000, -1)
idx = random_state.permutation(np.arange(train_x.shape[0]))
train_x = train_x[idx]
train_y = train_y[idx]

train_x -= train_x.mean(axis=1, keepdims=True)
train_x /= train_x.std(axis=1, keepdims=True)

X = train_x[:45000]
U, S, VT = svd(X)
components = VT[:800]
X = np.dot(X, components.T)
y = train_y[:45000]
vX = train_x[-5000:]
vX = np.dot(vX, components.T)
vy = train_y[-5000:]


n_classes = len(np.unique(y.ravel()))
minibatch_size = 100
lr = 0.01
r = 0.0002
epochs = 100

idx = np.arange(0, X.shape[0], minibatch_size)
mb_indices = zip(idx[:-1], idx[1:])
W = 0.01 * random_state.randn(X.shape[1], n_classes)
b = np.zeros(n_classes,)
onehot = np.zeros((X.shape[0], n_classes))
for n, yi in enumerate(y):
    onehot[n, yi] = 1.

def logsumexp(x, dim=-1):
    """Compute numerically stable log(sum(exp(x))).

       Use second argument to specify along which dimension to sum.
       If -1 (default), logsumexp is computed along the last dimension.
    """
    if len(x.shape) < 2:  #got only one dimension?
        xmax = x.max()
        return xmax + np.log(np.sum(np.exp(x - xmax)))
    else:
        if dim != -1:
            x = x.transpose(range(dim) + range(dim + 1, len(x.shape)) + [dim])
        lastdim = len(x.shape) - 1
        xmax = x.max(lastdim)
        return xmax + np.log(np.sum(np.exp(x-xmax[...,np.newaxis]), lastdim))

def softmax(X):
    return np.exp(X - logsumexp(X)[:, None])

def forward(X, W, b):
    return softmax(np.dot(X, W) + b)

def predict(X, W, b):
    probs = forward(X, W, b)
    return np.argmax(probs, axis=1)


for e in range(epochs):
    print("Starting epoch %i" % e)
    for i, j in mb_indices:
        X_n = X[i:j].astype('float32')
        y_n = y[i:j]
        onehot_n = onehot[i:j]
        probs = forward(X_n, W, b)
        grad_cost_W = np.dot(X_n.T, (probs - onehot_n)) + r * W
        grad_cost_b = np.sum(probs - onehot_n, axis=0)
        W = W - lr * grad_cost_W
        b = b - lr * grad_cost_b

    if (e % 10 == 0) or (e == epochs - 1):
        x_pred = softmax(np.dot(X, W) + b)
        x_class = x_pred[np.arange(len(y)), y]
        x_logprob = -np.mean(x_class)
        vx_pred = softmax(np.dot(vX, W) + b)
        vx_class = vx_pred[np.arange(len(vy)), vy]
        vx_logprob = -np.mean(vx_class)
        y_pred = np.argmax(x_pred, axis=1)
        vy_pred = np.argmax(vx_pred, axis=1)
        print("Training accuracy: %f" % np.mean(
            y.ravel().astype(int) == y_pred.astype(int)))
        print("Test accuracy: %f" % np.mean(
            vy.ravel().astype(int) == vy_pred.astype(int)))
        print("Training LL: %f" % x_logprob)
        print("Test LL: %f" % vx_logprob)
