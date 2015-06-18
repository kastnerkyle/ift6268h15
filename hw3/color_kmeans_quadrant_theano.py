import numpy as np
import tarfile
import os
import fnmatch
import theano.tensor as T
import theano
from scipy.linalg import svd
import matplotlib.pyplot as plt


def minibatch_indices(X, minibatch_size, lb=None, ub=None):
    if lb is None:
        lb = 0
    if ub is None:
        ub = len(X)
    minibatch_indices = np.arange(lb, ub, minibatch_size)
    minibatch_indices = np.asarray(list(minibatch_indices) + [ub])
    start_indices = minibatch_indices[:-1]
    end_indices = minibatch_indices[1:]
    return zip(start_indices, end_indices)


def kmeans(X, W=None, n_clusters=10, n_epochs=10, learningrate=0.01,
           batchsize=100, random_state=None, verbose=True):
    """
Code modded from R. Memisevic.
Copyright (c) 2013, Roland Memisevic
All rights reserved.

memisevr[at]iro[dot]umontreal[dot]ca

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

    * Redistributions of source code must retain the above copyright notice,
this list of conditions and the following disclaimer.
    * Redistributions in binary form must reproduce the above copyright
notice, this list of conditions and the following disclaimer in the
documentation and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
"AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING
IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY
OF SUCH DAMAGE.
    """
    if random_state is None:
        random_state = np.random.RandomState(None)
    if W is None:
        W = 0.1 * random_state.randn(n_clusters, X.shape[1])

    X2 = (X ** 2).sum(1)[:, None]
    for epoch in range(n_epochs):
        for i in range(0, X.shape[0], batchsize):
            D = -2 * np.dot(W, X[i:i+batchsize, :].T) + (W ** 2).sum(1)[:, None]
            D = D + X2[i:i+batchsize].T
            S = (D == D.min(0)[None, :]).astype("float").T
            W += learningrate * (np.dot(S.T, X[i:i+batchsize, :]) -
                                 S.sum(0)[:, None] * W)
        if verbose:
            print("epoch", epoch, "of", n_epochs, " cost: ", D.min(0).sum())
    return W


def patchify(imgs, patch_shape=(10, 10), patch_stride=(1, 1)):
    """
    imgs is an array of (n_images, X, Y, color)
    e.g. CIFAR10 is (50000, 32, 32, 3)
    Modified from
    http://stackoverflow.com/questions/16774148/fast-way-to-slice-image-into-overlapping-patches-and-merge-patches-to-image
    Can test with CIFAR10 and
    assert np.all(imgs[0, :10, :10] == patches[0, 0, 0])
    # with 2, 2 patch_stride
    assert np.all(imgs[0, 20:30, 20:30] == patches[0, -1, -1])
    assert np.all(imgs[-1, :10, :10] == patches[-1, 0, 0])
    assert np.all(imgs[-1, 20:30, 20:30] == patches[-1, -1, -1])
    """
    imgs = np.ascontiguousarray(imgs)  # won't make a copy if not needed
    n, X, Y, c = imgs.shape
    x, y = patch_shape
    shape = (n, (X - x + 1) / patch_stride[0], (Y - y + 1) / patch_stride[1], x,
             y, c)
    # The right strides can be thought by:
    # 1) Thinking of `img` as a chunk of memory in C order
    # 2) Asking how many items through that chunk of memory are needed
    #    when indices i,j,k,l are incremented by one
    strides = imgs.itemsize * np.array([X * Y * c, patch_stride[0] * Y * c,
                                        patch_stride[1] * c, Y * c, c, 1])
    patches = np.lib.stride_tricks.as_strided(imgs, shape=shape,
                                              strides=strides)
    return patches


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

test_files = []
for filepath in fnmatch.filter(os.listdir(data_dir), 'test*'):
    test_files.append(os.path.join(data_dir, filepath))

name2label = {k: v for v, k in enumerate(
    unpickle(os.path.join(data_dir, 'batches.meta'))['label_names'])}
label2name = {v: k for k, v in name2label.items()}


print("loading data...")
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
train_x = train_x.reshape(50000, 32, 32, 3)
train_y = train_y.reshape(len(train_x)).astype('int32')

d = unpickle(test_files[0])
test_x = d['data'].reshape(d['data'].shape[0], 3, 32 * 32)
test_x = test_x.transpose(0, 2, 1)
test_x = test_x.reshape(10000, 32, 32, 3)
test_y = np.array(d['labels']).astype('int32')

n_classes = len(np.unique(train_y))
patch_x = patchify(train_x)


def preprocess(patch_x):
    print("normalizing...")
    random_state = np.random.RandomState(1999)
    n_patch = 5000
    idx = random_state.randint(0, len(patch_x), n_patch)
    i1 = random_state.randint(0, patch_x.shape[1], n_patch)
    i2 = random_state.randint(0, patch_x.shape[2], n_patch)
    train_x_subset = patch_x[idx, i1, i2]
    train_x_subset = train_x_subset.reshape(len(train_x_subset), -1)
    m = train_x_subset.mean(axis=0)
    train_x_subset -= m[None]
    s = train_x_subset.std(axis=0)
    s += 1E-3
    train_x_subset /= (s[None])
    print("computing zca...")
    # ZCA on subset
    U, S, V = svd(train_x_subset)
    Z = np.dot(V.T * np.sqrt(1.0 / (S ** 2 / len(train_x_subset) + .1)), V)
    """
    print("computing pca...")
    U, S, V = svd(train_x_subset)
    # Keep top 10% of components
    Z = V[:30].T
    """
    return Z, m, s

random_state = np.random.RandomState(1999)
Z, m, s = preprocess(patch_x)

print("computing kmeans...")
n_patch = 5000
idx = random_state.randint(0, len(patch_x), n_patch)
i1 = random_state.randint(0, patch_x.shape[1], n_patch)
i2 = random_state.randint(0, patch_x.shape[2], n_patch)
train_x_subset = patch_x[idx, i1, i2]
shp = train_x_subset.shape
train_x_subset = train_x_subset.reshape(shp[0], -1)
train_x_subset = (train_x_subset - m[None]) / s[None]
train_x_subset = np.dot(train_x_subset, Z)
W = kmeans(train_x_subset, n_epochs=150, n_clusters=100,
           random_state=random_state).T

epochs = 50
minibatch_size = 500
learning_rate = .1

# create logistic regression
X = T.tensor4()
y = T.ivector()
shp = patch_x[:minibatch_size].shape
tx = patch_x[:minibatch_size].reshape(shp[0], shp[1], shp[2], -1)
ty = train_y[:minibatch_size]
X.tag.test_value = tx
y.tag.test_value = ty
b1 = shp[1] // 2
b2 = shp[2] // 2

W_sym = theano.shared(W)
Z_sym = theano.shared(Z)
normed = (X - m[None]) / s[None]
activation = T.dot(T.dot(normed, Z_sym), W_sym)
# relu
activation = activation * (activation > 1E-6)
# Quadrant pooling
upper_l = activation[:, :b1, :b2].mean(axis=(1, 2))
upper_r = activation[:, b1:, :b2].mean(axis=(1, 2))
lower_l = activation[:, :b1, b2:].mean(axis=(1, 2))
lower_r = activation[:, b1:, b2:].mean(axis=(1, 2))
final_activation = T.concatenate([upper_l, upper_r, lower_l, lower_r], axis=1)
# Quadrants == * 4
sW = theano.shared(0.1 * (random_state.rand(4 * W.shape[1], n_classes) - 0.5))
sb = theano.shared(np.zeros(n_classes))
pre_s = T.dot(final_activation, sW) + sb
out = T.nnet.softmax(pre_s)
cost = -T.mean(T.log(out)[T.arange(y.shape[0]), y])
params = [sW, sb]
grads = T.grad(cost, params)
updates = [(param_i, param_i - learning_rate * grad_i)
           for param_i, grad_i in zip(params, grads)]
train_function = theano.function([X, y], cost, updates=updates)
predict_function = theano.function([X], out)

np.save("kmeans_W.npy", W)

test_patch = patchify(test_x)
train_patch = patch_x
for e in range(epochs):
    for n, (i, j) in enumerate(minibatch_indices(patch_x, minibatch_size)):
        shp = patch_x[i:j].shape
        img_patch = patch_x[i:j].reshape(shp[0], shp[1], shp[2], -1)
        img_labels = train_y[i:j]
        batch_cost = train_function(img_patch, img_labels)
        print("epoch %i, batch %i, cost %f" % (e, n, batch_cost))

    test_pred = []
    for n, (i, j) in enumerate(minibatch_indices(test_patch, minibatch_size)):
        shp = test_patch[i:j].shape
        img_patch = test_patch[i:j].reshape(shp[0], shp[1], shp[2], -1)
        pred_x = np.argmax(predict_function(img_patch), axis=1)
        test_pred.append(pred_x)
    test_pred = np.array(test_pred).ravel()
    print("Test error %f" % np.mean(test_pred == test_y))


# Final predictions
train_pred = []
for n, (i, j) in enumerate(minibatch_indices(train_patch, minibatch_size)):
    shp = train_patch[i:j].shape
    img_patch = train_patch[i:j].reshape(shp[0], shp[1], shp[2], -1)
    pred_x = np.argmax(predict_function(img_patch), axis=1)
    train_pred.append(pred_x)
train_pred = np.array(train_pred).ravel()
print("Train error %f" % np.mean(train_pred == train_y))

test_pred = []
for n, (i, j) in enumerate(minibatch_indices(test_patch, minibatch_size)):
    shp = test_patch[i:j].shape
    img_patch = test_patch[i:j].reshape(shp[0], shp[1], shp[2], -1)
    pred_x = np.argmax(predict_function(img_patch), axis=1)
    test_pred.append(pred_x)
test_pred = np.array(test_pred).ravel()
print("Test error %f" % np.mean(test_pred == test_y))
