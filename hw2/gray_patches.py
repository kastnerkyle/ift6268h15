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
rgb2gray = np.array([.299, .587, .144])[:, None]
for f in train_files:
    d = unpickle(f)
    t = d['data'].reshape(d['data'].shape[0], 3, 32 * 32)
    t = t.transpose(0, 2, 1)
    train_x.append(np.dot(t, rgb2gray).squeeze().astype('float32'))
    train_y.append(d['labels'])

train_x = np.array(train_x)
train_x = train_x.reshape(50000, 32, 32)
random_state = np.random.RandomState(1999)
image_idx = random_state.randint(0, 50000, 1000)
left_idx = random_state.randint(0, 32 - 10, 1000)
upper_idx = random_state.randint(0, 32 - 10, 1000)
# Have to iterate to avoid invalid slice
train_x = train_x[image_idx]
train_patches = []
for n, i in enumerate(train_x):
    train_patches.append(i[left_idx[n]:left_idx[n] + 10,
                           upper_idx[n]:upper_idx[n] + 10])
train_x = np.array(train_patches).reshape(len(train_patches), -1)
train_x -= train_x.mean(axis=1, keepdims=True)
train_x /= train_x.std(axis=1, keepdims=True)
U, S, V = svd(train_x)

f1, pcaxarr = plt.subplots(10, 10)
plt.suptitle("Gray patch PCA basis functions")
f2, zcaxarr = plt.subplots(10, 10)
plt.suptitle("Gray patch ZCA basis functions")
for n, b in enumerate(V):
    pcaxarr.flat[n].imshow(b.reshape((10, 10)), cmap="gray")
    pcaxarr.flat[n].axis('off')

Z = np.dot(V.T * np.sqrt(1.0 / ( S ** 2 / len(train_x) + .1)), V)
for n, b in enumerate(Z):
    zcaxarr.flat[n].imshow(b.reshape((10, 10)), cmap="gray")
    zcaxarr.flat[n].axis('off')

plt.show()
