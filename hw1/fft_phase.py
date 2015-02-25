import numpy as np
import tarfile
import os
import fnmatch
import matplotlib.pyplot as plt
from numpy.fft import fftpack, fftshift


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
from IPython import embed; embed()
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
    train_x.append(np.dot(t, rgb2gray).squeeze())
    train_y.append(d['labels'])
train_x = np.array(train_x)
train_x = train_x.reshape((train_x.shape[0] * train_x.shape[1], -1))
train_y = np.array(train_y)
train_y = train_y.ravel()

f1, imaxarr = plt.subplots(5, 2)
f2, fftaxarr = plt.subplots(5, 2)

def get_abs(x):
    return np.abs(fftpack.fft2(
        x.reshape(32, 32)))

def get_angle(x):
    return np.angle(fftpack.fft2(
         x.reshape(32, 32)))

def calc_log_avg_fft(x):
    if len(x.shape) == 2:
        return np.log(fftshift(np.abs(fftpack.fft2(
            x.reshape(x.shape[0], 32, 32))).mean(axis=0)))
    else:
        return np.log(fftshift(np.abs(fftpack.fft2(
            x.reshape(32, 32)))))


for label in range(10):
    class_x = train_x[train_y == label]
    imaxarr.flat[label].matshow(class_x[0].reshape(32, 32), cmap="gray")
    imaxarr.flat[label].axis('off')
    imaxarr.flat[label].set_title(label2name[label])
    fftaxarr.flat[label].matshow(calc_log_avg_fft(class_x), cmap="gray")
    fftaxarr.flat[label].axis('off')
    fftaxarr.flat[label].set_title(label2name[label])

f1.tight_layout()
f2.tight_layout()

log_avg_fft = calc_log_avg_fft(train_x)
plt.matshow(log_avg_fft, cmap="gray")
plt.axis('off')
plt.title("Average amplitude spectrum")

f1, mixarr = plt.subplots(3, 2)
cat_x = train_x[train_y == name2label["cat"]][0]
auto_x = train_x[train_y == name2label["automobile"]][0]
mixarr.flat[0].matshow(
    cat_x.reshape(32, 32), cmap="gray")
mixarr.flat[0].axis('off')
mixarr.flat[0].set_title("cat")
mixarr.flat[1].matshow(
    auto_x.reshape(32, 32), cmap="gray")
mixarr.flat[1].axis('off')
mixarr.flat[1].set_title("auto")
mixarr.flat[2].matshow(
    calc_log_avg_fft(cat_x), cmap="gray")
mixarr.flat[2].axis('off')
mixarr.flat[2].set_title("cat")
mixarr.flat[3].matshow(
    calc_log_avg_fft(auto_x), cmap="gray")
mixarr.flat[3].axis('off')
mixarr.flat[3].set_title("automobile")

abs_cat = get_abs(cat_x)
angle_cat = get_angle(cat_x)
abs_auto = get_abs(auto_x)
angle_auto = get_angle(auto_x)

rec_cat = np.real(fftpack.ifft2(abs_cat * np.exp(1j * angle_auto)))
rec_auto = np.real(fftpack.ifft2(abs_auto * np.exp(1j * angle_cat)))

mixarr.flat[4].matshow(rec_cat, cmap="gray")
mixarr.flat[4].axis('off')
mixarr.flat[4].set_title("cat+automobile_reconstruction")

mixarr.flat[5].matshow(rec_auto, cmap="gray")
mixarr.flat[5].axis('off')
mixarr.flat[5].set_title("automobile+cat_reconstruction")

plt.show()
