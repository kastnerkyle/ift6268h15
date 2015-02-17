import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from numpy.lib.stride_tricks import as_strided
from numpy.fft import fftpack
import time

try:
    import urllib
    urllib.urlretrieve('http://google.com')
except AttributeError:
    import urllib.request as urllib
url = 'http://www.iro.umontreal.ca/~memisevr/teaching/ift6268_2015/nao_bw.jpg'
print('Downloading data from %s' % url)
urllib.urlretrieve(url, 'nao_bw.jpg')


def fft_convolve(im, filt):
    t1 = time.clock()
    shp_w = im.shape[0] + filt.shape[0]
    shp_h = im.shape[1] + filt.shape[1]
    f_c = fftpack.fft2(filt[::-1, ::-1], s=(shp_w, shp_h))
    f_im = fftpack.fft2(im, s=(shp_w, shp_h))
    res = np.real(fftpack.ifft2(f_c * f_im))
    b = (filt.shape[0] - 1) // 2
    res = res[b:-b - 1, b:-b - 1]
    t2 = time.clock()
    tt = t2 - t1
    print("Elapsed time fft_convolve %fs" % tt)
    return res, tt


def convolve(im, filt):
    t1 = time.clock()
    hpad = 2 * (filt.shape[0] // 2)
    vpad = 2 * (filt.shape[1] // 2)
    h = im.shape[0] + hpad
    v = im.shape[1] + vpad
    c = np.zeros((h, v))
    c[hpad // 2:-hpad // 2, vpad // 2:-vpad // 2] = im
    sz = c.itemsize
    shape = (filt.shape[0], filt.shape[1], im.shape[0],
             im.shape[1])
    strides = (sz, c.shape[0] * sz, c.shape[0] * sz, sz)
    fb = as_strided(c, shape=shape, strides=strides)
    # Convolving so have to flip x and y
    res = (filt[::-1, ::-1, None, None] * fb).sum(axis=0).sum(axis=0)
    t2 = time.clock()
    tt = t2 - t1
    print("Elapsed time convolve %fs" % tt)
    return res, tt


def gaussian_2d(shape, sigma):
    x, y = np.meshgrid(np.arange(-(shape[0] - 1) // 2, (shape[0] + 1) // 2),
                       np.arange(-(shape[1] - 1) // 2, (shape[1] + 1) // 2))
    g = np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    return g / g.sum()


random_state = np.random.RandomState(1999)
im = mpimg.imread('nao_bw.jpg')
filter_shapes = [
    (5, 5),
    (11, 11),
    (21, 21),
    (31, 31),
    (41, 41)]
filter_sizes = [s[0] for s in filter_shapes]
filters = []
reg_times = []
reg_res = []
fft_times = []
fft_res = []
for fshp in filter_shapes:
    filt = gaussian_2d(fshp, fshp[0] / 3.)
    r, t = convolve(im, filt)
    fft_r, fft_t = fft_convolve(im, filt)
    filters.append(filt)
    reg_times.append(t)
    reg_res.append(r)
    fft_times.append(fft_t)
    fft_res.append(fft_r)

plt.matshow(im, cmap="gray")
plt.axis('off')
plt.title('orig')
f1, imarr = plt.subplots(len(filter_sizes), 2)
f2, maskarr = plt.subplots(len(filters))
for i in range(len(filter_sizes)):
    imarr[i, 0].matshow(reg_res[i], cmap="gray")
    imarr[i, 0].axis('off')
    imarr[i, 0].set_title('time')
    imarr[i, 1].matshow(fft_res[i], cmap="gray")
    imarr[i, 1].axis('off')
    imarr[i, 1].set_title('fft')
    maskarr[i].matshow(filters[i], cmap="gray")
    maskarr[i].axis('off')
    maskarr[i].set_title('%ix%i' % (filter_sizes[i], filter_sizes[i]))

plt.figure()
plt.plot(filter_sizes, reg_times, color='r', label='time_conv')
plt.plot(filter_sizes, fft_times, color='b', label='fft_conv')
plt.legend()
plt.title("Convolution time in seconds")
plt.show()
