import numpy as np
import matplotlib.pyplot as plt
import sys

def dispims_color(M, border=0, bordercolor=[0.0, 0.0, 0.0], *imshow_args, **imshow_keyargs):
    """ Display an array of rgb images.

    The input array is assumed to have the shape numimages x numpixelsY x numpixelsX x 3
    """
    bordercolor = np.array(bordercolor)[None, None, :]
    numimages = len(M)
    M = M.copy()
    for i in range(M.shape[0]):
        M[i] -= M[i].flatten().min()
        M[i] /= M[i].flatten().max()
    height, width, three = M[0].shape
    assert three == 3
    n0 = np.int(np.ceil(np.sqrt(numimages)))
    n1 = np.int(np.ceil(np.sqrt(numimages)))
    im = np.array(bordercolor)*np.ones(
                             ((height+border)*n1+border,(width+border)*n0+border, 1),dtype='<f8')
    for i in range(n0):
        for j in range(n1):
            if i*n1+j < numimages:
                im[j*(height+border)+border:(j+1)*(height+border)+border,
                   i*(width+border)+border:(i+1)*(width+border)+border,:] = np.concatenate((
                  np.concatenate((M[i*n1+j,:,:,:],
                         bordercolor*np.ones((height,border,3),dtype=float)), 1),
                  bordercolor*np.ones((border,width+border,3),dtype=float)
                  ), 0)
    imshow_keyargs["interpolation"]="nearest"
    plt.imshow(im, *imshow_args, **imshow_keyargs)
    plt.show()

fname = sys.argv[1]
W = np.load(fname)
W = W.T
print(W.shape)
dispims_color(W.reshape(W.shape[0], 10, 10, 3))
