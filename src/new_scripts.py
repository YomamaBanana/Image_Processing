import pathlib
from matplotlib import colors
import numpy as np
import matplotlib.pyplot as plt
import cv2
import binascii
import struct
from PIL import Image
import numpy as np
import scipy
import scipy.misc
import scipy.cluster


plt.style.use("ggplot")

# img_path = r'D:\Python\git\Python_GUI\data\img\FLIR_09312.jpg'
img_path = r'D:\Python\git\Python_GUI\data\img\test.png'
# img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\FLIR_09312.jpg'
# img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\test.png'

NUM_CLUSTERS = 15

img = cv2.imread(img_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (128,128))

ar = np.asarray(img)
shape = ar.shape
ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)

print('finding clusters')
codes, dist = scipy.cluster.vq.kmeans(ar, NUM_CLUSTERS)
print('cluster centres:\n', codes)

vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
counts, bins = scipy.histogram(vecs, len(codes))    # count occurrences

index_max = scipy.argmax(counts)                    # find most frequent
peak = codes[index_max]
colour = binascii.hexlify(bytearray(int(c) for c in peak)).decode('ascii')
print('most frequent is %s (#%s)' % (peak, colour))

c = ar.copy()

for i, code in enumerate(codes):
    
    d = np.ones_like(c)    
    d[scipy.r_[scipy.where(vecs==i)],:] = code
    
    d = np.array(d.reshape(*shape), dtype=int)
    
    plt.imshow(d)
    plt.show()
    
# imageio.imwrite('clusters.png', c.reshape(*shape).astype(np.uint8))
print('saved clustered image')