import PySimpleGUI as sg
import os, cv2, io
import matplotlib.pyplot as plt
import scipy
import scipy.cluster
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb
from sklearn.mixture import BayesianGaussianMixture

from sklearn.cluster import KMeans

plt.style.use('dark_background')
plt.rcParams['lines.linewidth'] = 0.6
# plt.rcParams['ytick.left'] = False

def plot_image(image):
    # img = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(1,1,figsize=(6,4.5))
    ax.imshow(image)
    fig.tight_layout()
    
    item = io.BytesIO()
    fig.savefig(item, format='png') 
    plt.clf()
    plt.close('all')
    return item.getvalue()

def get_tree_data(parent, dirname):
    treedata = sg.TreeData()
    # https://github.com/PySimpleGUI/PySimpleGUI/blob/master/DemoPrograms/Demo_Tree_Element.py#L26
    def add_files_in_folder(parent, dirname):

        files = os.listdir(dirname)
        for f in files:
            fullname = os.path.join(dirname, f)
            if os.path.isdir(fullname):
                treedata.Insert(parent, fullname, f, values=[])#, icon=folder_icon)
                add_files_in_folder(fullname, fullname)
            else:

                treedata.Insert(parent, fullname, f, values=[
                                os.stat(fullname).st_size])#, icon=file_icon)

    add_files_in_folder(parent, dirname)

    return treedata

def draw_equal_hist(image):
    plt.style.use('dark_background')
    plt.clf()
    plt.figure(figsize=(2,1.7))
    plt.title("Eq'd_Histogram", fontsize=9)  
    plt.yticks([])
    plt.xticks(fontsize=8)
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_grayscale_image = cv2.equalizeHist(grayscale_image)
    histogram = cv2.calcHist([eq_grayscale_image], [0], None, [256], [0, 256])
    plt.plot(histogram, color='C5', lw=0.6)
    item = io.BytesIO()
    plt.xlim([0, 256])
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')
    return item.getvalue()

def draw_hist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eq_grayscale_image = cv2.equalizeHist(gray)
    
    fig, [ax2, ax1] = plt.subplots(2, 1, figsize=(2,3.5), sharex=True)
    # plt.title("Grayscale_Histogram", fontsize=9)  
    histogram = cv2.calcHist([gray], [0], None, [256], [0, 256])
    eq_histogram = cv2.calcHist([eq_grayscale_image], [0], None, [256], [0, 256])
    
    ax1.plot(histogram, color="C0", lw=0.6)
    ax1.set_title("GrayScale Hist")
    ax2.plot(eq_histogram, color="C8", lw=0.6)
    ax2.set_title("Eq'd Hist")
    
    fig.tight_layout()
    
    item = io.BytesIO()
    fig.savefig(item, format='png') 
    plt.clf()
    plt.close('all')
    return gray, eq_grayscale_image, item.getvalue()

def draw_hsv(img_f):
    plt.style.use('dark_background')
    plt.clf()
    plt.figure(figsize=(3,2))
    plt.yticks([])
    plt.xticks(fontsize=8)
    for i, channel in enumerate(("H", "S", "V")):
        histgram = cv2.calcHist([img_f], [i], None, [256], [0, 256])
        plt.plot(histgram, color = f"C{i+3}", label=channel)
        plt.xlim([0, 256])
    plt.legend()
    item = io.BytesIO()
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')
    return item.getvalue()

def draw_rgb(img_f):
    plt.style.use('dark_background')
    plt.clf()
    plt.figure(figsize=(3,2))
    plt.yticks([])
    plt.xticks(fontsize=8)
    for i, channel in enumerate(("r", "g", "b")):
            histgram = cv2.calcHist([img_f], [i], None, [256], [0, 256])
            plt.plot(histgram, color = channel, label = channel)
            plt.xlim([0, 256])
    plt.legend()
    item = io.BytesIO()
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')

    return item.getvalue()

def draw_spectrum(image):
    # plt.clf()
    from matplotlib.colors import LogNorm
    plt.figure(figsize=(3,3))
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    # fshift = np.fft.ifftshift(fshift)
    magnitude_spectrum = 20*(np.abs(fshift))
    plt.imshow(magnitude_spectrum, norm=LogNorm(vmin=5))
    plt.colorbar()
    item = io.BytesIO()
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')

    return item.getvalue()

def rgb2hsv(rgb):
    return rgb_to_hsv(rgb)

def hsv2rgb(hsv):
    return hsv_to_rgb(hsv)

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb

def elbow_plot(image, max_clusters=20):
    plt.close("all")
    plt.clf()
    plt.rcParams['ytick.labelleft'] = True
    
    plt.figure(figsize=(6,4))
    import numpy as np
    
    array = np.asarray(image)
    shape = array.shape
    
    array = array.reshape(np.product(shape[:2]), shape[2]).astype(float)
    
    distoration = []    
    
    for cluster in range(1, max_clusters):
        kmeans = KMeans(n_clusters=cluster)
        kmeans.fit(array)
        distoration.append(kmeans.inertia_)

    x = np.arange(1,max_clusters)
    y = np.array(distoration)
    
    plt.plot(x,y)
    plt.title("Elbow Graph")
    plt.ylabel("distance (euclidean )")
    plt.xlabel("clusters (n)")
    plt.xticks(range(1,max_clusters))
    
    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    plt.rcParams['ytick.labelleft'] = False
    
    return x, y, item.getvalue()    

def polyfit3d(x,y):
    z = np.polyfit(x,y,3) 
    
    order_2 = np.roots([6*z[0],2*z[1]])
    order_1 = np.roots([3*z[0], 2*z[1], z[2]])

    return z, order_1, order_2

def plot_top_colors(centroids, idx,array, vecs, shape, counts, indices):
    plt.style.use("dark_background")
    
    ratio = 100*counts[indices[idx]]/np.sum(counts)
        
    plt.clf()
    plt.figure(figsize=(3,3))
    plt.axis("off")
    plt.title(f"IDX {idx+1}: {ratio[0]:.1f}%", fontsize=10)
    
    res = np.zeros_like(array)
    res[scipy.r_[np.where(vecs==indices[idx])],:] = centroids
    res = np.array(res.reshape(*shape), dtype=int) 
    plt.imshow(res)

    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    
    return item.getvalue()

def plot_mask(centroids, idx,array, vecs, shape, counts, indices):
    plt.style.use("dark_background")
    
        
    plt.clf()
    plt.figure(figsize=(3,3))    
    plt.title("Mask", fontsize=10)
    
    plt.axis("off")
    res = np.zeros_like(array)
    res[scipy.r_[np.where(vecs==indices[idx])],:] = 255
    res = np.array(res.reshape(*shape), dtype=int) 
    plt.imshow(res)

    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    
    return item.getvalue()

def plot_color_histogram(centroids, counts):
    plt.style.use("dark_background")
    plt.clf()
    
    new_center = centroids[np.where(counts>0)]
    new_count = counts[np.where(counts>0)]
    
    row = len(new_count)
    col = 1
    
    list1 = [x for x,_ in sorted(zip(new_center, new_count), key = lambda pair: pair[1], reverse=True)]
    list2 = sorted(new_count, reverse=True)
    
    indices = []
    for i, (l1, l2) in enumerate(zip(list1, list2)):
        index = np.where(counts == l2)
        indices.append(index)
    
    
    fig, axs = plt.subplots(col,row, figsize=(6,3))
    fig.suptitle("Dominant Colors")
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    axs = axs.flatten()
    
    for i in range(len(new_center)):
        
        percentage = 100*list2[i]/np.sum(list2)
        
        axs[i].set_title(f'{percentage:.1f}%', fontsize=8)
        axs[i].set_xticks([])
        axs[i].set_yticks([])
        axs[i].set_xlabel(i+1)
        axs[i].imshow([[np.array(list1[i], dtype=int)]], aspect='auto')
    fig.tight_layout()
    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
        
    return list1, list2, indices, item.getvalue()

def gaussian_mixture(image, *params):
    
    img = cv2.resize(image, (150,150))

    ar = np.asarray(img)
    shape = ar.shape
    array = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    
    bgm = BayesianGaussianMixture(
        n_components=20, 
        random_state=1, 
        weight_concentration_prior_type="dirichlet_process", 
        init_params="kmeans",
        covariance_type="full")

    bgm.fit(array)

    vecs, dist = scipy.cluster.vq.vq(array, bgm.means_)
    counts, bins = np.histogram(vecs, len(bgm.means_))
    
    return bgm.means_, bgm.weights_, counts, vecs, shape, array

def k_means_clustering(image, num_clusters):
    
    img = cv2.resize(image, (150,150))

    ar = np.asarray(img)
    shape = ar.shape
    
    ar = ar.reshape(np.product(shape[:2]), shape[2]).astype(float)
    
    from sklearn.cluster import KMeans
    kmeans = KMeans(
        n_clusters=num_clusters).fit(ar)
    
    codes = kmeans.cluster_centers_

    vecs, dist = scipy.cluster.vq.vq(ar, codes)         # assign codes
    counts, bins = np.histogram(vecs, len(codes))    # count occurrences


    return codes, ar, vecs, shape, counts, kmeans

def empty_plot(w, h, text):
    plt.style.use("dark_background")
    plt.clf()
    plt.figure(figsize=(w,h))    
    plt.axis("off")
    plt.text(0.5,0,text,bbox={'facecolor':'black','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center')
    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    
    return item.getvalue()