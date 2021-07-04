from cv2 import data
import numpy as np
import matplotlib.pyplot as plt
import cv2
import scipy.cluster
from sklearn.mixture import BayesianGaussianMixture


"""
    1. Read image
    2. Preprocess image
    3. Apply Gaussaian Mixture
    4. Choose top 5 
    5. Plot results
"""



def read_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

def preprocess(img, *params):
    
    img = cv2.resize(img, (128,128))
    array = np.array(img)
    shape = array.shape
    array = array.reshape(np.product(shape[:2]), shape[2]).astype(float)

    return array, shape

def gaussian_mixture(array, *params):
    bgm = BayesianGaussianMixture(
        n_components=10, 
        random_state=1, 
        weight_concentration_prior_type="dirichlet_process", 
        init_params="kmeans",
        covariance_type="full")

    bgm.fit(array)

    vecs, dist = scipy.cluster.vq.vq(array, bgm.means_)
    counts, bins = np.histogram(vecs, len(bgm.means_))
    
    return bgm.means_, bgm.weights_, counts, vecs

def plot_histogram(centroids, counts):
    new_center = centroids[np.where(counts>0)]
    new_count = counts[np.where(counts>0)]
    
    row = len(new_count)
    col = 1
    
    list1 = [x for x,_ in sorted(zip(new_center, new_count), key = lambda pair: pair[1], reverse=True)]
    list2 = sorted(new_count, reverse=True)
    
    print(list1)
    print(list2)
    
    fig, axs = plt.subplots(row,col)
    fig.subplots_adjust(wspace=0.05, hspace=0.05)
    axs = axs.flatten()
    
    for i in range(len(new_center)):
        
        percentage = 100*list2[i]/np.sum(list2)
        
        axs[i].set_axis_off()
        axs[i].text(0,0,f'{percentage:.1f}%  {np.array(list1[i], dtype=int)}',
          bbox={'facecolor':'white','alpha':1,'edgecolor':'none','pad':1},
          ha='center', va='center', fontsize=10)
        axs[i].imshow([[np.array(list1[i], dtype=int)]], aspect='auto')
        
    return fig
    
def plot_result(centroids, array, vecs, shape):
    # plt.clf()
    for idx, color in enumerate(centroids):
        fig, ax = plt.subplots(1,1)
        ax.set_axis_off()
    
        res = np.ones_like(array)
        res[scipy.r_[np.where(vecs==idx)],:] = color
        res = np.array(res.reshape(*shape), dtype=int) 
        ax.imshow(res)
        # plt.show()
        # plt.pause(0.5)


def main():
    # img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\test.png'
    img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\FLIR_09312.jpg'

    img = read_image(img_path)
    array, shape = preprocess(img)
    centroids, wieghts, hist_counts, vecs = gaussian_mixture(array) 
    hist_plot = plot_histogram(centroids, hist_counts)

    plt.show()

    plot_result(centroids, array, vecs, shape)
    plt.show()

def hashing():
    # img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\test.png'
    # img_path = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img\FLIR_09312.jpg'

    def dhash(image, hash_size=8):
        resized = cv2.resize(image, (hash_size+1,hash_size))
        
        diff = resized[:,1:] > resized[:,:-1]
        
        return sum([2 ** i for (i, v) in enumerate(diff.flatten()) if v])


    test_hash = 629407987240849968

    database = {}

    img_dir = r'C:\Users\ipx\Desktop\Personal\Python_GUI\data\img'

    import glob, os
    for img in glob.glob(os.path.join(img_dir, "*.jpg")):

        image = cv2.imread(img)

        if image is None:
            continue        
        
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        imageHash = dhash(image)
        
        path = database.get(imageHash, [])
        

        path.append(img)
        database[imageHash] = path

        print(path)

    from scipy.spatial import distance

    for k, v in database.items():
        
        dist = distance.hamming(test_hash, k)
        
        
        print(dist)
        
        print(k)

    
    # hash = dhash(img)
    
    # print(hash)
    
    
    # plt.imshow(img, cmap="gray")
    # plt.show()



if __name__ == "__main__":
    main()    
    # hashing()