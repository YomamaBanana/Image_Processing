import PySimpleGUI as sg
import os, cv2, io
import matplotlib.pyplot as plt
from scipy import fftpack
import numpy as np
from matplotlib.colors import rgb_to_hsv, hsv_to_rgb


plt.style.use('dark_background')

# plt.clf()

# plt.yticks([])
# plt.xticks(fontsize=8)
# plt.xlim([0, 256])


plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['ytick.left'] = False

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
    # img_f = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
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