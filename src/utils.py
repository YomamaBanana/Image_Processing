import PySimpleGUI as sg
import os, cv2, io
import matplotlib.pyplot as plt
# from scipy import fftpack
import numpy as np

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

def draw_hist(img_f):
    plt.style.use('dark_background')
    plt.clf()
    histgram = cv2.calcHist([img_f], [0], None, [256], [0, 256])
    plt.figure()
    plt.plot(histgram, c="yellowgreen")
    plt.xlim([0, 256])
    plt.title('Histogram')
    plt.xlabel("RGB pixel")
    plt.ylabel("Number of pixels")
    item = io.BytesIO()
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')

    return item.getvalue()

def draw_plot(img_f):
    plt.style.use('dark_background')
    plt.clf()
    plt.figure(figsize=(3,2))
        
    for i, channel in enumerate(("r", "g", "b")):
            histgram = cv2.calcHist([img_f], [i], None, [256], [0, 256])
            plt.plot(histgram, color = channel)
            plt.xlim([0, 256])

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