from tkinter.constants import S
import PySimpleGUI as sg
import cv2, io
import matplotlib.pyplot as plt

import scipy
import scipy.cluster
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def elbow_plot(image, max_clusters=20):
    plt.close("all")
    plt.figure(figsize=(6,3))
    import pandas as pd
    import numpy as np
    
    image = cv2.resize(image, (128,128))
    array = np.asarray(image)
    shape = array.shape
    
    array = array.reshape(np.product(shape[:2]), shape[2]).astype(float)
    
    distoration = []    
    for cluster in range(1, max_clusters):
        _, dist = scipy.cluster.vq.kmeans(array, cluster)
        distoration.append(dist)

    x = np.arange(1,max_clusters)
    y = np.array(distoration)
    
    plt.plot(x,y)
    
    plt.ylabel("distortion")
    plt.xlabel("clusters (n)")
    plt.xticks(range(1,max_clusters))
    
    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    
    return x, y, item.getvalue()    

def polyfit3d(x,y):
    z = np.polyfit(x,y,3) 
    
    order_2 = np.roots([6*z[0],2*z[1]])
    order_1 = np.roots([3*z[0], 2*z[1], z[2]])

    return order_1, order_2

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
    
    
    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
        
    return list1, list2, item.getvalue()

def plot_top_colors(centroids, idx,array, vecs, shape):
    plt.style.use("dark_background")
    
    plt.clf()
    plt.figure(figsize=(2,2))
    plt.axis("off")
    res = np.ones_like(array)
    res[scipy.r_[np.where(vecs==idx)],:] = centroids
    res = np.array(res.reshape(*shape), dtype=int) 
    plt.imshow(res)

    item = io.BytesIO()
    plt.savefig(item, format="png")
    plt.close("all")
    
    return item.getvalue()

def tab2_layout():
    sg.theme('DarkGrey9')
    
    top5_layout = [[
        sg.Image(filename="", key="top1", enable_events=True),
        sg.Image(filename="", key="top2", enable_events=True),
        sg.Image(filename="", key="top3", enable_events=True),
        sg.Image(filename="", key="top4", enable_events=True),
        sg.Image(filename="", key="top5", enable_events=True),        
    ]]
    
    poly_layout = sg.Column([
            [sg.Text("ax^3: ",size=(5,1), justification='right'), sg.InputText("a",size=(5,4), background_color="gray",text_color="black", disabled=True, k="poly_a", justification='right')],
            [sg.Text("bx^2: ",size=(5,1), justification='right'), sg.InputText("b",size=(5,4), background_color="gray",text_color="black", disabled=True, k="poly_b", justification='right')],
            [sg.Text("cx: ",size=(5,1), justification='right'), sg.InputText("c",size=(5,4), background_color="gray",text_color="black", disabled=True, k="poly_c", justification='right')],
            [sg.Text("d: ",size=(5,1), justification='right'), sg.InputText("d",size=(5,4), background_color="gray",text_color="black", disabled=True, k="poly_d", justification='right')],       
    ], vertical_alignment='top')
    
    roots_layout = sg.Column([
        [sg.Text("",size=(8,1)),sg.Text("x",size=(3,1), justification='right')],
        [sg.Text("Extrema",size=(8,1), justification='right'), sg.InputText("a",size=(5,4), background_color="gray",text_color="black", disabled=True, k="roots_1", justification='right')],
        [sg.Text("",size=(8,1), justification='right'), sg.InputText("a",size=(5,4), background_color="gray",text_color="black", disabled=True, k="roots_2", justification='right')],
        [sg.Text("Points of Inflection",size=(8,2), justification='right'), sg.InputText("a",size=(5,4), background_color="gray",text_color="black", disabled=True, k="roots_3", justification='right')], 
    ], vertical_alignment='top')
    
    image_layout = [
            [sg.Text("===== ELBOW PLOT =====")],
            [sg.Text("Max num of clusters: "), sg.InputText(default_text="1", size=(6,4), k="t2-max_clus", justification='right'), sg.Button('Plot', k="plot_elbow")],
            [sg.Text("3rd degree approximation:")],
            [poly_layout, roots_layout],
            [],
            [sg.Image(filename="", key="t2-image", enable_events=True)]
            ]
    
    elbow_layout = [
        [sg.Image(filename="", key ="elbow")], [sg.Image(filename="", key ="color")]
        ]
        
    col_1 = sg.Column([
        [sg.Frame("Main", image_layout)],
        [sg.Frame("TOP5", top5_layout)]
        
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        # [sg.Button('Analyze', k="plot_elbow")],
        # [sg.Button('COLOR', k="plot_top5")],
        [sg.Frame("Plots", elbow_layout)]
    ], vertical_alignment='top')
    
    col_3 = sg.Column(
        [[sg.Button('COLOR', k="plot_top5")],],
        vertical_alignment='top')
    
    layout = [[col_1, col_2, col_3]]
    

    return layout

def main():

    img_path = r'..\data\img\FLIR_09312.jpg'
    
    img = cv2.imread(img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (128,128))
    img_bytes = cv2.imencode('.png', img)[1].tobytes()    

    array = np.array(img)
    shape = array.shape
    array = array.reshape(np.product(shape[:2]), shape[2]).astype(float)

    layout = tab2_layout()

    window = sg.Window(
        'Image_Processing_GUI', 
        layout,
        location=(10, 10),
        alpha_channel=1.0,
        no_titlebar=False,
        grab_anywhere=False,
         resizable=True, 
        element_justification="left").Finalize()    
    
    
    window["t2-image"].update(data=img_bytes)
    
    while True:
        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        if event == "plot_elbow":
            x,y,eblow = elbow_plot(img)
            window["elbow"].update(data=eblow)
            d1, d2 = polyfit3d(x,y)
            print(d1,d2)
        
        if event == "plot_top5":
            # for i in [f"top{x}" for x in range(1,6)]:
                # window[i].update(data=img_bytes)

            centroids, wieghts, hist_counts, vecs = gaussian_mixture(array)
            rgb, count, color_plot = plot_histogram(centroids, hist_counts)

            window["color"].update(data=color_plot)

            top_colors = []

            for color in range(1,6):
                fig = plot_top_colors(centroids[color], color, array, vecs, shape)
                window[f"top{color}"].update(data=fig)
            

        
        if event == "image":
            print("Testing")



if __name__ == "__main__":
    main()
    