from tkinter.constants import S
import PySimpleGUI as sg
import cv2, io
import matplotlib.pyplot as plt

from utils import *

import scipy
import scipy.cluster
import numpy as np
from sklearn.mixture import BayesianGaussianMixture


def gaussian_mixture(array, *params):
    bgm = BayesianGaussianMixture(
        n_components=12, 
        random_state=1, 
        weight_concentration_prior_type="dirichlet_process", 
        init_params="kmeans",
        covariance_type="full")

    bgm.fit(array)

    vecs, dist = scipy.cluster.vq.vq(array, bgm.means_)
    counts, bins = np.histogram(vecs, len(bgm.means_))
    
    return bgm.means_, bgm.weights_, counts, vecs

def tab2_layout():
    sg.theme('DarkGrey9')
    
    top5_layout = [
        [sg.Image(filename="", key="top1", enable_events=True), sg.Image(filename="", key="top2", enable_events=True)],
        [sg.Image(filename="", key="top3", enable_events=True), sg.Image(filename="", key="top4", enable_events=True)],
        [sg.Image(filename="", key="top5", enable_events=True), sg.Image(filename="", key="top6", enable_events=True)],
        [sg.Image(filename="", key="top7", enable_events=True), sg.Image(filename="", key="top8", enable_events=True)],
    ]
    
    clustering_methods = [
        "k-means",
        "BayesianGaussianMixture"
    ]
    
    clustering_layout = sg.Column([
        [sg.Combo(clustering_methods, default_value=clustering_methods[0], k="clus_method")],
        [sg.Text("Number of Clusters: "), sg.InputText("5", size=(4,1), k ="kmeans_num"), sg.Button("Run", k="apply_kmeans")]
        ])
    
    table_column = [
        "IDX",
        "  R  ",
        "  G  ",
        "  B  ",
        "  %  "
    ]
    
    image_layout = [
            [sg.Text("===== ELBOW PLOT =====")],
            [sg.Text("Max num of clusters: "), sg.InputText(default_text="20", size=(4,4), k="t2-max_clus", justification='right'), sg.Button('Plot', k="plot_elbow")],
            # [sg.Text("3rd degree approximation:")],
            # [poly_layout, roots_layout],
            [sg.Text("===== COLOR CLUSTERS =====")],
            [clustering_layout],
            # [],
            [sg.Image(filename="", key="t2-image", enable_events=True)],
            [sg.Table(
                values=[[0,0,0,0,0]], 
                headings=table_column, 
                max_col_width=5, 
                num_rows=10, 
                k="table",
                auto_size_columns=True,
                display_row_numbers=False, 
                col_widths = 2,
                enable_events=True)]
            ]
    
    elbow_layout = [
        [sg.Image(filename="", key ="color", size=(10,10))],
        [sg.Image(filename="", key ="elbow")]
        ]
        
    col_1 = sg.Column([
        [sg.Frame("Main", image_layout)],        
        [sg.Multiline(size=(33,12), disabled=True, background_color="black",font='courier 8', key='-t2_ML-')]
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        [sg.Frame("Plots", elbow_layout)],
    ], vertical_alignment='top')
    
    col_3 = sg.Column([
        [sg.Button('COLOR', k="plot_top5")],
        [sg.Frame("TOP5", top5_layout)]
        ], vertical_alignment='top')
    
    layout = [[col_1, col_2, col_3]]
    

    return layout

def main():

    img_path = r'..\data\img\test.png'
    
    img = cv2.imread(img_path)
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
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
        
        
        if values["clus_method"] == "BayesianGaussianMixture":
            window["kmeans_num"].update(text_color="gray", disabled=True)
            clustering_methhod = 1
        else:
            window["kmeans_num"].update(text_color="white", disabled=False)
            clustering_methhod = 0
        
            
        if event == "image":
            print("Testing")


        elif event == "apply_kmeans":
            try:
                num_cluster = int(values["kmeans_num"])
            except:
                pass
            
                                
            
            if clustering_methhod == 0:
                centroids, array, vecs, shape, counts = k_means_clustering(img, num_cluster)    
            elif clustering_methhod == 1:
                centroids, wieghts, counts, vecs, shape, array = gaussian_mixture(img)
            
            rgb_list, counts_list, indices,color_plot = plot_color_histogram(centroids, counts)
            window["color"].update(data=color_plot)


            tmp = np.zeros((len(rgb_list),5), dtype=object)

            for idx, rgb in enumerate(rgb_list):
                tmp[idx,0] = int(idx+1)
                tmp[idx,1] = int(rgb[0])
                tmp[idx,2] = int(rgb[1])
                tmp[idx,3] = int(rgb[2])
                tmp[idx,4] = round((100*counts_list[idx]/np.sum(counts_list)),1)
                
            window["table"].update(values=tmp.tolist())


if __name__ == "__main__":
    main()
    