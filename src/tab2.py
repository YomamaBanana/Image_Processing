from tkinter.constants import S
import PySimpleGUI as sg
import cv2, io
import matplotlib.pyplot as plt

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
        [sg.Text("Number of Clusters"), sg.InputText("5", size=(4,1), k ="kmeans_num"), sg.Button("Run", k="apply_kmeans")]
        ])
    
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
    
    table_column = [
        "Color",
        "RGB",
        "%"
    ]
    
    image_layout = [
            [sg.Text("===== ELBOW PLOT =====")],
            [sg.Text("Max num of clusters: "), sg.InputText(default_text="20", size=(6,4), k="t2-max_clus", justification='right'), sg.Button('Plot', k="plot_elbow")],
            # [sg.Text("3rd degree approximation:")],
            # [poly_layout, roots_layout],
            [sg.Text("===== COLOR CLUSTERS =====")],
            [clustering_layout],
            # [],
            [sg.Image(filename="", key="t2-image", enable_events=True)],
            [sg.Table(values=[[100,100,100]], headings=table_column, max_col_width=10, num_rows=15, k="table",
                                          auto_size_columns=False,
                    display_row_numbers=True, col_widths =5)]
            ]
    
    elbow_layout = [
        [sg.Image(filename="", key ="elbow")],
        [sg.Image(filename="", key ="color")]
        ]
        
    col_1 = sg.Column([
        [sg.Frame("Main", image_layout)],        
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
    