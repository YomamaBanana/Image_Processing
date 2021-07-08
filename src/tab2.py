from tkinter.constants import S
import PySimpleGUI as sg
import cv2, io
import matplotlib.pyplot as plt

from utils import *

import scipy
import scipy.cluster
import numpy as np
from sklearn.mixture import BayesianGaussianMixture

def tab2_layout():
    sg.theme('DarkGrey9')
    
    top5_layout = [
        [sg.Image(filename="", key="show_color", enable_events=False)],
        [sg.Image(filename="", key="show_mask", enable_events=False)],
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
        [sg.Multiline(size=(33,12), disabled=True, background_color="black",font='courier 8', key='-t2_ML-')],
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        [sg.Frame("Plots", elbow_layout)],
    ], vertical_alignment='top')
    
    col_3 = sg.Column([
        [sg.Frame("TOP5", top5_layout)]
        ], vertical_alignment='top')
    
    layout = [[col_1, col_2, col_3]]
    

    return layout

