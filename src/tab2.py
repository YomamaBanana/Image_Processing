import PySimpleGUI as sg
import os, cv2
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageColor
import time

import scipy
import scipy.cluster

plt.style.use("ggplot")

def elbow_plot(image, max_clusters=20):
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
    
    plt.show()
    




def tab2_layout():
    
    image_layout = [[
        sg.Image(filename="", key="image", enable_events=True)
    ]]
    
        
    col_1 = sg.Column([
        [sg.Frame("Testing", image_layout)],
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        [sg.Button('Analyze', k="test")],
        [sg.Image(filename="", key ="elbow")]
    ], vertical_alignment='top')
    
    
    layout = [[col_1, col_2]]
    
    
    
    
    return layout



def main():


    img_path = r'D:\Python\git\Python_GUI\data\img\FLIR_09312.jpg'
    
    img = cv2.imread(img_path)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = imutils.resize(img, width=640)
    
        
    

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
    
    while True:
        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        if event == "test":
            window["image"].update(data=cv2.imencode('.png', img)[1].tobytes())
            # elbow_plot(img)
        
        if event == "image":

            # print(event, values)

            print("Testing")

        # print(window["image"])



if __name__ == "__main__":
    main()
    