import PySimpleGUI as sg
import os, cv2, io, sys
from PySimpleGUI.PySimpleGUI import Frame
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *


tmp = [[sg.Image(filename='')]]

def define_layout():
    sg.theme('DarkGrey9')
    plt.style.use('dark_background')
    
    threshold_radio = sg.Radio('Threshold', 'Radio', size=(10, 1), key='-THRESHOLD-')
    threshold_slid = sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 15), key='-thslid-')

    blur_radio = sg.Radio('Blur', 'Radio', size=(10, 1), key='-BLUR-')
    blur_slid = sg.Slider((0, 11), 1, 1, orientation='h', size=(30, 15), key='-BLUR SLIDER-')

    hue_radio = sg.Radio('Hue', 'Radio', size=(10, 1), key='-HUE-')
    hue_slid = sg.Slider((0, 225), 0, 1, orientation='h', size=(30, 15), key='-HUE SLIDER-')

    ehance_radio =  sg.Radio('Enhance', 'Radio', size=(10, 1), key='-ENHANCE-')
    ehance_slid = sg.Slider((1, 225), 0, 1, orientation='h', size=(30, 15), key='-ENHANCE SLIDER-')

    canny_radio = sg.Radio('Canny', 'Radio', size=(10, 1), key='-CANNY-')
    canny_a = sg.Slider((0, 255), 128, 1, orientation='h', size=(15, 15), key='-CANNY SLIDER A-')
    canny_b = sg.Slider((0, 255), 128, 1, orientation='h', size=(15, 15), key='-CANNY SLIDER B-')

    denoise_radio = sg.Radio('Denoise', 'Radio', size=(10, 1), key='-DENOISE-')
    denoise_level = sg.Slider((1, 20), 1, 1, orientation='h', size=(30, 15), key='-DENOISE LEVEL-')
    
    setting_layout =[   
                    [threshold_radio, threshold_slid],
                    [hue_radio, hue_slid],
                    [blur_radio, blur_slid],
                    [denoise_radio, denoise_level],
                    [ehance_radio, ehance_slid],
                    [canny_radio, canny_a, canny_b]
                    ]
    
    menu_def = [['&Application', ['E&xit']],
                ['&About', ['&About']] ]

    original_image = [[sg.Image(filename='', key='-orginal_img-')]]
    modify_image = [[sg.Image(filename='', key='-modify_img-')]]

    treedata = get_tree_data("", os.getcwd()+"/../data")

    tree_layout=[[sg.Tree(
        data=treedata,
        headings=[],
        auto_size_columns=True,
        num_rows=20,
        col0_width=26,
        key="-TREE-",
        show_expanded=False,
        enable_events=True)]]

    read_layout = [[sg.Text("Folder: "), sg.InputText(key='-browse_folder-', enable_events=True, ),],
                    [sg.Text("Output:"), sg.InputText(key='-out_path-', enable_events=True,)],
                    [sg.Cancel(), sg.Button('Save',key='-save-',button_color=('black', '#4adcd6')), sg.ProgressBar(100, orientation='h', size=(20, 20),bar_color=("purple", "green"), key='-PROGRESS BAR-'), sg.Text("         ",key='-saved-')],
                    [sg.Image(filename='', key="-mod_img-")]
                    ]
    
    col_1 = sg.Column([
        [sg.FolderBrowse('Browse', key='-file-', target="-browse_folder-")],
        # [sg.Frame(title="Browser", layout=tree_layout)],
        [sg.Tree(data=treedata, headings=[], auto_size_columns=True, num_rows=20, col0_width=26, key="-TREE-", show_expanded=False, enable_events=True)],
        [sg.Frame(title="Original", layout=original_image)]
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        [sg.Frame("INPUTS", read_layout)],
        [sg.Frame("Modified", modify_image)],
        [sg.Multiline(size=(80,1), disabled=True, font='courier 8', key='-ML-')]
        ], vertical_alignment='top')
    
    
    page_1_layout = [[col_1, col_2]]
    
    layout = [[sg.Menu(menu_def, key='-MENU-')],
                [sg.Text('', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    
    layout +=[[sg.TabGroup([[   sg.Tab('Image_Processing', page_1_layout),
                                # sg.Tab('Plots', graph_layout)
                                ]], key='-TAB GROUP-')]]

    return layout



def main():    
    layout = define_layout()
    
    window = sg.Window(
        'Image_Processing_GUI', 
        layout,
        location=(10, 10),
        alpha_channel=1.0,
        no_titlebar=False,
        grab_anywhere=False, 
        element_justification="left").Finalize()
    
    def_img = cv2.imread("../data/img/default.png")
    # def_img = imutils.resize(def_img, width=400)
    def_img = cv2.resize(def_img, (640,480))
    def_bytes = cv2.imencode('.png', def_img)[1].tobytes()
    window["-modify_img-"].update(data=def_bytes)
    
    while True:
        
        now = datetime.now().strftime("%y-%m-%d-%H-%M-%S")

        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        elif event == "-browse_folder-":
            window["-TREE-"].update(values=get_tree_data("", values["-browse_folder-"]))

        elif event == "-TREE-":
            img_path = Path(values['-TREE-'][0])
            _, file_ext = os.path.splitext(img_path) 
            
            if file_ext != "":
                try:
                    src = cv2.imread(str(img_path))
                    src = imutils.resize(src, width=235)
                except Exception as ex:
                    
                    window['-ML-'].print(f'{str(type(ex).__name__)}', background_color='red',text_color='white', end='')
                    window['-ML-'].print(f' {ex.args[0]}.', end='')
                    window['-ML-'].print('(expected: [.jpg .jpeg .png])')
                else:
                    img_bytes = cv2.imencode('.png', src)[1].tobytes()
                    window["-orginal_img-"].update(data=img_bytes)
            
            # if file_ext in [".jpg", ".png", ".jpeg"]:
            #     src = cv2.imread(str(img_path))
            #     src = imutils.resize(src, width=235)
            #     img_bytes = cv2.imencode('.png', src)[1].tobytes()
            
            #     window["-orginal_img-"].update(data=img_bytes)
            #     window['-ML-'].print('[LOG]', background_color='green',text_color='white', end='')
            #     window['-ML-'].print(f"[{now}]")
            # elif file_ext not in ["" , ".jpg", ".png", ".jpeg"]:
            #     window['-ML-'].print('[LOG]', background_color='red',text_color='white', end='')
            #     window['-ML-'].print(f"[{now}]")
                
            # print(file_ext)
        
if __name__ == '__main__':  
    main()