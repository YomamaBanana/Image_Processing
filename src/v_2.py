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
    
    threshold_radio = sg.Radio('Threshold', 'Radio', size=(8, 1), key='-THRESHOLD-', font=("Helvetica", 8))
    threshold_slid = sg.Slider((0, 255), 0, 1, orientation='h', size=(20, 7), key='-thslid-', enable_events=True, disable_number_display=True)
    threshold_value = sg.T('0', size=(4,1), key='-thes_value-')

    blur_radio = sg.Radio('Blur', 'Radio', size=(8, 1), key='-BLUR-', font=("Helvetica", 8))
    blur_slid = sg.Slider((0, 11), 0, 1, orientation='h', size=(20, 7), key='-BLUR SLIDER-',enable_events=True, disable_number_display=True)
    blur_value = sg.T('0', size=(4,1), key='-blur_value-')

    hue_radio = sg.Radio('Hue', 'Radio', size=(8, 1), key='-HUE-', font=("Helvetica", 8))
    hue_slid = sg.Slider((0, 225), 0, 1, orientation='h', size=(20, 7), key='-HUE SLIDER-',enable_events=True, disable_number_display=True)
    hue_value = sg.T('0', size=(4,1), key='-hue_value-')

    ehance_radio =  sg.Radio('Enhance', 'Radio', size=(8, 1), key='-ENHANCE-', font=("Helvetica", 8))
    ehance_slid = sg.Slider((0, 225), 0, 1, orientation='h', size=(20, 7), key='-ENHANCE SLIDER-',enable_events=True, disable_number_display=True)
    enhance_value = sg.T('0', size=(4,1), key='-enhance_value-')

    canny_radio = sg.Radio('Canny', 'Radio', size=(6, 1), key='-CANNY-', font=("Helvetica", 8))
    canny_a = sg.Slider((0, 255), 0, 1, orientation='h', size=(8, 7), key='-CANNY SLIDER A-',enable_events=True, disable_number_display=True)
    canny_b = sg.Slider((0, 255), 0, 1, orientation='h', size=(8, 7), key='-CANNY SLIDER B-',enable_events=True, disable_number_display=True)
    canny_a_value = sg.T('0', size=(3,1), key='-canny_a-')
    canny_b_value = sg.T('0', size=(4,1), key='-canny_b-')


    denoise_radio = sg.Radio('Denoise', 'Radio', size=(8, 1), key='-DENOISE-', font=("Helvetica", 8))
    denoise_level = sg.Slider((0, 20), 0, 1, orientation='h', size=(20, 7), key='-DENOISE LEVEL-',enable_events=True, disable_number_display=True)
    denoise_value = sg.T('0', size=(4,1), key='-denoise_value-')

    
    setting_layout =[   
                    [threshold_radio, threshold_slid, threshold_value],
                    [hue_radio, hue_slid, hue_value],
                    [blur_radio, blur_slid, blur_value],
                    [denoise_radio, denoise_level, denoise_value],
                    [ehance_radio, ehance_slid, enhance_value],
                    [canny_radio,canny_a,canny_a_value, canny_b, canny_b_value],
                    [sg.Text("COlOR SPACE:")],
                    [sg.Checkbox('RGB', default=True, k='-rgb-'), sg.Checkbox('HSV', default=False, k='-hsv-')]
                    ]

    menu_def = [['&Application', ['E&xit']],
                ['&About', ['&About']] ]

    original_image = [[sg.Image(filename='', key='-orginal_img-')]]
    modify_image = [[sg.Image(filename='', key='-modify_img-')]]

    treedata = get_tree_data("", os.getcwd()+"/../data")

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
        [sg.Frame("TEST", read_layout)],
        [sg.Frame("Modified", modify_image)],
        [sg.Multiline(size=(80,1), disabled=True, font='courier 8', key='-ML-')]
        ], vertical_alignment='top')
    
    col_3 = sg.Column([
        [sg.Frame("Settings", setting_layout)],
        # [sg.Frame("RGB_Channel", tmp)],
        # [sg.Frame("RGB_HIST", tmp)]
    ], vertical_alignment='top')

    page_1_layout = [[col_1, col_2, col_3]]
    
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
            
        window['-thes_value-'].update(int(values['-thslid-']))
        window['-blur_value-'].update(int(values['-BLUR SLIDER-']))
        window['-hue_value-'].update(int(values['-HUE SLIDER-']))
        window['-enhance_value-'].update(int(values['-ENHANCE SLIDER-']))
        window['-denoise_value-'].update(int(values['-DENOISE LEVEL-']))
        window['-canny_a-'].update(int(values['-CANNY SLIDER A-']))
        window['-canny_b-'].update(int(values['-CANNY SLIDER B-']))
            
        print(values["-rgb-"])
            
if __name__ == '__main__':  
    main()