import PySimpleGUI as sg
import os, cv2, io
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
import numpy as np
from scipy import fftpack

from utils import *

def define_layout():
    sg.theme('DarkGrey9')
    plt.style.use('dark_background')
    
    threshold_radio = sg.Radio('Threshold', 'Radio', size=(10, 1), key='-THRESHOLD-')
    threshold_slid = sg.Slider((0, 255), 128, 1, orientation='h', size=(30, 15), key='-thslid-')

    blur_radio = sg.Radio('Blur', 'Radio', size=(10, 1), key='-BLUR-')
    blur_slid = sg.Slider((0, 11), 1, 1, orientation='h', size=(30, 15), key='-BLUR SLIDER-')

    hue_radio = sg.Radio('Hue', 'Radio', size=(10, 1), key='-HUE-')
    hue_slid = sg.Slider((0, 225), 0, 1, orientation='h', size=(30, 15), key='-HUE SLIDER-')

    ehance_radio =  sg.Radio('Ehance', 'Radio', size=(10, 1), key='-ENHANCE-')
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

    read_layout = [[sg.Text("Folder: "), sg.InputText(key='-browse_folder-', enable_events=True, ),
                    sg.FolderBrowse('Browse', key='-file-', target="-browse_folder-"), sg.Button('Read', key='-read_folder-')],
                    [sg.Text("Output:"), sg.InputText(key='-out_path-', enable_events=True,)],
                    [sg.Cancel(), sg.Button('Save',key='-save-',button_color=('black', '#4adcd6')), sg.ProgressBar(100, orientation='h', size=(20, 20),bar_color=("purple", "green"), key='-PROGRESS BAR-'), sg.Text("         ",key='-saved-')]]

    original_image = [[sg.Image(filename='', key='-orginal_img-')]]
    modify_image = [[sg.Image(filename='', key='-modify_img-')]]

    layout_2 = [[sg.Frame("Read", read_layout)],
                [sg.Frame("Setting", setting_layout)],
                [sg.Frame(title="Original", layout=original_image),
                sg.Frame(title="Modify", layout=modify_image)],
                [sg.Output(size=(60,2), font='Courier 8')]
                ]

    # grey_image = 

    graph_layout = [[sg.T('Anything that you would use for asthetics is in this tab!')],
                [sg.Frame(title="Original", layout=[[sg.Image(filename='', key='-hist_img-')]]), sg.Frame(title="Modify", layout=[[sg.Image(filename='', key='-mod_hist_img-')]])],
                [sg.Frame(title="GrayScale", layout=[[sg.Image(filename='', key='-gray_img-')]]), sg.Frame(title="Spectrum", layout=[[sg.Image(filename='', key='-fft_plot-')]])]
    ]

    menu_def = [['&Application', ['E&xit']],
                ['&About', ['&About']] ]

    treedata = get_tree_data("", os.getcwd()+"/../data")

    tree_layout=[[sg.Tree(
        data=treedata,
        headings=[],
        auto_size_columns=True,
        num_rows=30,
        col0_width=30,
        key="-TREE-",
        show_expanded=False,
        enable_events=True)]]

    layout_1 = [[sg.Frame(title="", layout=tree_layout), sg.Frame(title=None, layout=layout_2)]]

    layout = [[sg.Menu(menu_def, key='-MENU-')],
                [sg.Text('', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)]]
    layout +=[[sg.TabGroup([[   sg.Tab('Image_Processing', layout_1),
                                sg.Tab('Plots', graph_layout)
                                ]], key='-TAB GROUP-')]]
    return layout

def main():
    READ_File = False
    
    layout = define_layout()
    
    window = sg.Window('Image_Processing_GUI', layout,
                   location=(10, 10),alpha_channel=1.0,
                   no_titlebar=False,grab_anywhere=False, element_justification="center").Finalize()
    
    while True:
        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        elif event == "-read_folder-":
            window["-TREE-"].update(values=get_tree_data("", values["-browse_folder-"]))

        elif event == "-TREE-":
            read_path = Path(values['-TREE-'][0])
            _, file_ext = os.path.splitext(read_path) 
            if file_ext in [".jpg", ".png"]:
                print(f"[-LOG-] Chosen: {read_path.stem} ({file_ext[1:]})")
                READ_File = True
                default_out = str(read_path).replace(str(read_path.stem),str(read_path.stem)+"_modified")
                window['-out_path-'].Update(value=default_out)
                
                read_img = cv2.imread(str(read_path))
                read_img = imutils.resize(read_img, width=400)
                imgbytes = cv2.imencode('.png', read_img)[1].tobytes()
                mod_img = read_img.copy()
                window['-orginal_img-'].update(data=imgbytes)
                window['-modify_img-'].update(data=imgbytes)
                
                histbytes = draw_hist(read_img)
                window['-hist_img-'].update(data=histbytes)
            elif os.path.isdir(read_path):
                print(f"[-LOG-] Chosen: {read_path.stem} (dir)")
            else:
                print("[Error] Invalid input.")

        elif event == 'About':
            print("[LOG] Clicked About!")
            sg.popup(
                "GUI for image processing",
                '========================================',
                'TITLE:\tA simply python-based image processing GUI.',
                '========================================',                
                'Author: Adrian Tam',
                'Version: 1.0',
                'Updated: 2021/06/12',
                '=====================================8===',                
                'Instruction:',                
                '========================================',                
                'Choose an image from the file browser and change parameters to modify the image.',                
                'Press "Save" to save the modified image.',                
                )

        if values['-TAB GROUP-'] == "Plots":
            try:             
                gray_img = cv2.imread(str(read_path), 0)
                gray_img = imutils.resize(gray_img, width=400)
                graybytes = cv2.imencode('.png', gray_img)[1].tobytes()
                window["-gray_img-"].update(data=graybytes)

                specbytes = draw_spectrum(gray_img)
                window['-fft_plot-'].update(data=specbytes)
            
            
                mod_hist = draw_hist(mod_img)
                window['-mod_hist_img-'].update(data=mod_hist)
            except:
                print("[ERROR] No image selected.")
                pass
        else:
            try:
                plt.close("all")
            except:
                pass
        
        if READ_File:    
            if event == '-save-':
                out_path = values['-out_path-']
                for i in range(100):
                    window['-PROGRESS BAR-'].UpdateBar(i + 1)
                cv2.imwrite(out_path, mod_img)
                window['-saved-'].update(value="Saved!")
                
            elif values['-THRESHOLD-'] :
                _, mod_img  = cv2.threshold(read_img, int(values['-thslid-']), 255, cv2.THRESH_BINARY)
                window['-thslid-'].Update(int(values['-thslid-']))

            elif values['-BLUR-'] :
                mod_img  = cv2.GaussianBlur(read_img, (21, 21), values['-BLUR SLIDER-'])
                window['-BLUR SLIDER-'].Update(int(values['-BLUR SLIDER-']))

            elif values['-DENOISE-']:
                mod_img = cv2.fastNlMeansDenoisingColored(read_img, None, values["-DENOISE LEVEL-"],values["-DENOISE LEVEL-"],7,21)

            elif values['-HUE-'] :
                mod_img = cv2.cvtColor(read_img, cv2.COLOR_BGR2HSV)
                mod_img[:, :, 0] += int(values['-HUE SLIDER-'])
                mod_img = cv2.cvtColor(mod_img, cv2.COLOR_HSV2BGR)            
                window['-HUE SLIDER-'].Update(int(values['-HUE SLIDER-']))

            elif values['-ENHANCE-']:
                enh_val = values['-ENHANCE SLIDER-'] / 40
                clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
                lab = cv2.cvtColor(read_img, cv2.COLOR_BGR2LAB)
                lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                mod_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                window['-HUE SLIDER-'].Update(int(values['-HUE SLIDER-']))

            elif values['-CANNY-']:
                mod_img = cv2.Canny(read_img, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])

            mod_imgbytes = cv2.imencode('.png', mod_img)[1].tobytes()
            window['-modify_img-'].update(data=mod_imgbytes)
           
    window.close()
    exit(0)

if __name__ == '__main__':
    main()