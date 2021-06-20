import PySimpleGUI as sg
import os, cv2, io, sys
from PySimpleGUI.PySimpleGUI import Frame
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime

from utils import *

from color_chooser import main as testing

tmp = [[sg.Image(filename='')]]

def define_layout():
    sg.theme('DarkGrey9')
    plt.style.use('dark_background')
    
    none_radio = sg.Radio('None', 'Radio', size=(8, 1), key='-none-', font=("Helvetica", 8))
    
    threshold_radio = sg.Radio('Threshold', 'Radio', size=(8, 1), key='-THRESHOLD-', font=("Helvetica", 8))
    threshold_slid = sg.Slider((0, 255), 0, 1, orientation='h', size=(20, 7), key='-thslid-', enable_events=True, disable_number_display=True)
    threshold_value = sg.T('0', size=(4,1), key='-thes_value-')

    blur_radio = sg.Radio('Blur', 'Radio', size=(8, 1), key='-BLUR-', font=("Helvetica", 8))
    blur_slid = sg.Slider((1, 11), 1, 1, orientation='h', size=(20, 7), key='-BLUR SLIDER-',enable_events=True, disable_number_display=True)
    blur_value = sg.T('0', size=(4,1), key='-blur_value-')

    hue_radio = sg.Radio('Hue', 'Radio', size=(8, 1), key='-HUE-', font=("Helvetica", 8))
    hue_slid = sg.Slider((0, 225), 0, 1, orientation='h', size=(20, 7), key='-HUE SLIDER-',enable_events=True, disable_number_display=True)
    hue_value = sg.T('0', size=(4,1), key='-hue_value-')

    ehance_radio =  sg.Radio('Enhance', 'Radio', size=(8, 1), key='-ENHANCE-', font=("Helvetica", 8))
    ehance_slid = sg.Slider((1, 255), 1, 1, orientation='h', size=(20, 7), key='-ENHANCE SLIDER-',enable_events=True, disable_number_display=True)
    enhance_value = sg.T('0', size=(4,1), key='-enhance_value-')

    canny_radio = sg.Radio('Canny', 'Radio', size=(6, 1), key='-CANNY-', font=("Helvetica", 8))
    canny_a = sg.Slider((0, 255), 0, 1, orientation='h', size=(8, 7), key='-CANNY SLIDER A-',enable_events=True, disable_number_display=True)
    canny_b = sg.Slider((0, 255), 0, 1, orientation='h', size=(8, 7), key='-CANNY SLIDER B-',enable_events=True, disable_number_display=True)
    canny_a_value = sg.T('0', size=(3,1), key='-canny_a-')
    canny_b_value = sg.T('0', size=(4,1), key='-canny_b-')

    denoise_radio = sg.Radio('Denoise', 'Radio', size=(8, 1), key='-DENOISE-', font=("Helvetica", 8))
    denoise_level = sg.Slider((1, 20), 1, 1, orientation='h', size=(20, 7), key='-DENOISE LEVEL-',enable_events=True, disable_number_display=True)
    denoise_value = sg.T('0', size=(4,1), key='-denoise_value-')
    
    setting_layout =[
                    [none_radio],   
                    [threshold_radio, threshold_slid, threshold_value],
                    [hue_radio, hue_slid, hue_value],
                    [blur_radio, blur_slid, blur_value],
                    [denoise_radio, denoise_level, denoise_value],
                    [ehance_radio, ehance_slid, enhance_value],
                    [canny_radio,canny_a,canny_a_value, canny_b, canny_b_value],
                    [sg.Text("=====COlOR SPACE=====")],
                    [sg.Combo(values=["RGB","HSV"], size=(10,5),default_value="RGB",k='-color_space-'), sg.Button("COLOR", k="-color_chooser-")],
                    [sg.Image(filename="",k='-hist-')]
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
    
        
    def update_slider_values():
        window['-thes_value-'].update(int(values['-thslid-']))
        window['-blur_value-'].update(int(values['-BLUR SLIDER-']))
        window['-hue_value-'].update(int(values['-HUE SLIDER-']))
        window['-enhance_value-'].update(int(values['-ENHANCE SLIDER-']))
        window['-denoise_value-'].update(int(values['-DENOISE LEVEL-']))
        window['-canny_a-'].update(int(values['-CANNY SLIDER A-']))
        window['-canny_b-'].update(int(values['-CANNY SLIDER B-']))
        
    window = sg.Window(
        'Image_Processing_GUI', 
        layout,
        location=(10, 10),
        alpha_channel=1.0,
        no_titlebar=False,
        grab_anywhere=False, 
        element_justification="left").Finalize()
    
    def_img = cv2.imread("../data/img/default.png")
    def_img = cv2.resize(def_img, (640,480))
    def_bytes = cv2.imencode('.png', def_img)[1].tobytes()
    window["-modify_img-"].update(data=def_bytes)
    window["-browse_folder-"].update(value=os.getcwd()+"/../data")
    
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
                    src = imutils.resize(src, width=640)
                    src_copy = np.copy(src)
                    src_resize = imutils.resize(src, width=235)
                    window["-none-"].update(value=True)
                    window['-modify_img-'].update(data=cv2.imencode('.png', src_copy)[1].tobytes())
                except Exception as ex:
                    window['-ML-'].print(f'{str(type(ex).__name__)}', background_color='red',text_color='white', end='')
                    window['-ML-'].print(f' {ex.args[0]}.', end='')
                    window['-ML-'].print('(expected: [.jpg .jpeg .png])')
                else:
                    img_bytes = cv2.imencode('.png', src_resize)[1].tobytes()
                    window["-orginal_img-"].update(data=img_bytes)
        
        if values["-color_space-"] == "HSV" and 'src_copy' in locals():
            src_copy = cv2.cvtColor(src, cv2.COLOR_BGR2HSV)
            src_copy = np.copy(src_copy)
            histbytes = draw_hsv(src_copy)
            window["-hist-"].update(data=histbytes)
            
        elif values["-color_space-"] == "RGB" and 'src_copy' in locals():
            src_copy = np.copy(src)
            histbytes = draw_rgb(src_copy)
            window["-hist-"].update(data=histbytes)
        
        if "src_copy" in locals():
            if not values["-none-"]:
                if values['-THRESHOLD-']:
                    _, mod_img  = cv2.threshold(src_copy, int(values['-thslid-']), 255, cv2.THRESH_BINARY)
                elif values['-BLUR-'] :
                    mod_img  = cv2.GaussianBlur(src_copy, (21, 21), values['-BLUR SLIDER-'])
                elif values['-DENOISE-']:
                    mod_img = cv2.fastNlMeansDenoisingColored(src_copy, None, values["-DENOISE LEVEL-"],values["-DENOISE LEVEL-"],7,21)
                elif values['-HUE-'] :
                    mod_img = cv2.cvtColor(src_copy, cv2.COLOR_BGR2HSV)
                    mod_img[:, :, 0] += int(values['-HUE SLIDER-'])
                    mod_img = cv2.cvtColor(mod_img, cv2.COLOR_HSV2BGR)   
                elif values['-ENHANCE-']:
                    enh_val = values['-ENHANCE SLIDER-'] / 40
                    clahe = cv2.createCLAHE(clipLimit=enh_val, tileGridSize=(8, 8))
                    lab = cv2.cvtColor(src_copy, cv2.COLOR_BGR2LAB)
                    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
                    mod_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
                elif values['-CANNY-']:
                    mod_img = cv2.Canny(src_copy, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])

                if "mod_img" in locals():    
                    mod_imgbytes = cv2.imencode('.png', mod_img)[1].tobytes()
                    window['-modify_img-'].update(data=mod_imgbytes)

                    if values["-color_space-"] == "RGB" and not values['-CANNY-']:
                        histbytes = draw_rgb(mod_img)
                        window["-hist-"].update(data=histbytes)
                    if values["-color_space-"] == "HSV" and not values['-CANNY-']:
                        histbytes = draw_hsv(mod_img)
                        window["-hist-"].update(data=histbytes)
            else:
                mod_imgbytes = cv2.imencode('.png', src_copy)[1].tobytes()
                window['-modify_img-'].update(data=mod_imgbytes)

        if event == "-color_chooser-":
            # sg.Window("TEST", [[sg.Text("Hello Wordl")]], keep_on_top=True).read(timeout=0)
            test = testing()
            while True:
                a, b = test.read()
                print(a,b)
                if a == sg.WIN_CLOSED:
                    break
            test.close()
        update_slider_values()
        
        
            
if __name__ == '__main__':  
    main()