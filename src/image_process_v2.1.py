import PySimpleGUI as sg
import os, cv2
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageColor
import time

from utils import *

plt.style.use('dark_background')
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['axes.titlesize'] = 10

def define_layout():
    sg.theme('DarkGrey9')
    
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
    
    HSV_slider = sg.Column([
        [sg.T("H",size=(1,1), font=("Helvetica", 8)), sg.T("S",size=(1,1), font=("Helvetica", 8)), sg.T("V",size=(1,1), font=("Helvetica", 8))],
        [sg.Slider((0, 360), 0, 1, orientation='v', size=(4, 8), key='-H_value-',enable_events=True, disable_number_display=True),\
            sg.Slider((0, 100), 0, 1, orientation='v', size=(4, 10), key='-S_value-',enable_events=True, disable_number_display=True),\
                sg.Slider((0, 100), 0, 1, orientation='v', size=(4, 10), key='-V_value-',enable_events=True, disable_number_display=True)],
        [sg.T("0",size=(2,1), font=("Helvetica", 6), k="-H-"), sg.T("0",size=(2,1), font=("Helvetica", 6), k="-S-"), sg.T("0",size=(2,1), font=("Helvetica", 6), k="-V-")]
    ])
    
    Upper_slider = sg.Column([
        [sg.T("H",size=(1,1), font=("Helvetica", 8)), sg.T("S",size=(1,1), font=("Helvetica", 8)), sg.T("V",size=(1,1), font=("Helvetica", 8))],
        [sg.Slider((0, 360), 360, 1, orientation='v', size=(4, 8), key='-up_H_value-',enable_events=True, disable_number_display=True),\
            sg.Slider((0, 100), 100, 1, orientation='v', size=(4, 10), key='-up_S_value-',enable_events=True, disable_number_display=True),\
                sg.Slider((0, 100), 100, 1, orientation='v', size=(4, 10), key='-up_V_value-',enable_events=True, disable_number_display=True)],
        [sg.T("0",size=(2,1), font=("Helvetica", 6), k="-up_H-"), sg.T("0",size=(2,1), font=("Helvetica", 6), k="-up_S-"), sg.T("0",size=(2,1), font=("Helvetica", 6), k="-up_V-")]
    ])
    
    range_button = sg.Column([
        [sg.Radio("HSV range","Radio", k="-hsv_range-")],
        [sg.ColorChooserButton("UPPER", target="-upper_color-", k="-upper_button-", font=("Helvetica", 8)),sg.Button(" ", k="-upper_set-", size=(2,1), font=("Helvetica", 8))],
        [sg.InputText(key="-upper_color-", size=(0,0),font=("Helvetica", 8))],
        [Upper_slider],
        [sg.ColorChooserButton("LOWER", target="-lower_color-", k="-lower_button-",font=("Helvetica", 8)), sg.Button(" ", k="-lower_set-", size=(2,1), font=("Helvetica", 8))],
        [sg.InputText(key="-lower_color-",size=(0,0), font=("Helvetica", 8))],
        [HSV_slider],
    ])
    
    hist_layout = sg.Column([
        [sg.Button("Replot", key="-replot-"), sg.Checkbox("Eq_Gray", default=False, key="-show-", font=("Helvetica", 8)), sg.Checkbox("Gray", default=False, key="-gray-", font=("Helvetica", 8))],
        [sg.Image(filename="",k='-hist-')],
        [sg.Image(filename="",k='-eq_hist-')],        
    ], vertical_alignment='top')
    
    
    setting_layout =[
                    [none_radio],   
                    [threshold_radio, threshold_slid, threshold_value],
                    [hue_radio, hue_slid, hue_value],
                    [blur_radio, blur_slid, blur_value],
                    [denoise_radio, denoise_level, denoise_value],
                    [ehance_radio, ehance_slid, enhance_value],
                    [canny_radio,canny_a,canny_a_value, canny_b, canny_b_value],
                    [range_button, sg.Frame("Hist", [[hist_layout]])],

                    ]

    menu_def = [['&Application', ['E&xit']],
                ['&About', ['&About']] ]

    original_image = [[sg.Image(filename='', key='-orginal_img-')]]
    modify_image = [[sg.Image(filename='', key='-modify_img-')]]

    treedata = get_tree_data("", os.getcwd())

    read_layout = [[sg.Text("Folder: "), sg.InputText(key='-browse_folder-', enable_events=True, ),],
                    [sg.Text("Output:"), sg.InputText(key='-out_path-', enable_events=True,)],
                    [sg.Cancel(button_color="red"), sg.Button('Save',key='-save-',button_color=('black', '#4adcd6')), sg.ProgressBar(100, orientation='h', size=(24, 20),bar_color=("dark blue", "light gray"), key='-PROGRESS BAR-')],
                    [sg.Image(filename='', key="-mod_img-")]
                    ]
    
    info_layout = [
        [sg.T("Title:", size=(10,1), justification="left"), sg.T("", size=(18,1), justification="right", k="-name-", font=("Helvetica", 10))],
        [sg.T("Format:", size=(10,1), justification="left"), sg.T("", size=(18,1), justification="right", k="-ext-", font=("Helvetica", 10))],
        [sg.T("File size:", size=(10,1), justification="left"), sg.T("", size=(18,1), justification="right", k="-file_size-", font=("Helvetica", 10))],
        [sg.T("Image size (raw):", size=(15,1), justification="left"), sg.T("", size=(13,1), justification="right", k="-img_size-", font=("Helvetica", 10))],

        ]
    
    col_1 = sg.Column([
        [sg.FolderBrowse('Browse', key='-file-', target="-browse_folder-"), sg.Button('Refresh', key='-refresh-', button_color=('black', 'green'))],
        [sg.Tree(data=treedata, headings=[], auto_size_columns=True, num_rows=20, col0_width=26, key="-TREE-", show_expanded=False, enable_events=True)],
        [sg.Frame(title="Original", layout=original_image)]
        ], vertical_alignment='top')
    
    col_2 = sg.Column([
        [sg.Frame("Paths", read_layout), sg.Frame("INFO", info_layout)],
        [sg.Frame("Modified", modify_image)],
        ], vertical_alignment='top')
    
    col_3 = sg.Column([
        [sg.Frame("Settings", setting_layout)],
        [sg.Multiline(size=(47,5), disabled=True, background_color="black",font='courier 8', key='-ML-')]
    ], vertical_alignment='top')

    page_1_layout = [[col_1, col_2, col_3]]
    
    layout = [[sg.Menu(menu_def, key='-MENU-')],
                [sg.Text('', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)],
                    [sg.Text('',size=(150,1), justification='right', font=("Helvetica", 10), k="-time-")]]

    layout +=[[sg.TabGroup([[   sg.Tab('Image_Processing', page_1_layout),
                                sg.Tab('Test', [[sg.Image(filename="")]])
                                ]], key='-TAB GROUP-')]]

    return layout

def main():  
    
    sg.popup_quick_message('Loading... please wiat...', background_color='gray', text_color='white', font='Any 14') 
    time.sleep(1)     
    layout = define_layout()
            
    def update_info():
        filename = Path(values["-TREE-"][0]).stem
        filesize = os.path.getsize(values["-TREE-"][0])/1E3
        
        window["-name-"].update(filename)    
        window["-ext-"].update(file_ext)    
        window["-file_size-"].update(f"{filesize:.1f} KB")    
        window["-img_size-"].update(f"{img_width} x {img_height}")    




    def update_slider_values():
        window['-thes_value-'].update(int(values['-thslid-']))
        window['-blur_value-'].update(int(values['-BLUR SLIDER-']))
        window['-hue_value-'].update(int(values['-HUE SLIDER-']))
        window['-enhance_value-'].update(int(values['-ENHANCE SLIDER-']))
        window['-denoise_value-'].update(int(values['-DENOISE LEVEL-']))
        window['-canny_a-'].update(int(values['-CANNY SLIDER A-']))
        window['-canny_b-'].update(int(values['-CANNY SLIDER B-']))
        window['-H-'].update(int(values['-H_value-']))
        window['-S-'].update(int(values['-S_value-']))
        window['-V-'].update(int(values['-V_value-']))
        window['-up_H-'].update(int(values['-up_H_value-']))
        window['-up_S-'].update(int(values['-up_S_value-']))
        window['-up_V-'].update(int(values['-up_V_value-']))
        
    def update_lower_color():
        hsv = (values["-H_value-"]/360, values["-S_value-"]/100, values["-V_value-"]/100)
        rgb = hsv2rgb(hsv)
        rgb = (np.array(rgb)*255)   
        color = rgb2hex((round(rgb[0]), round(rgb[1]), round(rgb[2])))        
        window["-lower_set-"].update(button_color=color)
            
    def update_upper_color():
        hsv = (values["-up_H_value-"]/360, values["-up_S_value-"]/100, values["-up_V_value-"]/100)
        rgb = hsv2rgb(hsv)
        rgb = (np.array(rgb)*255)    
        color = rgb2hex((round(rgb[0]), round(rgb[1]), round(rgb[2])))
        window["-upper_set-"].update(button_color=color)
                
    window = sg.Window(
        'Image_Processing_GUI', 
        layout,
        location=(10, 10),
        alpha_channel=1.0,
        no_titlebar=False,
        grab_anywhere=False,
         resizable=True, 
        element_justification="left").Finalize()
    
    def_img = cv2.imread("../data/img/default.png")
    def_img = cv2.resize(def_img, (640,540))
    def_bytes = cv2.imencode('.png', def_img)[1].tobytes()
    window["-modify_img-"].update(data=def_bytes)
    
    hsv_cyclinder = cv2.imread("../data/img/hsv_cyclinder.png")


    window["-browse_folder-"].update(value=r"D:\Python\git\Python_GUI\data")

    check_image_ok = False

    while True:
        now = datetime.now().strftime("20%y/%m/%d %H:%M:%S")
        log_now = datetime.now().strftime("[%H:%M] ")
        window["-time-"].update(value=now)

        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        elif event == "-browse_folder-":
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='dark blue', text_color='white', font='Any 14')
            window["-TREE-"].update(values=get_tree_data("", values["-browse_folder-"]))

        elif event == "-TREE-":
            img_path = Path(values['-TREE-'][0])
            _, file_ext = os.path.splitext(img_path) 
            
            if file_ext != "":
                try:
                    src = cv2.imread(str(img_path))
                    img_width, img_height = src.shape[1], src.shape[0]
                    # src_main = imutils.resize(src, width=640)
                    src_main = cv2.resize(src, (640,480))
                    src_copy = np.copy(src_main)
                    src_resize = imutils.resize(src, width=235)
                    window["-none-"].update(value=True)
                    window['-modify_img-'].update(data=cv2.imencode('.png', src)[1].tobytes())
                    window["-out_path-"].update(value=str(values["-TREE-"][0].replace('.jpg', '_modified.jpg').replace('.png', '_modified.png').replace('.jpeg', 'modified_.jpeg')))
                    
                    _, _, hist_bytes = draw_hist(src_copy)
                    window["-hist-"].update(data=hist_bytes)
                    
                    window['-ML-'].print(log_now, end="")
                    window["-ML-"].print(f"Successful", background_color='green',text_color='white', end='')
                    window["-ML-"].print(f" Image loaded successfully.", text_color="green")
                    
                    update_info()
                    
                    check_image_ok = True
                except Exception as ex:
                    window['-ML-'].print(log_now, end="")
                    window['-ML-'].print(f'{str(type(ex).__name__)}', background_color='red',text_color='white', end='')
                    window['-ML-'].print(f' {ex.args[0]}. (expected: [.jpg .jpeg .png])', text_color='red')
                else:
                    img_bytes = cv2.imencode('.png', src_resize)[1].tobytes()
                    window["-orginal_img-"].update(data=img_bytes)

        elif event == "-lower_set-": 
            try:
                lower_rgb = ImageColor.getcolor(values["-lower_color-"], "RGB")
                lower_hsv = rgb2hsv(lower_rgb)            
                
                window['-H_value-'].update(lower_hsv[0]*360)
                window['-S_value-'].update(lower_hsv[1]*100)
                window['-V_value-'].update(lower_hsv[2]*100/255)
            except:
                window['-ML-'].print(log_now, end="")
                window["-ML-"].print(f"ValueError", background_color='red',text_color='white', end='')
                window["-ML-"].print(" Invalid Hex-code.", text_color="red")
                
        elif event == "-upper_set-":
            try:
                upper_rgb =  ImageColor.getcolor(values["-upper_color-"], "RGB")
                upper_hsv = rgb2hsv(upper_rgb)

                window['-up_H_value-'].update(int(upper_hsv[0]*360))
                window['-up_S_value-'].update(int(upper_hsv[1]*100))
                window['-up_V_value-'].update(int(upper_hsv[2]*100/255))
            
            except:
                window['-ML-'].print(log_now, end="")
                window["-ML-"].print(f"ValueError", background_color='red',text_color='white', end='')
                window["-ML-"].print(" Invalid Hex-code.", text_color="red")
                
        elif event == "-save-":
            window["-ML-"].print(log_now, end="")
            window["-ML-"].print(" Saving...", end="")
            progress_bar = window['-PROGRESS BAR-']
            for i in range(101):
                progress_bar.UpdateBar(i + .1)
            try:
                img_out = cv2.resize(mod_img, (img_width, img_height))
                cv2.imwrite(values["-out_path-"], img_out)
                # window["-ML-"].print(log_now, end="")
                window["-ML-"].print("Done.")
                window["-ML-"].print(" File saved to: "+values["-out_path-"], text_color='green')
                time.sleep(1)
            except Exception as ex:
                window["-ML-"].print(log_now, end="")
                window['-ML-'].print('TypeError', background_color='red',text_color='white', end='')
                window['-ML-'].print(' Cound not save image. Make sure the format is correct (expected: .jpg, .png, .jpeg)', text_color='red')

            
            progress_bar.UpdateBar(0)
            
        elif event == "-refresh-":
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='dark blue', text_color='white', font='Any 14')
            window["-TREE-"].update(values=get_tree_data("", values["-browse_folder-"]))
            
        if check_image_ok:
            mod_img = src_copy
            
            if not values["-hsv_range-"]:
                window["-orginal_img-"].update(data=cv2.imencode('.png', src_resize)[1].tobytes())   
            
            if values["-hsv_range-"]:
                hsv_cyclinder = imutils.resize(hsv_cyclinder, width=235)

                lower_range = np.array((values["-H_value-"], round(values["-S_value-"]*2.55), round(values["-V_value-"]*2.55)), dtype=int)
                upper_range = np.array((values["-up_H_value-"], round(values["-up_S_value-"]*2.55), round(values["-up_V_value-"]*2.55)), dtype=int)
                
                try:   
                    img_hsv = cv2.cvtColor(src_copy, cv2.COLOR_RGB2HSV)
                    mask = cv2.inRange(img_hsv, lower_range, upper_range)
                    mod_img = cv2.bitwise_and(src_copy, src_copy, mask=mask)
                    
                    hsv_c = cv2.cvtColor(hsv_cyclinder, cv2.COLOR_RGB2HSV)
                    hsv_mask = cv2.inRange(hsv_c, lower_range, upper_range)
                    hsv_c = cv2.bitwise_and(hsv_cyclinder, hsv_cyclinder, mask=hsv_mask)
                    
                    img_bytes = cv2.imencode('.png', hsv_c)[1].tobytes()
                    window["-orginal_img-"].update(data=img_bytes)            
                except Exception as ex:
                    pass
            elif values['-THRESHOLD-']: 
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

            if event == "-replot-":
                _, _, hist_bytes = draw_hist(mod_img)
                window["-hist-"].update(data=hist_bytes)                

            if values['-show-']:
                _, mod_img, _ = draw_hist(mod_img)
            elif values['-gray-']:
                mod_img, _, _ = draw_hist(mod_img)
                
            if values["-show-"] and values["-gray-"]:
                window["-ML-"].print(log_now, end="")
                window['-ML-'].print('WARNING:',text_color='yellow', end='')
                window['-ML-'].print(" Both 'Gray' amd 'Equalized Gray' are seleted.", text_color='yellow')
                window["-show-"].update(value=False)
                window["-gray-"].update(value=False)


            if values['-CANNY-']:
                mod_img = cv2.Canny(src_copy, values['-CANNY SLIDER A-'], values['-CANNY SLIDER B-'])
                check_canny = True
            else:
                check_canny = False

            window['-modify_img-'].update(data=cv2.imencode('.png', mod_img)[1].tobytes())
        
        
        update_lower_color()
        update_upper_color()
        update_slider_values()
        
    window.close()
    exit(0)
        
if __name__ == "__main__":
    main()
    