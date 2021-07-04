import PySimpleGUI as sg
import os

from utils import get_tree_data


def tab1_layout():
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
    
    return page_1_layout