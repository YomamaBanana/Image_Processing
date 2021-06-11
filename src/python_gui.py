import datetime
import io
from pathlib import Path

import cv2
import numpy as np
from matplotlib import pyplot as plt
import PySimpleGUI as sg

import imutils


def draw_plot(img_f):
        
    plt.clf()
    plt.figure(figsize=(4,2))
        
    for i, channel in enumerate(("r", "g", "b")):
            histgram = cv2.calcHist([img_f], [i], None, [256], [0, 256])
            plt.plot(histgram, color = channel)
            plt.xlim([0, 256])

    item = io.BytesIO()
    plt.savefig(item, format='png') 
    plt.clf()
    plt.close('all')

    return item.getvalue()

original_image = [[sg.Image(filename='', key='-orginal_img-')]]
modify_image = [[sg.Image(filename='', key='-modify_img-')]]

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

read_layout = [[sg.Text("File:"), sg.InputText(key='-input_file-', enable_events=True, ),
                sg.FileBrowse('Browse', key='-file-', target="-input_file-",), sg.Button('Read', key='-read_file-')],
                [sg.Text("Output:"), sg.InputText(key='-out_path-', enable_events=True,)],
                [sg.Cancel(), sg.Button('Save',key='-save-',button_color=('black', '#4adcd6')), sg.ProgressBar(100, orientation='h', size=(20, 20),bar_color=("purple", "green"), key='-PROGRESS BAR-'), sg.Text("         ",key='-saved-')]]

setting_layout =[
                [threshold_radio, threshold_slid],
                [blur_radio, blur_slid],
                [hue_radio, hue_slid],
                [ehance_radio, ehance_slid],
                [canny_radio, canny_a, canny_b]
                ]
hist_graph = [[sg.Image(filename='', key='-hist_img-')]]


layout_tot= [[sg.Frame(title='Read',layout=read_layout)],
             [sg.Frame(title='Parameter', layout=setting_layout),sg.Frame(title='Histgram',layout=hist_graph)],
             [sg.Frame(title='Original',layout=original_image),sg.Frame(title='Results',layout=modify_image),]]


READ_File = False

sg.theme('Dark Blue 3')

window = sg.Window('Image Processing', layout_tot,
                   location=(10, 10),alpha_channel=1.0,
                   no_titlebar=False,grab_anywhere=False).Finalize()

while True:

    event, values = window.read(timeout=20)

    if event in (None, 'Cancel'):
        break

    elif event == '-read_file-':
        read_path = Path(values['-input_file-'])
        default_out = str(read_path).replace(str(read_path.stem),str(read_path.stem)+"_modified")
        window['-out_path-'].Update(value=default_out)

        read_img = cv2.imread(str(read_path))
        read_img = imutils.resize(read_img, width=400)
        imgbytes = cv2.imencode('.png', read_img)[1].tobytes()
        mod_img = read_img.copy()
        histbytes = draw_plot(read_img)
        window['-orginal_img-'].update(data=imgbytes)
        window['-modify_img-'].update(data=imgbytes)
        window['-hist_img-'].update(data=histbytes)
        READ_File = True

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

        if not values['-CANNY-']:
            histbytes = draw_plot(mod_img)
            window['-hist_img-'].update(data=histbytes)

        mod_imgbytes = cv2.imencode('.png', mod_img)[1].tobytes()
        window['-modify_img-'].update(data=mod_imgbytes)


window.close()