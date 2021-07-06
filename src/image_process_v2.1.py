import PySimpleGUI as sg
import os, cv2
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageColor
import time


from utils import *
from tab2 import tab2_layout
from tab1 import tab1_layout

plt.style.use('dark_background')
plt.rcParams['lines.linewidth'] = 0.6
plt.rcParams['ytick.labelleft'] = False
plt.rcParams['axes.titlesize'] = 10

def define_layout():    
    sg.theme('DarkGrey9')
    
    menu_def = [['&Application', ['E&xit']],
                ['&About', ['&About']] ]

    layout = [[sg.Menu(menu_def, key='-MENU-')],
                [sg.Text('', size=(38, 1), justification='center', font=("Helvetica", 16), relief=sg.RELIEF_RIDGE, k='-TEXT HEADING-', enable_events=True)],
                    [sg.Text('',size=(150,1), justification='right', font=("Helvetica", 10), k="-time-")]]

    tab1 = tab1_layout()
    tab2 = tab2_layout()

    layout +=[[sg.TabGroup([[   sg.Tab('Image_Processing', tab1),
                                sg.Tab('Color_Separation', tab2)
                                ]], key='-TAB GROUP-')]]
    
    return layout

def main():  
    
    sg.popup_quick_message('Loading... please wiat...', background_color='gray', text_color='white', font='Any 14') 
    time.sleep(1)     
            
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
                
                
    layout = define_layout()
    
        
                
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
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='gray', text_color='white', font='Any 14')
            window["-TREE-"].update(values=get_tree_data("", values["-browse_folder-"]))

        elif event == "-TREE-":
            img_path = Path(values['-TREE-'][0])
            _, file_ext = os.path.splitext(img_path) 
            
            if file_ext != "":
                try:
                    src = cv2.imread(str(img_path))
                    img_width, img_height = src.shape[1], src.shape[0]
                    src_main = imutils.resize(src, height=max(min(600, img_height),540))
                    src_copy = imutils.resize(src, height=max(min(600, img_height),540))
                    # src_main = cv2.resize(src, (640,480))
                    # src_copy = np.copy(src_main)
                    src_resize = imutils.resize(src, width=235)
                    window["-none-"].update(value=True)
                    window['-modify_img-'].update(data=cv2.imencode('.png', src_main)[1].tobytes())
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
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='gray', text_color='white', font='Any 14')
            
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
        
        
        ### TAB2 PROCESS
        check_rgb = False
        if values["-TAB GROUP-"] == "Color_Separation":
            try:
                t2_img = imutils.resize(mod_img, width=235)
                window["t2-image"].update(data=cv2.imencode('.png', t2_img)[1].tobytes())
                t2_copy = np.copy(t2_img)
            except:
                pass

            if not check_rgb:
                t2_copy[:,:,[0,2]] = t2_copy[:,:,[2,0]]
                check_rgb = True

            if event == "plot_elbow":
                try:
                    max_cluster = int(values["t2-max_clus"])                    
                    x,y,eblow = elbow_plot(t2_img, max_clusters=max_cluster)
                    window["elbow"].update(data=eblow)
                    z, d1, d2 = polyfit3d(x,y)
                    print(d1,d2)                    
                    # for idx, win in enumerate(["poly_a", "poly_b", "poly_c", "poly_d"]):
                        # window[win].update(round(z[idx],2))
                    
                    for idx, win in enumerate(["roots_1", "roots_2", "roots_3"]):
                        print(idx)
                        if idx == 2:
                            window[win].update(round(d2[0],2))
                        else:
                            if not np.iscomplex(d1[idx]): 
                                window[win].update(round(d1[idx],2))
                            else:
                                window[win].update("complex")
                except Exception as er:                    
                    print(er)
            
            elif event == "apply_kmeans":
                try:
                    num_cluster = int(values["kmeans_num"])
                except:
                    pass
                
                if clustering_methhod == 0:
                    centroids, array, vecs, shape, counts = k_means_clustering(t2_copy, num_cluster)    
                elif clustering_methhod == 1:
                    centroids, wieghts, counts, vecs, shape, array = gaussian_mixture(t2_copy)
                
                rgb_list, counts_list, indices,color_plot = plot_color_histogram(centroids, counts)
                window["color"].update(data=color_plot)



                table_list = np.zeros((len(rgb_list),5), dtype=object)

                for idx, rgb in enumerate(rgb_list):
                    table_list[idx,0] = int(idx+1)
                    table_list[idx,1] = int(rgb[0])
                    table_list[idx,2] = int(rgb[1])
                    table_list[idx,3] = int(rgb[2])
                    table_list[idx,4] = round((100*counts_list[idx]/np.sum(counts_list)),1)
                    
                window["table"].update(values=table_list.tolist())




                
            if event == "plot_top5":
                
                for color in range(1, min(7,len(centroids)+1)):
                    idx = color - 1
                    window[f"top{color}"].update(data=None)
                    fig = plot_top_colors(rgb_list[idx], idx, array, vecs, shape, counts, indices)
                    window[f"top{color}"].update(data=fig)
                
            if values["clus_method"] == "BayesianGaussianMixture":
                window["kmeans_num"].update(text_color="gray", disabled=True)
                clustering_methhod = 1
            else:
                window["kmeans_num"].update(text_color="white", disabled=False)
                clustering_methhod = 0
                                
            
        update_lower_color()
        update_upper_color()
        update_slider_values()
        
    window.close()
    exit(0)
        
if __name__ == "__main__":
    main()
    