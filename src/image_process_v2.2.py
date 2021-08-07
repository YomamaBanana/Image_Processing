from tkinter.constants import E
import warnings
import PySimpleGUI as sg
import os, cv2
from pathlib import Path
import imutils
import matplotlib.pyplot as plt
from datetime import datetime
from PIL import ImageColor
import time
from kneed import KneeLocator
from sklearn.metrics import silhouette_score

from utils import *
from tab2 import tab2_layout
from tab1 import tab1_layout
from tab3 import tab3_layout

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
    tab3 = tab3_layout()

    layout +=[[sg.TabGroup([[   sg.Tab('Image_Processing', tab1),
                                sg.Tab('Color_Separation', tab2),
                                sg.Tab('YOLOv3', tab3)
                                ]], key='-TAB GROUP-')]]
    
    return layout

def main():  
    
    sg.popup_quick_message('Loading... please wiat...', background_color='darkblue', text_color='white', font='Any 14') 
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

    check_image_ok, t2_init = False, True

    while True:
        now = datetime.now().strftime("20%y/%m/%d %H:%M:%S")
        log_now = datetime.now().strftime("[%H:%M] ")
        window["-time-"].update(value=now)

        event, values = window.read(timeout=100)
        
        if event in (None, 'Cancel', 'Exit'):
            break

        elif event == "-browse_folder-":
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='darkblue', text_color='white', font='Any 14')
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
            sg.popup_quick_message('Updating folder... Please wiat...', background_color='darkblue', text_color='white', font='Any 14')
            
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
            
            if t2_init:
                fig = empty_plot(3,3,"Color Result")
                window["show_color"].update(data=fig)
                
                fig = empty_plot(3,3,"Mask Result")
                window[f"show_mask"].update(data=fig)
                
                fig = empty_plot(6,3,"Elbow Graph")
                window[f"elbow"].update(data=fig)

                fig = empty_plot(6,3,"Dominant Colors")
                window[f"color"].update(data=fig)

                t2_init = False
            
            try:
                t2_img = imutils.resize(mod_img, width=235)
                window["t2-image"].update(data=cv2.imencode('.png', t2_img)[1].tobytes())
                t2_copy = np.copy(t2_img)
            except Exception as er:
                sg.popup_ok("No image seleted.\nPress OK to continue... ",title="WARNING!")
                window["-TAB GROUP-"].Widget.select(0)

            if not check_rgb:
                try:
                    t2_copy[:,:,[0,2]] = t2_copy[:,:,[2,0]]
                except:
                    print("SHIT")
                    
                check_rgb = True

            if event == "plot_elbow":
                try:
                    window['-t2_ML-'].print(log_now+" running...", end="")
                    sg.popup_quick_message('Processing... \nThis might take a while...', background_color='darkblue', text_color='white', font='Any 14')
                    
                    max_cluster = int(values["t2-max_clus"])                    
                    x, y, eblow = elbow_plot(t2_img, max_clusters=max_cluster)
                    
                    window["elbow"].update(data=eblow)
                    
                    window["-t2_ML-"].print(f"successful.", end="")
                    window["-t2_ML-"].print("")
                    
                    kl = KneeLocator(range(1, max_cluster), y, curve="convex", direction="decreasing")
                    window["-t2_ML-"].print(f"Approximation:", background_color='darkblue', text_color='white', end="")
                    # window["-t2_ML-"].print("")                    
                    window["-t2_ML-"].print(f"  {kl.elbow} clusters.")
                    
                except Exception as er:                    
                    print(er)
            
            elif event == "apply_kmeans":
                    
                try:
                    window['-t2_ML-'].print(log_now+" running...", end="")
                    num_cluster = int(values["kmeans_num"])
                    
                except Exception as ex:
                    window['-t2_ML-'].print(log_now, end="")
                    window['-t2_ML-'].print(f'{str(type(ex).__name__)}', background_color='red',text_color='white', end='')
                    window['-t2_ML-'].print(f' {ex.args[0]}. (expected: int)', text_color='red')
                
                if clustering_methhod == 0:
                    sg.popup_quick_message('Processing K-means ... ', background_color='darkblue', text_color='white', font='Any 14')
                    centroids, array, vecs, shape, counts, kmeans = k_means_clustering(t2_copy, num_cluster)    
                    

                    score = silhouette_score(array, kmeans.labels_)
                    index_max = np.argmax(counts)
                    peak = centroids[index_max]


                    window["-t2_ML-"].print(f"successful.", end="")
                    window["-t2_ML-"].print("")

                    window["-t2_ML-"].print("[Summary]", background_color='blue',text_color='white', end="")
                    window["-t2_ML-"].print("")
                    window["-t2_ML-"].print(" ", background_color="blue", end="")
                    window["-t2_ML-"].print(f"n_iter: {kmeans.n_iter_}")
                    window["-t2_ML-"].print(" ", background_color="blue", end="")
                    window["-t2_ML-"].print(f"inertia: {kmeans.inertia_:.2f}")
                    window["-t2_ML-"].print(" ", background_color="blue", end="")
                    window["-t2_ML-"].print(f"silhoette score: {score:.2f}")
                    window["-t2_ML-"].print(" ", background_color="blue", end="")
                    window["-t2_ML-"].print(f"most frequent: {np.array(peak, dtype=int)}")
                    window["-t2_ML-"].print("[--END--]", background_color='blue',text_color='white', end="")
                    window["-t2_ML-"].print("")



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

            if event == "table":
                try:
                    idx = int(values["table"][0])
                    fig = plot_top_colors(rgb_list[idx], idx, array, vecs, shape, counts, indices)
                    window[f"show_color"].update(data=fig)
                    
                    mask = plot_mask(rgb_list[idx], idx, array, vecs, shape, counts, indices)
                    window[f"show_mask"].update(data=mask)  
                except:
                    window["-t2_ML-"].print(log_now, end="")
                    window['-t2_ML-'].print('WARNING:',text_color='yellow', end='')
                    window['-t2_ML-'].print(" No cluster found.", text_color='yellow')
                    
        
        ### TAB3 PROCESS
        if values["-TAB GROUP-"] == "YOLOv3":
            try:
                t2_img = imutils.resize(mod_img, width=235)
                (H, W) = t2_img.shape[:2]
                window["show_image"].update(data=cv2.imencode('.png', t2_img)[1].tobytes())
                t2_copy = np.copy(t2_img)
            except Exception as er:
                sg.popup_ok("No image seleted.\nPress OK to continue... ",title="WARNING!")
                window["-TAB GROUP-"].Widget.select(0)

            if event == "run":
                labels = os.path.join(values["network"], "coco.names")
                cfg = os.path.join(values["network"], "yolov3.cfg")
                weights = os.path.join(values["network"], "yolov3.weights")
                
                print("[INFO] loading YOLO from disk...", end="\r")
                net = cv2.dnn.readNetFromDarknet(cfg, weights)
                print("[INFO] loading YOLO from disk... done.")

                img = t2_img.copy()

                def run_yolo():
                    ln = net.getLayerNames()
                    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
                    blob = cv2.dnn.blobFromImage(img, 1 / 255.0, (416, 416),
                    swapRB=True, crop=False)
                    net.setInput(blob)
                    start = time.time()
                    layerOutputs = net.forward(ln)
                    end = time.time()

                    LABELS = open(labels).read().strip().split("\n")

                    COLORS = np.random.randint(0, 255, size=(len(LABELS), 3),
                        dtype="uint8")

                    # show timing information on YOLO
                    print("[INFO] YOLO took {:.6f} seconds".format(end - start))
                    boxes = []
                    confidences = []
                    classIDs = []

                    # loop over each of the layer outputs
                    for output in layerOutputs:
                        # loop over each of the detections
                        for detection in output:
                            # extract the class ID and confidence (i.e., probability) of
                            # the current object detection
                            scores = detection[5:]
                            classID = np.argmax(scores)
                            confidence = scores[classID]

                            # filter out weak predictions by ensuring the detected
                            # probability is greater than the minimum probability
                            if confidence > 0.5:
                                # scale the bounding box coordinates back relative to the
                                # size of the image, keeping in mind that YOLO actually
                                # returns the center (x, y)-coordinates of the bounding
                                # box followed by the boxes' width and height
                                box = detection[0:4] * np.array([W, H, W, H])
                                (centerX, centerY, width, height) = box.astype("int")

                                # use the center (x, y)-coordinates to derive the top and
                                # and left corner of the bounding box
                                x = int(centerX - (width / 2))
                                y = int(centerY - (height / 2))

                                # update our list of bounding box coordinates, confidences,
                                # and class IDs
                                boxes.append([x, y, int(width), int(height)])
                                confidences.append(float(confidence))
                                classIDs.append(classID)

                    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
                
                    if len(idxs) > 0:
                        # loop over the indexes we are keeping
                        for i in idxs.flatten():
                            # extract the bounding box coordinates
                            (x, y) = (boxes[i][0], boxes[i][1])
                            (w, h) = (boxes[i][2], boxes[i][3])

                            # draw a bounding box rectangle and label on the image
                            color = [int(c) for c in COLORS[classIDs[i]]]
                            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
                            text = "{}: {:.4f}".format(LABELS[classIDs[i]], confidences[i])
                            cv2.putText(img, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX,
                                0.5, color, 2)
                
                run_yolo()
                
                imgbytes = cv2.imencode('.png', img)[1].tobytes()
                window["yolo_image"].update(data=imgbytes)

        update_lower_color()
        update_upper_color()
        update_slider_values()
        
    window.close()
    exit(0)
        
if __name__ == "__main__":
    main()
    