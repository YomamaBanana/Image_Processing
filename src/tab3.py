import PySimpleGUI as sg

from tab1 import tab1_layout


def tab3_layout():
    sg.theme('DarkGrey9')
    
    
    
    col_1 = sg.Column([
        [sg.Text('Image'), sg.In(r'path',size=(30,1), key='image'), sg.FileBrowse("File", k="file")],
        [sg.Text('Network'), sg.In(r'path',size=(30,1), key='network'), sg.FolderBrowse()],
        # [sg.Text('Weights'), sg.In(r'path',size=(30,1), key='image'), sg.FileBrowse()],
    ])
    
    col_2 = sg.Column([
        [sg.T("COL 2")]
    ])
    
    col_3 = sg.Column([
        [sg.T("COL 3")]
    ])
    
    
    page_3_layout = [
        [col_1, col_2, col_3],
        [sg.T("Result"), sg.Button("Run", k="run")],
        [sg.Image(filename="", key="show_image")],
        [sg.Image(filename="", key="yolo")]
        ]
    
    
    
    return page_3_layout

def main():
    
    # confidence = 0.5
    # threshold = 0.1
    
    layout = tab3_layout()
    
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
        
        if event in (None, 'Exit'):
            break

        if event == "run":
            import cv2, os, time
            import numpy as np
            
            
            try:
                img = cv2.imread(values["image"])
                
                img = cv2.resize(img, (400,300))
                (H, W) = img.shape[:2]
                
                window["show_image"].update(data=cv2.imencode(".png", img)[1].tobytes())
            except:
                pass
            labels = os.path.join(values["network"], "coco.names")
            cfg = os.path.join(values["network"], "yolov3.cfg")
            weights = os.path.join(values["network"], "yolov3.weights")
            
            print("[INFO] loading YOLO from disk...", end="\r")
            net = cv2.dnn.readNetFromDarknet(cfg, weights)
            print("[INFO] loading YOLO from disk... done.")
            
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
        
            imgbytes = cv2.imencode('.png', img)[1].tobytes()
            window["yolo"].update(data=imgbytes)
            

    
                
if __name__ == "__main__":
    main()    
    