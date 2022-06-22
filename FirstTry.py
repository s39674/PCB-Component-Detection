"""
This program can be used to extract valuable information from the pcb wacv 2019 dataset: https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection
"""


import sys
import os
import xml.etree.ElementTree as ET
import cv2

unwantedComps = ["text", "pins", "pads", "unknown", "switch", "test"]
wantedComps = ["resistor", "capacitor", "inductor", "diode", "led", "ic", "transistor", "connector", "jumper", "emi_filter",  "button", "clock", "transformer", "potentiometer", "heatsink", "fuse", "ferrite_bead", "buzzer", "display", "battery"]
frequency_wantedComps = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]

labels_map = {
    0: wantedComps[0],
    1: wantedComps[1],
    2: wantedComps[2],
    3: wantedComps[3],
    4: wantedComps[4],
    5: wantedComps[5],
    6: wantedComps[6],
    7: wantedComps[7],
    8: wantedComps[8],
    9: wantedComps[9],
    10: wantedComps[10],
    11: wantedComps[11],
    12: wantedComps[12],
    13: wantedComps[13],
    14: wantedComps[14],
    15: wantedComps[15],
    16: wantedComps[16],
    17: wantedComps[17],
    18: wantedComps[18],
    19: wantedComps[19]


}

datasetPath = "pcb_wacv_2019/"
xmlPath = None

formattedDatasetPath = "pcb_wacv_2019_formatted/"

compCounter = 0

# from: https://stackoverflow.com/questions/35180764/opencv-python-image-too-big-to-display
def ResizeWithAspectRatio(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        r = width / float(w)
        dim = (width, int(h * r))

    return cv2.resize(image, dim, interpolation=inter)

with open("pcb_wacv_2019_formatted.csv", "a") as csvFile:
    for folder in os.listdir(datasetPath):
        xmlPath = None
        folderPath = os.path.join(datasetPath, folder)
        if os.path.isdir(folderPath):
            for file in os.listdir(folderPath):
                if file.endswith(".xml"):
                    xmlPath = os.path.join(folderPath, file)
        else: continue
        
        if not xmlPath:
            print("[WW] xml not recogonized, skip!")
            continue
        
        # loag xml
        print(f"Proccesing xml: {xmlPath}")
        tree = ET.parse(xmlPath)
        root = tree.getroot()

        # load img
        imgPath = os.path.join(folderPath, root[1].text)
        img = cv2.imread(imgPath)
        if img is None:
            print(f"[WW] couldn't load image! path: {imgPath}; skip!")
            continue
        
        showImg = img.copy()
        # if image too large, resize with aspect ratio
        if img.shape[0] > 1820 or img.shape[1] > 980:
            showImg = ResizeWithAspectRatio(showImg, 1200)
            
        #cv2.imshow(f"PCB: {imgPath}", showImg)
        #cv2.waitKey(0)

        # first six blocks are metadata
        for i in range(6,len(list(root))):
            bndbox = root[i][4]
            Xmin = int(bndbox[0].text)
            Ymin = int(bndbox[1].text)
            Xmax = int(bndbox[2].text)
            Ymax = int(bndbox[3].text)

            compName = root[i][0].text

            modified = False

            if any(x in compName for x in unwantedComps):
                # Could be intresting to create a OCR model to detect pcb text
                continue

            # getting rid of position; exp: resistor R3 => resistor
            for idx, x in enumerate(wantedComps):
                if x in compName: 
                    compName = wantedComps[idx]
                    modified = True
                    break
            
            # Keeping connector type: connector CN15 => connector CN
            #if "connector" in compName:
            #    compName = compName[0:12]
            #    modified = True
            #if "jumper" in compName:
            #    compName = compName[0:9]
            #    modified = True
            if "emi filter" in compName:
                compName = "emi_filter"
                modified = True
            elif "ferrite bead" in compName:
                compName = "ferrite_bead"
                modified = True

            if not modified:
                print(f"[WW] Found a component without a case: {compName}")
                continue
            
            index = wantedComps.index(compName)
            try:
                frequency_wantedComps[index] += 1
            except:
                print(f"CompName: {compName} index: {index}")
            fileName = compName + str(frequency_wantedComps[index])
            filePath = compName + "/" + fileName + ".jpg"

            print(f"name: {compName}; filepath: {filePath}")
            Component = img[Ymin: Ymax, Xmin: Xmax ].copy()
            cv2.imwrite(f'{formattedDatasetPath}{filePath}', Component)
            csvFile.write(f"{fileName}.jpg, {index}\n")
            #cv2.imshow("Component", Component)
            #cv2.waitKey(0)