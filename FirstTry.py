"""
This program can be used to extract valuable information from the pcb wacv 2019 dataset: https://sites.google.com/view/chiawen-kuo/home/pcb-component-detection
"""


import sys
import os
import xml.etree.ElementTree as ET
import cv2

unwantedComps = ["text", "pins", "pads", "unknown", "switch", "test"]
wantedComps = ["resistor", "capacitor", "inductor", "diode", "led", "ic", "transistor", "emi filter",  "button", "clock", "transformer", "potentiometer", "heatsink", "fuse", "ferrite bead", "buzzer", "display", "battery"]

Path = "pcb_wacv_2019/"
xmlPath = None

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


for folder in os.listdir(Path):
    xmlPath = None
    folderPath = os.path.join(Path, folder)
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
        
    cv2.imshow(f"PCB: {imgPath}", showImg)
    cv2.waitKey(0)

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
        if "connector" in compName:
            compName = compName[0:12]
            modified = True
        elif "jumper" in compName:
            compName = compName[0:9]
            modified = True

        if not modified:
            sys.exit(f"Found a component without a case: {compName}")

        print(f" name: {compName}; Xmin: {Xmin}; Ymin: {Ymin}; Xmax: {Xmax} Ymax: {Ymax}")
        Component = img[Ymin: Ymax, Xmin: Xmax ].copy()
        cv2.imshow("Component", Component)
        cv2.waitKey(0)
