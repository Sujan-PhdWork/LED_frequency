import pandas as pd
import numpy as np
import cv2
import matplotlib.pyplot as plt

from glob import glob
import IPython.display as ipd
from tqdm.notebook import tqdm

import subprocess

# import pytesseract

# Mention the installed location of Tesseract-OCR in your system
# pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files/Tesseract-OCR/tesseract.exe'


input_file = './video_files/video_3.mp4'
cap = cv2.VideoCapture(input_file)
n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

# boundaries_blue = [([255,255,190],[255,255,255])]
boundaries_blue = [([100,0,0],[255,255,0])]
boundaries_red = [([139,112,240],[255,255,255])]
boundaries_white = [([240,240,240],[255,255,255])]

for (lower_blue, upper_blue) in boundaries_blue:
    lower_blue = np.array(lower_blue, dtype = "uint8")
    upper_blue = np.array(upper_blue, dtype = "uint8")

for (lower_red, upper_red) in boundaries_red:
    lower_red = np.array(lower_red, dtype = "uint8")
    upper_red = np.array(upper_red, dtype = "uint8")
    
for (lower_white, upper_white) in boundaries_white:
    lower_white = np.array(lower_white, dtype = "uint8")
    upper_white = np.array(upper_white, dtype = "uint8")
    
threshold = 0.8
ping = 0
pong = 0

font = cv2.FONT_HERSHEY_SIMPLEX
color = (0, 0, 0) # red
fontsize = 1
thickness = 2
text = "test"
position = (50, 50)

img_idx = 0
for frame in range(n_frames):
    ret, img = cap.read()
    if ret == False:
        break
    if frame % 1 == 0:
        match_found = False
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Cropping an image
        
        # for video 1
        # cropped_image = img[1040:1350, 0:1080]
        # for video 2
        # cropped_image = img[340:630, 0:1080]
        # for video 3
        cropped_image = img[1290:1600, 0:1080]
        mask_blue = cv2.inRange(cropped_image, lower_blue, upper_blue)
        mask_red = cv2.inRange(cropped_image, lower_red, upper_red)
        
        # cv2.imwrite('./images/res'+str(img_idx)+'.png', cropped_image)
        output_blue = cv2.bitwise_and(cropped_image, cropped_image, mask = mask_blue)
        output_red_m = cv2.bitwise_and(cropped_image, cropped_image, mask = mask_red)
        # cv2.imwrite('./images/res'+str(img_idx)+'_blue.png', output)
        mask_white = cv2.inRange(output_red_m, lower_white, upper_white)
        output_red = cv2.bitwise_not(output_red_m, output_red_m, mask = mask_white)
        
        if np.mean(output_blue) > 0.05:
            match_found = True
            cv2.imwrite('./images/res'+str(img_idx)+'_mask.png',output_blue)
        
        if np.mean(output_blue) > 0.05 and np.mean(output_red) > 0.05:
            cv2.putText(cropped_image, "red and blue", position, font, fontsize, color, thickness, cv2.LINE_AA)
        elif np.mean(output_blue) > 0.05:
            cv2.putText(cropped_image, "blue", position, font, fontsize, color, thickness, cv2.LINE_AA)
        elif np.mean(output_red) > 0.05:
            cv2.putText(cropped_image, "red", position, font, fontsize, color, thickness, cv2.LINE_AA)
        
        cv2.imwrite('./images/res'+str(img_idx)+'.png', cropped_image)
            
        
        if match_found:
            ping += 1
            pong = 0
        if not match_found:
            if ping > 0:
                print(f'frequency: {ping/fps:0.1f}')
            ping = 0
            pong += 1
            
        img_idx += 1

cap.release()