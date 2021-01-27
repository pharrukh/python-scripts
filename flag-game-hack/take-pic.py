import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

try:
    import Image
except ImportError:
    from PIL import Image
import pytesseract

import json
from fuzzywuzzy import fuzz

with open('countries.json') as f:
    countries = json.load(f)

def get_potential_country(recognized_country_name):
    max_score = 0
    potential_country = {}

    for country in countries:
        country_name = country['name']
        result = fuzz.ratio(recognized_country_name, country_name)
        if result > 50:
            if max_score < result:
                max_score = result
                potential_country = country
    return (max_score, potential_country)


def get_flag(path):
    flag_img = mpimg.imread(flag_path, cv2.IMREAD_UNCHANGED)
    h,w = flag_img.shape[:2]
    resized_image = cv2.resize(flag_img, (h,w))
    new_img = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)
    return new_img

def add_text_to_image(img, text):
    # font 
    font = cv2.FONT_HERSHEY_COMPLEX 
    # org 
    org = (50, 50) 
    # fontScale 
    fontScale = 1
    # Blue color in BGR 
    color = (255, 0, 0) 
    # Line thickness of 2 px 
    thickness = 2
    # Using cv2.putText() method
    return cv2.putText(frame, result[1]['name'], org, font, fontScale, color, thickness, cv2.LINE_AA) 

def add_image_to_image(img_large, img_small):
    h1, w1 = resized_image.shape[:2]
    pip_h = 50
    pip_w = 50
    frame[pip_h:pip_h+h1,pip_w:pip_w+w1] = resized_image
    return frame

cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

resized_image = {}

while True:
    ret, frame = cap.read()
    frame = cv2.resize(frame, None, fx=1, fy=1, interpolation=cv2.INTER_AREA)
    grayFrame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    img_rgb = cv2.cvtColor(grayFrame, cv2.COLOR_BGR2RGB)

    path_to_trained_data = '/Users/farrukhnormuradov/dev/flag-game-hack/tessdata'
    recognized_text = pytesseract.image_to_string(img_rgb, lang='rus', config='--tessdata-dir ' + path_to_trained_data)
    result = get_potential_country(recognized_text)
    if result[0] != 0:
        flag_path = f'flags/{result[1]["alpha2"]}.png'
        resized_image = get_flag(flag_path)
        frame = add_image_to_image(frame, resized_image)
        image = add_text_to_image(frame, recognized_text)
        cv2.imshow('Input', image)
    else:
        cv2.imshow('Input', frame)

    c = cv2.waitKey(60)
    if c == 27:
        break

cap.release()
cv2.destroyAllWindows()