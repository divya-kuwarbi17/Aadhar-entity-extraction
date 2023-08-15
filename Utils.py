import re
import time
from datetime import datetime
import base64
import json
from PIL import Image
import numpy as np
import pandas as pd
from io import BytesIO
import cv2
# import pytesseract
import matplotlib.pyplot as plt



def getTime(date_time=False):
    """
    Args:
    date_time: boolean. If True, this function will return
    two variables, time and datetime string, else, only time will be returned.
    """
    s = time.time()
    if date_time:
        dt = datetime.now()
        dt = dt.strftime("%d/%m/%Y %H:%M:%S")
        return s, dt
    return s


def jsonToImage(data=None):

    try:
        d1 = data['image_byte_string']
        enc = d1.encode()
        img_bytestring = base64.b64decode(enc)
        image = Image.open(BytesIO(img_bytestring))
        return image

    except Exception as e:
        raise ValueError(f"Error in JSON2IMG: {str(e)}")


def add_margin(pil_img, top, right, bottom, left, color):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

# cicular stamp


def circular(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    output = img.copy()

    output2 = img.copy()

    # detect circles in the image
    circles = cv2.HoughCircles(gray, method=cv2.HOUGH_GRADIENT, dp=3, minDist=400 )
    mask = np.zeros((img.shape[0],img.shape[1]), np.uint8)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            a=cv2.circle(output, (x, y), r, (255, 255,255), -1)
            b=cv2.circle(output2, (x, y), r-70, (255,255,255), -1)
            mask = cv2.subtract(a,b)

            _,mask=cv2.threshold(mask,20,100,cv2.THRESH_BINARY_INV) # 10,100
            mask=cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            mask_copy=mask.copy()
        # cv2.imwrite('./Output/circled_stamp.jpeg', mask)
    #delete unnecessary variable
    del output
    del output2
    del img

    # threshold the obtained mask for further processing where we will find bbox of actual stamp in the image
    vec=np.vectorize(lambda p: 0 if p > 60 else 255)

    mask=vec(mask)

    ymin,ymax=np.min(np.where(mask.sum(axis=1))),np.max(np.where(mask.sum(axis=1)))

    xmin,xmax=np.min(np.where(mask.sum(axis=0))),np.max(np.where(mask.sum(axis=0)))

    #Apply cropping using above coordinates on original copy of the mask i.e output
    cropped= mask_copy[ymin:ymax,xmin:xmax]

    #straighten the circular text
    polar_image = cv2.linearPolar(src = cropped, center = (cropped.shape[0]/2, cropped.shape[1]/2), maxRadius = 150.0, flags= cv2.INTER_CUBIC + cv2.WARP_POLAR_LINEAR)#cv2.WARP_POLAR_LINEAR+cv2.WARP_FILL_OUTLIERS)
    polar_image = polar_image.astype(np.uint8)
    rotated = cv2.rotate(polar_image, cv2.ROTATE_90_COUNTERCLOCKWISE)

    # Add padding and squeeze height of image for better ocr
    img = Image.fromarray(rotated)
    img = img.resize((500,101), Image.Resampling.LANCZOS)
    img= add_margin(img, 40, 40, 40, 40, 255)
    img = img.point( lambda p: 255 if p > 60 else 0 ) 
    # img.save('c.jpeg')
    
    return pytesseract.image_to_string(img,config='--psm 1')


def linear(imgf):

    # gray = cv2.cvtColor(imgf, cv2.COLOR_BGR2GRAY)
    # _,thresh=cv2.threshold(gray,130,255,cv2.THRESH_BINARY)

    # kernel = np.ones((3, 3), np.uint8)
    # thresh = cv2.erode(thresh, kernel, iterations=1)
    # thresh= cv2.medianBlur(thresh, 3)
    # cv2.imwrite('./Output/linear.jpeg',thresh)notinc
    custom_config = r'-c preserve_interword_spaces=1 --oem 1 --psm 4'
    return pytesseract.image_to_string(thresh,config = custom_config)

def encoder(image):
    """
    Convert image to base64 byte string.
    """
    ENCODING = 'utf-8'
    buf = BytesIO()
    image.save(buf, format='JPEG')
    byte_content = buf.getvalue()

    base64_bytes = base64.b64encode(byte_content)
    base64_string = base64_bytes.decode(ENCODING)
    
    return base64_string

