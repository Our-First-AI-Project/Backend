"""웹 라이브러리"""

from flask import Flask, jsonify, request
from flask_restful import Resource, Api
from flask_cors import CORS
import json

"""인공지능 라이브러리"""

# import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np

"""이미지 라이브러리"""

import cv2
import urllib.request
import requests
from io import BytesIO

app = Flask(__name__)
api = Api(app)
CORS(app)

model = load_model('./binary_cnn/binary_model_saved0.h5') # 경로 수정

def image_crop(file, image_width, image_height, cropped_width, cropped_height):
    cropped_images = []
    x = 0
    x_center = image_width // 2
    y_center = image_height // 2
    
    while x < image_width:
        y = 0
        while y < image_height:
            cropped_image = file[x : x + cropped_width, y : y + cropped_height]
            cropped_images.append(cropped_image)
            y += cropped_height
        x += cropped_width
    
    center_image = file[x_center - cropped_width // 2 : x_center + cropped_width // 2,
                       y_center - cropped_height // 2 : y_center + cropped_height // 2]
    
    cropped_images.append(center_image)
    
    return cropped_images

def binary(url, model):
    
    image_width = 180
    image_height = 180

    is_ad = 0
    not_ad = 0
    
    X = []
    cropped_images = []
    
    # 이미지가 명백히 gif 형식일 경우 -> 무조건 제거한다.
    if ".gif" in url:
        return "absolute"
    # 이미지가 명백히 .svg 형식일 경우 -> ?
    elif ".svg" in url:
        return "non-ad"
    # 이 외의 경우에는 이미지를 연다. (binary)
    else:
        try:
            image_nparray = np.asarray(bytearray(requests.get(url, verify=False).content), dtype=np.uint8)
        # 이미지 경로가 잘못된 경우 제거한다.
        except:
            return "absolute"
    # 보안 문제로 인해 열리지 않는 경우 제거
    if image_nparray.size == 0:
        return "absolute"
    # binary 형태로 읽은 파일을 decode -> 1D-array에서 3D-array로 변경
    image_bgr = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    # 보안 문제로 인해 열리지 않는 경우 제거
    if image_bgr is None:
        return "absolute"
    # BGR에서 RGB로 변경
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    
    # 이미지가 너무 작은 경우 -> 무조건 제거한다.
    if image_rgb.shape[0] < 64 | image_rgb.shape[1] < 64:
        return "absolute"

    img = cv2.resize(image_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)

    cropped_images = image_crop(img, image_width, image_height, image_width // 2, image_height //2)

    for cropped_image in cropped_images:
        data = np.asarray(cropped_image)
        X.append(data)
        
    #data = np.asarray(img)
    #X.append(data)

    X = np.array(X)
    X = X.astype(float) / 255

    prediction = model.predict(X)
    prediction = np.round(prediction)

    for p in prediction:
        if p == 0:
            is_ad += 1
        elif p ==1:
            not_ad += 1
    
    if is_ad > not_ad:
        return 0
    else:
        return 1
        
@app.route("/",methods=['GET','POST'])
def check():
    if request.method=='GET':
        
        url = request.args.get('url')
        result = binary(url, model)
        
        if result == 0:           ## result 수정
            config = {"class": "ad"}
        elif result == "absolute":
            config = {"class": "absolute"}
        elif result == "non-ad":
            config = {"class": "non-ad"}
        else:
            config = {"class": "non-ad"}

        result = json.dumps(config, ensure_ascii=False)
        
        return result

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0', port='4000')
