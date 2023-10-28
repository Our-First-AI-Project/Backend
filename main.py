"""웹 라이브러리"""
from flask import Flask, request
from flask_restful import Api
from flask_cors import CORS
import json
import urllib3

"""인공지능 라이브러리"""
from tensorflow.keras.models import load_model
import numpy as np

"""이미지 라이브러리"""
import cv2
import requests
import base64

# InsecureRequestWarning 경고 제거
# https://urllib3.readthedocs.io/en/latest/advanced-usage.html#tls-warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

app = Flask(__name__)
api = Api(app)
# TODO : CORS 좀 더 구체적으로 설정하기
CORS(app)

# 모델 불러오기
model = load_model('./binary_model_saved180.h5')

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

"""
image_to_binary : url을 통해 이미지를 열고 bytearray 형태로 변환한다.
Args :
    url : 이미지를 가져올 url
Returns :
    image : bytearray 타입의 이미지 데이터
    "ad" : gif 형식의 이미지인 경우 -> Not Error
    "non-ad" : svg 형식의 이미지인 경우 -> Not Error
"""
def image_to_binary(url):
    if (".gif" in url):
        # 이미지가 gif 형식일 경우 -> 제거한다.
        # TODO : 더 좋은 방법을 고민해보기
        return "ad"
    if (".svg" in url):
        # 이미지가 svg 형식일 경우 -> 기호일 가능성이 높으므로 제거하지 않는다.
        return "non-ad"
    if (".htm" in url):
        # html 형식일 경우 -> 제거하면 안된다.
        return "non-ad"

    # 이 외의 경우에는 url을 통해 이미지를 가져온다.
    try:
        # 1. data:image 형식의 이미지를 가져오는 경우
        if (url.startswith("data:image")):
            encoded_data = url.split(',')[1].replace(" ", "").replace("\n", "")
            if (len(encoded_data) % 4 != 0):
                encoded_data += "=" * (4 - len(encoded_data) % 4)
            image_data = base64.b64decode(encoded_data)
            return bytearray(image_data) # bytearry 형식으로 변환

        # 2. 일반적인 url 형식의 이미지를 가져오는 경우
        image = requests.get(url, verify=False).content
        return bytearray(image) # bytearry 형식으로 변환
    
    except Exception as e:
        # 비어있거나 열 수 없는 이미지 파일인 경우 -> 제거하지 않는다. (path-error)
        # data:image 형식의 이미지에서 열 수 없는 경우가 있음
        # 잘못된 경로를 입력하는 경우는 아직 발견되지 않음.
        return "non-ad"

def binary(url, model):
    
    image_width = 180
    image_height = 180

    is_ad = 0
    not_ad = 0

    X = []
    cropped_images = []

    binary_image_data = image_to_binary(url)

    if (binary_image_data == "ad" or binary_image_data == "non-ad"):
        return binary_image_data

    image_nparray = np.asarray(binary_image_data, dtype=np.uint8)
        
    # 응답이 204 No Content인 경우 -> 제거하지 않는다. (open-size-zero-error)
    # google adsense에서 가져온 이미지가 204 No Content인 경우가 있음
    if image_nparray.size == 0:
        return "non-ad"
    
    # binary 형태로 읽은 파일을 decode -> 1D-array에서 3D-array로 변경
    image_bgr = cv2.imdecode(image_nparray, cv2.IMREAD_COLOR)
    
    # 이미지가 확인되지 않는 경우 -> 제거하지 않는다. (open-none-error)
    # 몇몇 url에서 이미지가 아니거나, 1*1 픽셀의 이미지를 가져오는 경우가 있음
    if image_bgr is None:
        return "non-ad"

    # 이미지가 너무 작은 경우 -> 제거하지 않는다. (small-image-error)
    if image_bgr.shape[0] < 64 | image_bgr.shape[1] < 64:
        return "non-ad"

    # BGR에서 RGB로 변경
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    img = cv2.resize(image_rgb, (image_width, image_height), interpolation=cv2.INTER_LINEAR)
    cropped_images = image_crop(img, image_width, image_height, image_width // 2, image_height //2)

    for cropped_image in cropped_images:
        data = np.asarray(cropped_image)
        X.append(data)

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
        return "ad"
    else:
        return "non-ad"
        
@app.route("/",methods=['GET','POST'])
def check():
    if request.method=='POST':
        try:
            url = request.get_json()['url']
            result = binary(url, model)
            response_body = {"class": result}
            response = json.dumps(response_body, ensure_ascii=False)
            return response, 200
        except Exception as e:
            print(e)
            return "Internal Server Error", 500
    return "Method Not Allowed", 405

if __name__ == '__main__':
    app.run(debug=False, host= '0.0.0.0', port='5000')
