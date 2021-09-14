import os 
import time

import numpy as np
import torch
import cv2

from src.utils import get_config
from src.libs.CRAFT.craft import CRAFT
from src.libs.DeepText.Deeptext_pred import Deeptext_predict, load_model_Deeptext

from src.src import craft_text_detect, load_model_Craft
from src.src import yolo_detect
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# setup flask
from flask import Flask, jsonify, request, json
from flask_cors import CORS

from flask import Flask,flash, render_template, Response, request, session, redirect, url_for, jsonify
from flask import send_from_directory ,send_file
# from flask_restful import Resource, Api
from flask_cors import CORS
from werkzeug.utils import secure_filename
from itsdangerous import (TimedJSONWebSignatureSerializer as Serializer, BadSignature, SignatureExpired)
from configparser import SafeConfigParser
import rcode

import base64
from io import BytesIO
from PIL import Image
import numpy as np
import cv2
import io
import base64
import shutil
import glob
import datetime
import time
app = Flask(__name__)
CORS(app)

# setup config
cfg = get_config()
cfg.merge_from_file('./src/configs/pipeline.yaml')
cfg.merge_from_file('./src/configs/craft.yaml')
cfg.merge_from_file('./src/configs/Deeptext_vie.yaml')

DEEPTEXT_CONFIG = cfg.DEEPTEXT
CRAFT_CONFIG = cfg.CRAFT
NET_CRAFT = CRAFT()
PIPELINE_CFG = cfg.PIPELINE

# load all model
# model text detct
print ('[LOADING] Text detecttion model')
CRAFT_MODEL = load_model_Craft(CRAFT_CONFIG, NET_CRAFT)
print ('[LOADING SUCESS] Text detection model')
# model regconition
print ('[LOADING] Text regconition model')
DEEPTEXT_MODEL, DEEPTEXT_CONVERTER = load_model_Deeptext(DEEPTEXT_CONFIG)
print ('[LOADING SUCESS] Text regconition model')

def regconition(cfg, img):

    # predict region of text bounding box
    bboxes, polys, score_text = text_detect_CRAFT(img, CRAFT_CONFIG, CRAFT_MODEL, PIPELINE_CFG.Y_DIST_FOR_MERGE_BBOX,  PIPELINE_CFG.EXPAND_FOR_BBOX)
    bboxes = sorted_boxes(bboxes)
    texts = []
    # count = 1
    #   shutil.rmtree('./reg')
    folder_name = './reg_{}/'.format(datetime.datetime.now().strftime("%Y%m%d%H%M%S"))
    if not os.path.exists(folder_name) :
        os.mkdir(folder_name)
    for index, bbox in enumerate(bboxes):
        # merge bbox on a line
        if bbox[0][0] < 0: bbox[0][0] = 0
        if bbox[0][1] < 0: bbox[0][1] = 0
        if bbox[1][0] < 0: bbox[1][0] = 0
        if bbox[1][1] < 0: bbox[1][1] = 0
        img_reg = img[int(bbox[0][1]):int(bbox[2][1]), int(bbox[0][0]):int(bbox[2][0])]
        # img_reg = improve_resolution(img_reg, super_resolution_model)
        cv2.imwrite(folder_name+'img_{}_reg.jpg'.format(index), img_reg)
        # texts.append(text)
    text = text_recog(cfg, DEEPTEXT_CONFIG, folder_name, DEEPTEXT_MODEL, DEEPTEXT_CONVERTER)
    # shutil.rmtree('./reg')
    # full_text = ''.join(texts)
    return text
    

# Take in base64 string and return cv image
def stringToRGB(base64_string):
    imgdata = base64.b64decode(str(base64_string))
    image = Image.open(io.BytesIO(imgdata))
    return cv2.cvtColor(np.array(image), cv2.COLOR_BGR2RGB)
    
@app.route("/", methods=['GET', 'POST'])
def index():
    return render_template('upload.html')

def convert_json_data(data):
    '''
    Return as RGB order
    '''
    imgString64 = data["src"].split(',')[-1].encode()
    im_bytes = base64.b64decode(imgString64)
    im_file = BytesIO(im_bytes)
    img = Image.open(im_file).convert('RGB')
    return np.array(img,dtype = np.uint8)

@app.route('/api/predict', methods=['POST'])
def predictT():
    if request.method == 'POST':
        queryImg_json = json.loads(request.form.get('queryImage'))
        queryImg = convert_json_data(queryImg_json)

        print('-'*20,"Here",'-'*20)
        # request_data = json.loads(request.get_data().decode('utf-8'))
        # image = stringToRGB(request_data['src'])
        start = time.time()
        result_detect = regconition(cfg, queryImg)
        end = time.time()
        print ("Time inference is: ", end - start,"s")
        # print (result_detect)
        return_result = {'code': '1000', 'status': rcode.code_1000, 'data': {'text': result_detect , 
                                # 'img_src': data_base64,
                                'predTime (s)': int(end-start)
                                }}
        return jsonify(return_result)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port="8001", debug=True)
