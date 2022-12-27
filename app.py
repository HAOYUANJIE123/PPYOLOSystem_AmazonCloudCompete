# coding:utf-8

from werkzeug.utils import secure_filename
import os
import numpy as np
from flask import Flask, render_template, Response
import cv2
import onnxruntime
from flask import Flask, render_template, request, redirect, url_for
from deploy import imgs_infer, imgs_infer_video

app = Flask(__name__)


@app.route('/')
def main_UI():

    return render_template('index.html')


needPredic = False
imgFile = ''
fileName = ''
basepath = os.path.dirname(__file__)  # 当前文件所在路径
modepath='static/ppyolo_tiny.onnx'
# session_img = onnxruntime.InferenceSession('static/ppyolov2.onnx')
session_img = onnxruntime.InferenceSession(modepath)
session_video = onnxruntime.InferenceSession(modepath)

@app.route('/upload', methods=['POST', 'GET'])
def upload():
    global imgFile
    global fileName
    if request.method == 'POST':
        f = request.files['file']
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        filename = f.filename
        upload_path = os.path.join(basepath, 'static/uploads')
        upload_file = os.path.join(upload_path, secure_filename(filename))
        clearDir(upload_path)
        f.save(upload_file)
        src = cv2.imread(upload_file, 1)
        cv2.imwrite(upload_file, src, [cv2.IMWRITE_JPEG_QUALITY, 50])

        imgFile = upload_file
        fileName = filename
        oraImg = os.path.join('../static/uploads/', filename)
        return {"oraImg": oraImg}


@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        basepath = os.path.dirname(__file__)  # 当前文件所在路径
        pre_path = os.path.join(basepath, 'static/preResult')
        clearDir(pre_path)
        preimg(imgfile=imgFile)
        preImg = os.path.join('../static/preResult', fileName)
        return {"preImg": preImg}




def preimg(imgfile=''):
    imgs_infer(session_img, [imgfile], [imgfile.replace('uploads', 'preResult')])


def clearDir(upload_path):
    if os.path.exists(upload_path):
        for i in os.listdir(upload_path):
            path_file = os.path.join(upload_path, i)
            if os.path.isfile(path_file):
                os.remove(path_file)


# 这个地址返回视频流响应
@app.route('/video_feed')
def video_feed():
    return Response(gen(VideoCamera(), False), mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/video_feed1')
def video_feed1():
    return Response(gen(VideoCamera(), True), mimetype='multipart/x-mixed-replace; boundary=frame')


class VideoCamera(object):
    def __init__(self):
        # 通过opencv获取实时视频流
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        self.video.release()

    def get_frame(self, needPredict):
        success, image = self.video.read()
        # 在这里处理视频帧
        if not needPredict:
            cv2.putText(image, "None", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0))
        else:
            image = predictVideoImg(image)
        # 因为opencv读取的图片并非jpeg格式，因此要用motion JPEG模式需要先将图片转码成jpg格式图片
        ret, jpeg = cv2.imencode('.jpg', image)
        return jpeg.tobytes()


def gen(camera, needPredict):
    while True:
        frame = camera.get_frame(needPredict)
        # 使用generator函数输出视频流， 每次请求输出的content类型是image/jpeg
        yield (b'--frame\r\n'               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')


def predictVideoImg(singleImg):
    img = imgs_infer_video(session_video, singleImg)
    img = np.array(img)
    return img


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
