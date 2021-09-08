# coding:utf-8
from flask import Flask, render_template, request, Response, jsonify
# from flask_cors import CORS
import base64
import numpy
import json
import cv2
app = Flask(__name__)
# CORS(app, supports_credentials=True)


@app.route('/')
def main_UI():

    return render_template('index1.html')


@app.route('/video_sample/')
def video_sample():
    return render_template('camera.html')


@app.route('/receiveImage/', methods=["POST"])
def receive_image():
    if request.method == "POST":
        data = request.data.decode('utf-8')
        json_data = json.loads(data)
        str_image = json_data.get("imgData")
        img = base64.b64decode(str_image)
        img_np = numpy.fromstring(img, dtype='uint8')
        new_img_np = cv2.imdecode(img_np, 1)
        # cv2.imshow(new_img_np)
        cv2.imwrite('./images/rev_image.jpg', new_img_np)
        print('data:{}'.format('success'))

    return Response('upload')


if __name__ == '__main__':
    app.run(debug=True, host='127.0.0.1', port=5000)
