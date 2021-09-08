import os, sys, glob
import onnxruntime
import numpy as np
import cv2, time
import glob
import warnings
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
id2label = {
    0:'no mask',
    1:'mask',
    2:'mask not standard'
}
warnings.filterwarnings("ignore")

image_shape=(640, 640)
video_shape = (320, 320)
mean=(0.485, 0.456, 0.406)
std=(0.229, 0.224, 0.225)

# region utils

def normalize(im):
    im = im / 255.0
    im -= mean
    im /= std
    return im


def colormap(rgb=False):
    """
    Get colormap
    """
    color_list = np.array([
        0.000, 0.447, 0.741, 0.850, 0.325, 0.098, 0.929, 0.694, 0.125, 0.494,
        0.184, 0.556, 0.466, 0.674, 0.188, 0.301, 0.745, 0.933, 0.635, 0.078,
        0.184, 0.300, 0.300, 0.300, 0.600, 0.600, 0.600, 1.000, 0.000, 0.000,
        1.000, 0.500, 0.000, 0.749, 0.749, 0.000, 0.000, 1.000, 0.000, 0.000,
        0.000, 1.000, 0.667, 0.000, 1.000, 0.333, 0.333, 0.000, 0.333, 0.667,
        0.000, 0.333, 1.000, 0.000, 0.667, 0.333, 0.000, 0.667, 0.667, 0.000,
        0.667, 1.000, 0.000, 1.000, 0.333, 0.000, 1.000, 0.667, 0.000, 1.000,
        1.000, 0.000, 0.000, 0.333, 0.500, 0.000, 0.667, 0.500, 0.000, 1.000,
        0.500, 0.333, 0.000, 0.500, 0.333, 0.333, 0.500, 0.333, 0.667, 0.500,
        0.333, 1.000, 0.500, 0.667, 0.000, 0.500, 0.667, 0.333, 0.500, 0.667,
        0.667, 0.500, 0.667, 1.000, 0.500, 1.000, 0.000, 0.500, 1.000, 0.333,
        0.500, 1.000, 0.667, 0.500, 1.000, 1.000, 0.500, 0.000, 0.333, 1.000,
        0.000, 0.667, 1.000, 0.000, 1.000, 1.000, 0.333, 0.000, 1.000, 0.333,
        0.333, 1.000, 0.333, 0.667, 1.000, 0.333, 1.000, 1.000, 0.667, 0.000,
        1.000, 0.667, 0.333, 1.000, 0.667, 0.667, 1.000, 0.667, 1.000, 1.000,
        1.000, 0.000, 1.000, 1.000, 0.333, 1.000, 1.000, 0.667, 1.000, 0.167,
        0.000, 0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000,
        0.000, 0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000,
        0.000, 0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000,
        0.833, 0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.167, 0.000, 0.000,
        0.333, 0.000, 0.000, 0.500, 0.000, 0.000, 0.667, 0.000, 0.000, 0.833,
        0.000, 0.000, 1.000, 0.000, 0.000, 0.000, 0.143, 0.143, 0.143, 0.286,
        0.286, 0.286, 0.429, 0.429, 0.429, 0.571, 0.571, 0.571, 0.714, 0.714,
        0.714, 0.857, 0.857, 0.857, 1.000, 1.000, 1.000
    ]).astype(np.float32)
    color_list = color_list.reshape((-1, 3)) * 255
    if not rgb:
        color_list = color_list[:, ::-1]
    return color_list

#endregion


def img_read(img_path=r'/home/aistudio/PaddleDetection/demo/000000014439.jpg'):

    image = Image.open(img_path).convert('RGB')
    img_array = np.array(image)
    img_array = cv2.resize(img_array,image_shape, cv2.INTER_LINEAR)
    img_array = normalize(img_array)
    image_tp = img_array.transpose(2,0,1)
    image_tp = np.expand_dims(image_tp, 0)
    return image, image_tp


def bbox_post(dt,img_shape):
    x_rate = img_shape[0]/image_shape[0]
    y_rate = img_shape[1]/image_shape[1]
    max_rate = max(x_rate, y_rate)
    num_id, score, x1, y1, x3, y3 = dt.tolist()
    x1 *= x_rate
    x3 *= x_rate
    y1 *= y_rate
    y3 *= y_rate
    x2, y2, x4, y4 = x3, y1, x1, y3
    category_id = int(num_id)
    rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
    dt_res = {
        'category_id': category_id,
        'bbox': rbox,
        'score': score
    }
    return dt_res,max_rate

def bbox_post_video(dt, img_shape):
    x_rate = img_shape[0] / video_shape[0]
    y_rate = img_shape[1] / video_shape[1]
    max_rate = max(x_rate, y_rate)
    num_id, score, x1, y1, x3, y3 = dt.tolist()
    x1 *= x_rate
    x3 *= x_rate
    y1 *= y_rate
    y3 *= y_rate
    x2, y2, x4, y4 = x3, y1, x1, y3
    category_id = int(num_id)
    rbox = [x1, y1, x2, y2, x3, y3, x4, y4]
    dt_res = {
        'category_id': category_id,
        'bbox': rbox,
        'score': score
    }
    return dt_res,max_rate

def draw_bbox_video(image, catid2name, bboxes, fps, threshold=0.6):
    """
    Draw bbox on image
    """
    # image = Image.open(img_path).convert('RGB')
    # image = Image.fromarray(image)
    # image = image.resize((640, 640), Image.ANTIALIAS)
    # draw = ImageDraw.Draw(image)
    #
    # catid2color = {}
    # color_list = colormap(rgb=True)[:40]
    # for dt in np.array(bboxes):
    #     dt = bbox_post(dt)
    #     catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
    #     if score < threshold:
    #         continue
    #
    #     if catid not in catid2color:
    #         idx = np.random.randint(len(color_list))
    #         catid2color[catid] = color_list[idx]
    #     color = tuple(catid2color[catid])
    #
    #     # draw bbox
    #     if len(bbox) == 4:
    #         # draw bbox
    #         xmin, ymin, w, h = bbox
    #         xmax = xmin + w
    #         ymax = ymin + h
    #         draw.line(
    #             [(xmin, ymin), (xmin, ymax), (xmax, ymax), (xmax, ymin),
    #              (xmin, ymin)],
    #             width=2,
    #             fill=color)
    #     elif len(bbox) == 8:
    #         x1, y1, x2, y2, x3, y3, x4, y4 = bbox
    #         draw.line(
    #             [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
    #             width=2,
    #             fill=color)
    #         xmin = min(x1, x2, x3, x4)
    #         ymin = min(y1, y2, y3, y4)
    #     else:
    #         # logger.error('the shape of bbox must be [M, 4] or [M, 8]!')
    #         pass
    #     # draw label
    #     # text = "{} {:.2f}".format(catid2name[catid], score)
    #     text = "{} {:.2f}".format(catid, score)
    #     tw, th = draw.textsize(text)
    #     draw.rectangle(
    #         [(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
    #     draw.text((xmin + 1, ymin - th), text, fill=(255, 255, 255))

    image = Image.fromarray(image)
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        dt,max_rate = bbox_post_video(dt, image.size)
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        if catid not in catid2color:
            catid2color[catid] = color_list[catid]
        color = tuple(catid2color[catid])

        # draw bbox
        if len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            print('the shape of bbox must be [M, 8]!')


        # draw label
        text = "{} {:.2f}".format(id2label[catid], score)
        tw, th = draw.textsize(text)

        tw, th = int(tw*max_rate),int(th*max_rate)
        # "/usr/share/fonts/dejavu-lgc/DejaVuLGCSansCondensed-Bold.ttf"
        # "/Users/admin/Library/Fonts/InputSans-Regular.ttf"
        ft = ImageFont.truetype('C:/windows/fonts/Arial.ttf', int(10*max_rate))

        draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, font = ft, fill=(255, 255, 255))

        draw.text((0, 0), 'FPS:%d'%fps, font = ft, fill=(255, 255, 255))
    return image

def draw_bbox(image, catid2name, bboxes, threshold=0.5):
    draw = ImageDraw.Draw(image)

    catid2color = {}
    color_list = colormap(rgb=True)[:40]
    for dt in np.array(bboxes):
        dt, max_rate = bbox_post(dt,image.size)
        catid, bbox, score = dt['category_id'], dt['bbox'], dt['score']
        if score < threshold:
            continue

        if catid not in catid2color:
            catid2color[catid] = color_list[catid]
        color = tuple(catid2color[catid])

        # draw bbox
        if len(bbox) == 8:
            x1, y1, x2, y2, x3, y3, x4, y4 = bbox
            draw.line(
                [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x1, y1)],
                width=2,
                fill=color)
            xmin = min(x1, x2, x3, x4)
            ymin = min(y1, y2, y3, y4)
        else:
            print('the shape of bbox must be [M, 8]!')

        # draw label
        text = "{} {:.2f}".format(id2label[catid], score)
        tw, th = draw.textsize(text)

        tw, th = int(tw*max_rate),int(th*max_rate)
        # "/usr/share/fonts/dejavu-lgc/DejaVuLGCSansCondensed-Bold.ttf"
        # "/Users/admin/Library/Fonts/InputSans-Regular.ttf"
        ft = ImageFont.truetype('C:/windows/fonts/Arial.ttf', int(10*max_rate))

        draw.rectangle([(xmin + 1, ymin - th), (xmin + tw + 1, ymin)], fill=color)
        draw.text((xmin + 1, ymin - th), text, font = ft, fill=(255, 255, 255))
    return image

def imgs_infer(session, img_list, dst_list):
    for img_path, dst_path in zip(img_list, dst_list):
        img, img_tp = img_read(img_path)
        inputs = {
            session.get_inputs()[0].name: np.array(image_shape).reshape([-1, 2]).astype(np.float32),
            session.get_inputs()[1].name: img_tp.astype(np.float32),
            session.get_inputs()[2].name: np.array([1, 1]).reshape([-1, 2]).astype(np.float32),
        }
        outputs = session.run(None, inputs)
        bboxes, bbox_n = outputs
        img_draw = draw_bbox(img, None, bboxes)

        img_draw.save(dst_path)
        print(img_path)

def imgs_infer_video(session, singleimg):
    start = time.time()
    image = singleimg
    img_array = np.array(image)
    img_array = cv2.resize(img_array, video_shape, cv2.INTER_LINEAR)
    img_array = normalize(img_array)
    image_tp = img_array.transpose(2, 0, 1)
    image_tp = image_tp.reshape(1, image_tp.shape[0], image_tp.shape[1], image_tp.shape[2])
    img, img_tp = image, image_tp
    inputs = {
        session.get_inputs()[0].name: np.array(video_shape).reshape([-1, 2]).astype(np.float32),
        session.get_inputs()[1].name: img_tp.astype(np.float32),
        session.get_inputs()[2].name: np.array([1, 1]).reshape([-1, 2]).astype(np.float32),
    }
    outputs = session.run(None, inputs)
    bboxes, bbox_n = outputs
    end = time.time()
    seconds = end - start
    fps = 1 / seconds
    img_draw = draw_bbox_video(img, None, bboxes, fps)
    return img_draw

def path2list(img_path, dst_path):
    if os.path.isfile(img_path):
        img_list = [img_path]

        if os.path.isfile(dst_path):
            dst_list = [dst_path]
        elif os.path.isdir(dst_path):
            dst_list = [os.path.join(dst_path, os.path.basename(img_path))]
        else:
            print('input is a file, while output path is not a file!')
            return None, None

    elif os.path.isdir(img_path):
        img_list = glob.glob(os.path.join(img_path), '*.jpg') + glob.glob(os.path.join(img_path), '*.png') + glob.glob(
            os.path.join(img_path), '*.tif')
        if os.path.isdir(dst_path):
            dst_list = [img_file.replace(img_path, dst_path) for img_file in img_list]
        else:
            print('input is a dir, while output path is not a dir!')
            return None, None
    else:
        return None, None
    return img_list, dst_list


if __name__ == '__main__':
    img_path = sys.argv[1]
    dst_path = sys.argv[2]

    img_list, dst_list = path2list(img_path, dst_path)

    # session = model_init()

    # imgs_infer(session, img_list, dst_list)

    # img, img_tp = img_read()
    # outputs = img_infer(img_tp)
    # post_process(img, outputs)
