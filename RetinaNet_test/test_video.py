# pip install keras-resnet --user

# import miscellaneous modules
# import matplotlib.pyplot as plt
# from skimage.io import imsave
import numpy as np
import pandas as pd
import tensorflow as tf
import keras
import os
import time
import cv2

from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import (preprocess_image, read_image_bgr,
                                         resize_image)
from keras_retinanet.utils.visualization import draw_box, draw_caption

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if not os.path.exists('result'):
    os.mkdir('result')

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


def predict_video(model, video_path):
    # load image
    img_name_list = []
    bboxes_list = []
    class_list = []
    score_list = []

    cap = cv2.VideoCapture(video_path) 


    # 保存视频成mp4
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
    sz = (int(cap.get(3)),int(cap.get(4)))
    fps = int(cap.get(5))
    video_out = cv2.VideoWriter()
    video_out.open('result/output.mp4',fourcc,fps,sz)

    while(cap.isOpened()): 
        ret, image_BGR = cap.read() 
        draw = image_BGR
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) # 不需要
        image = np.asarray(image)[:, :, ::-1].copy()
        # draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB) # 不需要
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        # print(image.shape)
        # print(scale)
        # boxes: 预测的box,scores:预测的概率，labels预测的labels
        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        # correct for image scale
        boxes /= scale
        i = 0
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break,概率小于0.5的被舍弃
            if score < 0.5:
                break
            color = label_color(label)
            b = box.astype(int)
            bboxes_list.append(b)
            class_list.append(labels[0][i])
            score_list.append(score)
            i += 1
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw = draw_caption(draw, b, caption)

        video_out.write(draw)
        cv2.imshow('RetinaNet-test', draw) 
        k = cv2.waitKey(20) 
        #q键退出
        if (k & 0xff == ord('q')): 
            break 

    cap.release() 
    video_out.release()
    cv2.destroyAllWindows()
     
    # submit = pd.DataFrame()
    # submit['img_name'] = img_name_list
    # submit['bbox'] = bboxes_list
    # submit['class'] = class_list
    # submit['score'] = score_list
    # submit.to_csv('submit.csv', index=None)
    # submit.to_pickle('submit.pkl')


if __name__ == "__main__":
    keras.backend.tensorflow_backend.set_session(get_session())
    model_path = "./checkpoint/xxx.h5"
    video_path = "./test.mp4"
    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet101')
    model = models.convert_model(model)
    # print(model.summary())
    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'hat', 1: 'person'}
    
    predict_video(model, video_path)
