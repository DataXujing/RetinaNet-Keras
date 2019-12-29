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


def predict_img(model, test_img_fold, test_img_list):
    # load image
    img_name_list = []
    bboxes_list = []
    class_list = []
    score_list = []
    for i in range(len(test_img_list)):
        img_name = test_img_list[i]
        if ".xml" in img_name:
            continue
        img_path = os.path.join(test_img_fold, img_name)
        # image = read_image_bgr(img_path)
        image_BGR = cv2.imread(img_path)
        # copy to draw on
        draw = image_BGR
        image = cv2.cvtColor(image_BGR, cv2.COLOR_BGR2RGB) # 不需要
        image = np.asarray(image)[:, :, ::-1].copy()
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
            if score < 0.2:
               break
            color = label_color(label)
            b = box.astype(int)
            img_name_list.append(img_name)
            bboxes_list.append(b)
            class_list.append(labels[0][i])
            score_list.append(score)
            i += 1
            draw_box(draw, b, color=color)
            print(label)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        # imsave('result/'+img_name, draw)
        # print(draw)
        cv2.imwrite('result/'+img_name, draw)
    submit = pd.DataFrame()
    submit['img_name'] = img_name_list
    submit['bbox'] = bboxes_list
    submit['class'] = class_list
    submit['score'] = score_list
    submit.to_csv('submit.csv', index=None)
    # submit.to_pickle('submit.pkl')


if __name__ == "__main__":
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

    model_path = "./checkpoint/resnet101_csv_39.h5"
    test_img_fold = "./test_data"
    test_img_list = os.listdir(test_img_fold)
    # print(len(test_img_list)/2)

    # load retinanet model
    print("[info] wait seconds to load and transfer model!")
    model = models.load_model(model_path, backbone_name='resnet101')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model   
    # 训练好的模型需要转化成推断网络： 
    # keras_retinanet/bin/convert_model.py /path/to/training/model.h5 /path/to/save/inference/model.h5
    # model = models.convert_model(model)
    model = models.convert_model(model)

    # print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'hat', 1: 'person',-1:'NaN'}


    predict_img(model, test_img_fold, test_img_list)
