# pip install keras-resnet --user
import os
import time

import keras
# import miscellaneous modules
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# set tf backend to allow memory to grow, instead of claiming everything
import tensorflow as tf
from skimage.io import imsave

import cv2
from keras_retinanet import models
from keras_retinanet.utils.colors import label_color
from keras_retinanet.utils.image import (preprocess_image, read_image_bgr,
                                         resize_image)
from keras_retinanet.utils.visualization import draw_box, draw_caption

if not os.path.exists('result'):
    os.mkdir('result')


def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)


# os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def predict_save(model, test_img_fold, test_img_list):
    # load image
    img_name_list = []
    bboxes_list = []
    class_list = []
    score_list = []
    for i in range(len(test_img_list)):
        # for i in range(1):
        img_name = test_img_list[i]
        img_path = os.path.join(test_img_fold, img_name)
        image = read_image_bgr(img_path)
        # copy to draw on
        draw = image.copy()
        draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)
        # preprocess image for network
        image = preprocess_image(image)
        image, scale = resize_image(image)
        # process image
        start = time.time()
        # print(image.shape)
        # print(scale)
        boxes, scores, labels = model.predict_on_batch(
            np.expand_dims(image, axis=0))
        print("processing time: ", time.time() - start)
        # correct for image scale
        boxes /= scale
        i = 0
        for box, score, label in zip(boxes[0], scores[0], labels[0]):
            # scores are sorted so we can break
            if score < 0.5:
                break
            color = label_color(label)
            b = box.astype(int)
            img_name_list.append(img_name)
            bboxes_list.append(b)
            class_list.append(labels[0][i])
            score_list.append(score)
            i += 1
            draw_box(draw, b, color=color)
            caption = "{} {:.3f}".format(labels_to_names[label], score)
            draw_caption(draw, b, caption)
        imsave('result/'+img_name, draw)
    submit = pd.DataFrame()
    submit['img_name'] = img_name_list
    submit['bbox'] = bboxes_list
    submit['class'] = class_list
    submit['score'] = score_list
    # submit.to_csv('submit.csv', index=None)
    submit.to_pickle('submit.pkl')


if __name__ == "__main__":
    # set the modified tf session as backend in keras
    keras.backend.tensorflow_backend.set_session(get_session())

    # adjust this to point to your downloaded/trained model
    # models can be downloaded here: https://github.com/fizyr/keras-retinanet/releases

    model_path = os.path.join('snapshots', 'old.h5')

    # load retinanet model
    model = models.load_model(model_path, backbone_name='resnet50')

    # if the model is not converted to an inference model, use the line below
    # see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
    model = models.convert_model(model)

    # print(model.summary())

    # load label to names mapping for visualization purposes
    labels_to_names = {0: 'tieke', 1: 'heiding',
                       2: 'daoju', 3: 'dian', 4: 'jiandao'}

    test_img_fold = 'keras_retinanet/CSV/data/jinnan2_round1_test_a_20190306/'
    test_img_list = os.listdir(test_img_fold)
    print(len(test_img_list))
    predict_save(model, test_img_fold, test_img_list)
