import keras

# import keras_retinanet
from keras_retinanet import models
from keras_retinanet.utils.image import read_image_bgr, preprocess_image, resize_image
from keras_retinanet.utils.visualization import draw_box, draw_caption
from keras_retinanet.utils.colors import label_color

# import miscellaneous modules
import matplotlib.pyplot as plt
import cv2
import os
import numpy as np
import time

# GPU Flag
#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# 配置GPU资源
import tensorflow as tf

def get_session():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return tf.Session(config=config)



# Keras配置资源
keras.backend.tensorflow_backend.set_session(get_session())
model_path = os.path.join('.', 'snapshots', 'resnet50_coco_best_v2.1.0.h5')
# 加载RetinaNet模型
model = models.load_model(model_path, backbone_name='resnet50')

# 如果模型没有转换为Inference模式，可以参考下面运行代码
# see: https://github.com/fizyr/keras-retinanet#converting-a-training-model-to-inference-model
#model = models.convert_model(model)

# 打印模型结构
print(model.summary())

# label:name 映射,coco数据集一共80个类别
labels_to_names = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train',\
 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', \
 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant',\
 21: 'bear', 22: 'zebra', 23: 'giraffe', 24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', \
 28: 'suitcase', 29: 'frisbee', 30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', \
 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', \
 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 45: 'bowl', 46: 'banana',\
 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 51: 'carrot', 52: 'hot dog', 53: 'pizza', \
 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 59: 'bed', \
 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard',\
 67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book',\
 74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'}



def pred_img(img_path):

	# 加载数据
	image = read_image_bgr(img_path)

	# 图像copy
	draw = image.copy()
	# draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

	# 输入网络准备
	image = preprocess_image(image)
	image, scale = resize_image(image)

	# 图像 inference
	start = time.time()
	boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
	print("processing time: ", time.time() - start)

	# 映射到原图
	boxes /= scale

	# 可视化检测结果
	for box, score, label in zip(boxes[0], scores[0], labels[0]):
	    # 是排序额score
	    if score < 0.5:
	        break
	        
	    color = label_color(label)
	    
	    b = box.astype(int)
	    draw_box(draw, b, color=color)
	    
	    caption = "{} {:.3f}".format(labels_to_names[label], score)
	    draw_caption(draw, b, caption,color=color)
	    font = cv2.FONT_HERSHEY_SIMPLEX
	    cv2.putText(draw,"RetinaNet GPU:NVIDIA Tesla V100 32GB by XJ",(50,50),font,1.2,(0,255,0),3)
	    cv2.imwrite("./images/test_result.png",draw)
	


	# cv2.imshow('result.jpg',draw)
	cv2.waitKey(0)
	cv2.destroyAllWindows()



def pred_video(video_path):
	cap = cv2.VideoCapture(video_path)
	width = cap.get(3)   
	height = cap.get(4)   
	fps = cap.get(5)  
	print((width,height))


	fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
	writer = cv2.VideoWriter()
	writer.open('./images/output.mp4',fourcc,int(fps),(int(width),int(height)))


	while(cap.isOpened()):
		# Capture frame-by-frame
		ret, frame = cap.read()
		if ret == True:

			# 图像copy
			draw = frame.copy()
			# draw = cv2.cvtColor(draw, cv2.COLOR_BGR2RGB)

			# 输入网络准备
			image = preprocess_image(frame)
			image, scale = resize_image(image)

			# 图像 inference
			start = time.time()
			boxes, scores, labels = model.predict_on_batch(np.expand_dims(image, axis=0))
			print("processing time: ", time.time() - start)

			# 映射到原图
			boxes /= scale

			# 可视化检测结果
			for box, score, label in zip(boxes[0], scores[0], labels[0]):
			    # 是排序额score
			    if score < 0.5:
			        break
			        
			    color = label_color(label)
			    
			    b = box.astype(int)
			    draw_box(draw, b, color=color)
			    
			    caption = "{} {:.3f}".format(labels_to_names[label], score)
			    draw_caption(draw, b, caption,color=color)

			font = cv2.FONT_HERSHEY_SIMPLEX
			cv2.putText(draw,"Model:RetinaNet GPU:NVIDIA Tesla V100 32GB by XJ",(40,50),font,0.6,(0,0,255),2)
			cv2.imshow('result.jpg',draw)
			writer.write(draw)

		# Press Q on keyboard to  exit
		if cv2.waitKey(25) & 0xFF == ord('q'):
		  break
	writer.release()
	cv2.destroyAllWindows()


if __name__ == "__main__":
	# pred_img("./images/test.jpg")
	# pred_img("./images/test2.jpg")
	pred_video("./images/test.mp4")
	# pred_video("./images/test2.mov")

