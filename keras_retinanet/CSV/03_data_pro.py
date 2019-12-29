import pandas as pd
import os
import random


# train_label_fix rename
img_path = "./JPEGImages"

annotion = pd.read_csv("./train_label_fix1.csv")
file_path = [img_path+"/"+i for i in list(annotion.filename) ]
annotion['filepath'] = file_path

files = os.listdir(img_path)

validation = random.sample(files,int(len(files)*0.01))

all_files = list(annotion['filename'])
print(all_files)
valid_index = [all_files.index(i) for i in validation]
train = list(set(files) - set(validation))
train_index = [all_files.index(i) for i in train]


valid_data = annotion.iloc[valid_index]
train_data = annotion.iloc[train_index]

train_save = train_data[['filepath',"X1","Y1","X3","Y3","type"]]
valid_save = valid_data[['filepath',"X1","Y1","X3","Y3","type"]]


train_save.to_csv("train_annotations.csv",header=None,index=False)
valid_save.to_csv("val_annotations.csv",header=None,index=False)

print(valid_data.head())
