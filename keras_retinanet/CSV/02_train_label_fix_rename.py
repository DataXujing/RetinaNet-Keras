import pandas as pd
import os
import random


# train_label_fix rename
img_path = "./JPEGImages"
def file_rename():
    annotion = pd.read_csv("./train_label_fix.csv",dtype={'type': 'str'})
    file_name = [file.split(".")[0] for file in list(annotion['filename'])]

    img_name = [file.split(".")[0] for file in os.listdir(img_path)]
    img_tail = [file.split(".")[1] for file in os.listdir(img_path)]

    new_file_name = []
    for file in file_name:
        file_index = img_name.index(file)
        new_file_name.append(file+"."+img_tail[file_index])


    annotion['new_file'] = new_file_name
    annotion = annotion[['new_file',"X1","Y1","X2","Y2","X3","Y3","X4","Y4","type"]]

    annotion.to_csv("./train_label_fix1.csv",index=0)


if __name__ == "__main__":
    file_rename()

# 手动把new_file改成filename
