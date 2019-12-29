import pandas as pd
import numpy as np
import os
import shutil
import json



def get_img():
    test_data = pd.read_csv("./val_annotations1.csv",header=None)
    img_list = list(test_data[0])
    for file in img_list:
        img_name = file.split("/")[-1]
        shutil.copy(file,"./test_data/"+img_name)



def get_real_label():
    test_data = pd.read_csv("./val_annotations1.csv",header=None)

    img_label_dict = {}
    img_list = list(test_data[0])
    for file in img_list:
        img_name = file.split("/")[-1]
        if img_name in img_label_dict.keys():
            img_label_dict[img_name].extend(list(test_data.iloc([img_list.index(file)][5])))
        else:
            img_label_dict[img_name] = list(test_data.iloc([img_list.index(file)][5]))

    json_str = json.dumps(img_label_dict)

    with open("./file_label.json",'a') as f:
        f.write(json_str)


if __name__ == "__mian__":
    pass


