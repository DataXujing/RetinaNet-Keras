try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
import os
import pandas as pd
import numpy as np


 


def label_rename(label_str):
    if 'hat' in label_str:
        return 'hat'
    elif 'dog' in label_str:
        return 'hat'
    elif 'person' in label_str:
        return 'person'


def GetAnnotBoxLoc(AnotPath):
    '''
    AnotPath VOC标注文件路径
    '''
    filename = []
    x1 = []
    x2 = []
    x3 = []
    x4 = []
    y1 = []
    y2 = []
    y3 = []
    y4 = []
    types = []

    files = os.listdir(AnotPath)

    for file in files:
        print(file)
        tree = ET.ElementTree(file=AnotPath+"/"+file)  #打开文件，解析成一棵树型结构
        root = tree.getroot()#获取树型结构的根
        ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
        ObjBndBoxSet={} #以目标类别为关键字，目标框为值组成的字典结构
        for Object in ObjectSet:
            ObjName=Object.find('name').text
            BndBox=Object.find('bndbox')
            x_min = int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
            y_min = int(BndBox.find('ymin').text)#-1
            x_max = int(BndBox.find('xmax').text)#-1
            y_max = int(BndBox.find('ymax').text)#-1
            
            try:
                x1.append(x_min)
                x4.append(x_min)
                x2.append(x_max)
                x3.append(x_max)

                y1.append(y_min)
                y2.append(y_min)
                y3.append(y_max)
                y4.append(y_max)
                types.append(label_rename(ObjName))
                filename.append(file)

                if len(filename) != len(x1):
                    print(file)
                    print("-----------------------")
                    break
            except Exception as e:
                print("-----------"+file)
                print(str(e))

    # print(len(filename)) 
    # print(len(x1))
    # print(len(x2))
    # print(len(x3))
    # print(len(x4))
    # print(len(types))
    df_res = pd.DataFrame({"filename":filename,"X1":x1,"Y1":y1,
        "X2":x2,"Y2":y2,"X3":x3,"Y3":y3,"X4":x4,"Y4":y4,"type":types})

    df_res = df_res[["filename","X1","Y1","X2","Y2","X3","Y3","X4","Y4","type"]]

    df_res.to_csv("./train_label_fix.csv",index=0)



if __name__ == "__main__":
    GetAnnotBoxLoc("./Annotations")







