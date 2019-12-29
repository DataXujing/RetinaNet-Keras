import pandas as pd
# 标注数据中有错误标注

train_path = "./train_annotations.csv"
val_path = "./val_annotations.csv"


train_df = pd.read_csv(train_path,header=None)
print(train_df.head())

j = 0
drop_index = []
for i in range(train_df.shape[0]):
    my_dat = train_df.iloc[i]

    if float(my_dat[1]) >= float(my_dat[3]) or (float(my_dat[2]) >= float(my_dat[4])):
        print(my_dat[0])

        j += 1
        # print(my_dat)
        drop_index.append(i)
    else:
        pass

print("------------------")
print(j)

train_df.drop(index=drop_index,inplace=True)
train_df = train_df[[0,1,2,3,4,5]]
train_df.to_csv("./train_annotations1.csv",header=None,index=0)

print("-------------------------------")

val_df = pd.read_csv(val_path,header=None)
print(val_df.head())


j = 0
drop_index=[]
for i in range(val_df.shape[0]):
    my_dat = val_df.iloc[i]

    if float(my_dat[1]) >= float(my_dat[3]) or (float(my_dat[2]) >= float(my_dat[4])):
        print(my_dat[0])
        j += 1
        # print(my_dat)
        drop_index.append(i)
    else:
        pass

print("------------------")
print(j)
val_df.drop(index=drop_index,inplace=True)
val_df = val_df[[0,1,2,3,4,5]]
val_df.to_csv("./val_annotations1.csv",header=None,index=0)

