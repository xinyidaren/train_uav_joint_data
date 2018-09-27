import scipy.io as scio
import numpy as np
import cv2
import os
import tensorflow as tf

def int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

# 配置参数
width = 640     #图像宽度
height = 480    #图像高度
datadir = '/home/learn/work_midas/python_uva_data/' # 原图像地址
train_filename = 'uav_joint_train.tfrecord'         # 保存数据集合名称

with tf.python_io.TFRecordWriter(train_filename) as tfrecord_writer:
    # 列出文件夹下所有的目录与文件
    list_1 = os.listdir(datadir)
    # 排序
    list_1.sort()
    for dataFile in list_1[0:50000]:
        if((int(dataFile[14:20])) % 50 == 0):
            print(dataFile)
            # 读取mat文件
            data = scio.loadmat(datadir + dataFile)

            # 获取图像最小值,用作归一化使用
            min = 10000
            for i in range(height):
                for j in range(width):
                    if data['synthdepth'][i][j] >0 and data['synthdepth'][i][j] <min:
                        min = data['synthdepth'][i][j]

            #获取图像最大值,用作归一化使用
            max = data['synthdepth'].max()

            #将数值归一化至0-255之间
            for i in range(height):
                for j in range(width):
                    if data['synthdepth'][i][j] > 0:
                        data['synthdepth'][i][j] = int((data['synthdepth'][i][j] - min + 10)/(max-min + 10)*255) #10为容错数据
            # 将归一化后的文件转换为uint8类型
            img = data['synthdepth'].astype(np.uint8)

            #将图像修改成可二值图像后,进行二值操作
            ret, binary = cv2.threshold(img,1,255,cv2.THRESH_BINARY)
            cv2.imshow("bin",binary)

            #提取外轮廓
            image, contours, hierarchy = cv2.findContours(binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

            #制作32*32*32的图像数据
            if contours:
                #截取手部图像
                x, y, w, h = cv2.boundingRect(contours[0])
                img = cv2.resize(img[y - 10:y + h + 10, x - 10:x + w + 10], (32, 32), interpolation=cv2.INTER_AREA)
                origin_3d = [img/255 for i in range(32)]  # 选择单一通道复制32份形成32×32×32的方阵

                origin_3d = np.transpose(origin_3d, (1, 2, 0))  # 方阵转至并归一化
                create_3d = np.fromfunction(lambda x, y, z: z / 32, (32, 32, 32))  # 创造32×32×32的结果方阵

                img3d = create_3d - origin_3d
                cv2.imshow("ddd",img)

                # 合并jnt的x,y坐标,并展开成[40]大小的label
                label = np.append((data['jnt_uvdx'] - x)/ w, (data['jnt_uvdy'] - y)/ h, axis=0).flatten(order='F')

                # 制作tfrecord数据集
                feature = {'train/image': bytes_feature(img3d.tostring()),
                           'train/label': bytes_feature(label.tostring()),  # label: integer from 0-N
                           'train/name': int64_feature(i),
                           'train/height': int64_feature(height),
                           'train/width': int64_feature(width)}

                # create example protocol buffer
                example = tf.train.Example(features=tf.train.Features(feature=feature))

                # serialize protocol buffer to string
                # tfrecord_writer.write(example.SerializeToString())

            cv2.waitKey(1)