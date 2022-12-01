"""
Created on 19.10.8 16:35
@File:generate_frames_and_bbox.py
@author: coderwangson
"""
"#codeing=utf-8"
# TODO using pip install
from mtcnn.mtcnn import MTCNN
import cv2
import os
from glob import glob
detector = MTCNN()
true_img_start = ('1', '2', 'HR_1')
def generate_frames_and_bbox(db_dir,save_dir,skip_num):
    file_list = open(save_dir+"/file_list.txt","w")
    for file in glob("%s/*/*/*.avi"%db_dir):
        print("Processing video %s"%file)
        dir_name = os.path.join(save_dir, *file.replace(".avi", "").split("/")[-3:])
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        frame_num = 0
        count = 0
        vidcap = cv2.VideoCapture(file)  #获取视频文件的处理句柄
        success, frame = vidcap.read()  #按帧读取视频
        while success:

            # 只保存有人脸的帧
            detect_res = detector.detect_faces(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))  #cvtColor 将图像从一个颜色空间转换到另一个颜色空间  opencv中默认颜色制式排列是BGR
            if len(detect_res)>0 and count%skip_num==0:  #每3帧取一张图片

                file_name = os.path.join(dir_name,"frame_%d.jpg" % frame_num)
                # bbox = (x,y,w,h)  x和y为人脸矩形框左上角顶的的x、y坐标值 w和h为人脸矩形框的宽高
                bbox = (detect_res[0]['box'][0],detect_res[0]['box'][1],detect_res[0]['box'][2],detect_res[0]['box'][3])

                # 数据集已经确定哪些前缀视频为活人
                label_txt = file.replace(".avi", "").split("/")[-1]

                # 给每帧图像打标签 1为活人 0位假人
                label = 1 if label_txt in true_img_start else 0

                # file_name x y w h label
                file_list.writelines("%s %d %d %d %d %d\n"%(file_name,bbox[0],bbox[1],bbox[2],bbox[3],label))

                cv2.imwrite(file_name,frame)
                frame_num+=1
            count+=1
            success, frame = vidcap.read()  # 获取下一帧

        vidcap.release()

    file_list.close()
def read():
    file = open("/home/centos/pad/lcnn-pytorch/dataset/CASIA_frames/file_list.txt")  # 打开文件
    for line in file:
        print(line.strip("\n").split(" "))


if __name__ == '__main__':
    db_dir = "/home/centos/pad/lcnn-pytorch/dataset/CASIA"
    save_dir = "/home/centos/pad/lcnn-pytorch/dataset/CASIA_frames"
    generate_frames_and_bbox(db_dir,save_dir,3)


    # read()
