import cv2
import os
import time


def catch_video(tag, window_name='catch face', camera_idx=0):
    cv2.namedWindow(window_name)
  # 视频来源，可以来自一段已存好的视频，也可以直接来自摄像头
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    while cap.isOpened():
       # 读取一帧数据
        ok, frame = cap.read()
        if not ok:
            break
      # 抓取人脸的方法, 后面介绍
        catch_face(frame, tag)
        # 输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
  # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()
    return 0

def catch_face(frame, tag):
    # 告诉OpenCV使用人脸识别分类器
    classfier = cv2.CascadeClassifier("C:\Program Files (x86)\Microsoft Visual Studio\Shared\Python37_64\Lib\site-packages\cv2\data\haarcascade_frontalface_alt2.xml")
    # 识别出人脸后要画的边框的颜色，RGB格式
    color = (0, 255, 0)
    # 将当前帧转换成灰度图像
    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 人脸检测，1.2和2分别为图片缩放比例和需要检测的有效点数
    face_rects = classfier.detectMultiScale(grey, scaleFactor=1.2, minNeighbors=3, minSize=(32, 32))
    num = 1
    if len(face_rects) > 0: # 大于0则检测到人脸
        # 图片帧中有多个图片，框出每一个人脸
        for face_rects in face_rects:
            x, y, w, h = face_rects
            image = frame[y - 10:y + h + 10, x - 10:x + w + 10]
            # 保存人脸图像
            #save_face(image, tag, num)
            cv2.rectangle(frame, (x - 10, y - 10), (x + w + 10, y + h + 10), color, 2)
            num += 1


def save_face(image, tag, num):
  # DATA_TRAIN为抓取的人脸存放目录
    DATA_TRAIN = './data/train'
    img_name = os.path.join(DATA_TRAIN, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
    # 保存人脸图像到指定的位置, 其中会创建一个tag对应的目录，用于后面的分类训练
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_name, image)

catch_video(tag=4)
