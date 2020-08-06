import cv2
from mtcnn.core.detect import create_mtcnn_net, MtcnnDetector
from mtcnn.core.vision import vis_face
import os
import time

def catch_face(frame):
    #img = cv2.imread("./test11.jpg")
    img_bg = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # b, g, r = cv2.split(img)
    # img2 = cv2.merge([r, g, b])

    bboxs, landmarks = mtcnn_detector.detect_face(frame)
    print("I found {} face(s) in this photograph.".format(len(bboxs)))
    color = (0, 255, 0)
    if len(bboxs)>0 :
        for bbox in bboxs:
            x = int(bbox[0])
            y = int(bbox[1])
            h = int((bbox[3] - bbox[1]))
            w = int((bbox[2] - bbox[0]))

            image = frame[(y):(y+h),(x):(x+w) ]
            # 保存人脸图像
            #save_face(image, tag, num)
            num=1
            d = cv2.waitKey(1)
            if d & 0xFF == ord('c'):
                window_name = 'catch face'
                cv2.namedWindow(window_name)
                cv2.imshow(window_name, image)
                save_face(image, 5, num)
            num+=1

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

    if landmarks is not None:
        for i in range(landmarks.shape[0]):
            landmarks_one = landmarks[i, :]
            landmarks_one = landmarks_one.reshape((5, 2))
            for j in range(5):
                cv2.circle(frame,(int(landmarks_one[j,0]),int(landmarks_one[j,1])),radius=2,color=(0, 0, 255))

def save_face(image, tag, num):
  # DATA_TRAIN为抓取的人脸存放目录
    DATA_TRAIN = './Data/FaceID'
    img_name = os.path.join(DATA_TRAIN, str(tag), '{}_{}.jpg'.format(int(time.time()), num))
    # 保存人脸图像到指定的位置, 其中会创建一个tag对应的目录，用于后面的分类训练
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(img_name, image)


if __name__ == '__main__':
    pnet, rnet, onet = create_mtcnn_net(p_model_path="./original_model/pnet_epoch.pt",
                                        r_model_path="./original_model/rnet_epoch.pt",
                                        o_model_path="./original_model/onet_epoch.pt", use_cuda=False)
    mtcnn_detector = MtcnnDetector(pnet=pnet, rnet=rnet, onet=onet, min_face_size=24)

    window_name = 'main'
    camera_idx = 0
    cv2.namedWindow(window_name)
    # 视频来源，可以来自一段已存好的视频，也可以直接来自摄像头
    cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
    while cap.isOpened():
        # 读取一帧数据
        ok, frame = cap.read()
        if not ok:
            break
        # 抓取人脸的方法, 后面介绍
        catch_face(frame)
        # 输入'q'退出程序
        cv2.imshow(window_name, frame)
        c = cv2.waitKey(1)
        if c & 0xFF == ord('q'):
            break
    # 释放摄像头并销毁所有窗口
    cap.release()
    cv2.destroyAllWindows()



