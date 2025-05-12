import torch
import cv2
import numpy as np
from rtmpose.mmpose import TopDownEstimation
from videopose3d.videopose3d import VideoPose3D
from emotion_walk.emotionwalk import Pose2emotion

if __name__ == '__main__':
    cap = cv2.VideoCapture(r'C:\Users\84728\Desktop\videopose3d_emotionwalk\sd1732688500_2.MP4')
    pose2d_model = TopDownEstimation(det_path=r'D:\ai\ai\models\yolov8_m.engine',
                             pose_path=r'D:\ai\ai\models\rtmpose_m.onnx',
                             track_path=r'D:\ai\ai\models\deepsort.engine')
    pose2d = list()

    pose3d_model = VideoPose3D(pose_path=r'C:\Users\84728\Desktop\videopose3d_emotionwalk\videopose3d\checkpoint\pretrained_h36m_detectron_coco.bin')
    pose3d = list()

    pose_emotion = Pose2emotion(weights_path=r'C:\Users\84728\Desktop\videopose3d_emotionwalk\emotion_walk\weights\stgcn_500_5.pt')

    keep_idx = torch.tensor([0,
                              4, 5, 6, 6,
                              1, 2, 3, 3,
                              7, 8, 9, 10,
                              11, 12, 13, 13,
                              14, 15, 16, 16], device='cuda:0')
    ### 推理二维骨架序列
    while cap.isOpened():
        success, img = cap.read()
        if img is None:
            break
        # pose2d = np.expand_dims(pose2d_model.estimate(img,score_thr=0.5)[0]['kps'],axis=0)
        pose_2d = pose2d_model.estimate(img, score_thr=0.5)[0]['kps']
        pose2d.append(pose_2d)

    ### 二维骨架lifting三维骨架
    pose2d = np.array(pose2d)
    pose_17_3d = pose3d_model(pose2d)

    ### human3.6m骨架映射为emotion_walk要求的骨架
    pose_21_3d = pose_17_3d.index_select(dim=1, index=keep_idx)

    ### 三维骨架序列预测情绪
    results = pose_emotion(pose_21_3d.unsqueeze(0).double())

    label = torch.argmax(results)

    if label == 0:
        print("happy")
    elif label == 1:
        print("sad")
    elif label == 2:
        print("angry")
    elif label == 3:
        print("neutral")
