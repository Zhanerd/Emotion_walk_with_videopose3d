# Emotion_walk_with_videopose3d

2D pose use yolov8 and rtmpose. Only supports onnx inference.

3D pose use videopose3d. Only supports torch inference.

Gait Emotion recognition use stgcn. Only supports torch inference.

# requirements

pytorch==2.1.0
onnxruntime==1.18.0

# model download
U can find rtmpose and yolov8 onnx model in https://drive.google.com/drive/folders/1DfTw0aEpuEyXpo7XJXIvCzDTuZ-wNOy8?usp=drive_link .  (In the pose folder! U can put it any where, but need to fill in the corresponding position when initializing class TopDownEstimation.)

The 3d pose model can find in https://github.com/facebookresearch/VideoPose3D. (Pls use the ###coco### model 'pretrained_h36m_detectron_coco.bin'! And put it in videopose3d/checkpoint)

The emotion model can find in https://github.com/PeterZs/take_an_emotion_walk.  (And put it in emotion_walk/weights)



