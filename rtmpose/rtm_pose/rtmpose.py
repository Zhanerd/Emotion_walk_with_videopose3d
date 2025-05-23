from typing import List, Tuple

import numpy as np

from ..base import BaseTool
from .post_processings import convert_coco_to_openpose, get_simcc_maximum
from .pre_processings import bbox_xyxy2cs, top_down_affine


class RTMPose(BaseTool):

    def __init__(self,
                 model_path: str,
                 model_input_size: tuple = (192, 256),
                 mean: tuple = (123.675, 116.28, 103.53),
                 std: tuple = (58.395, 57.12, 57.375),
                 to_openpose: bool = False,
                 gpu_id: int = 0):
        super().__init__(model_path=model_path,model_input_size=model_input_size, mean=mean, std=std,
                         gpu_id=gpu_id)
        self.to_openpose = to_openpose

    # def __call__(self, image: np.ndarray, bboxes: list = []):
    #     if len(bboxes) == 0:
    #         bboxes = [[0, 0, image.shape[1], image.shape[0]]]
    #
    #     keypoints, scores = [], []
    #     for bbox in bboxes:
    #         img, center, scale = self.preprocess(image, bbox)
    #         outputs = self.inference(img)
    #         kpts, score = self.postprocess(outputs, center, scale)
    #
    #         keypoints.append(kpts)
    #         scores.append(score)
    #
    #     keypoints = np.concatenate(keypoints, axis=0)
    #     scores = np.concatenate(scores, axis=0)
    #
    #     if self.to_openpose:
    #         keypoints, scores = convert_coco_to_openpose(keypoints, scores)
    #
    #     return keypoints, scores

    def __call__(self, image: np.ndarray, bboxes: list = []):
        if not bboxes:
            bboxes = [[0, 0, image.shape[1], image.shape[0]]]

        keypoints, scores = [], []


        # 预处理所有边界框
        preprocessed_data = [self.preprocess(image, bbox) for bbox in bboxes]

        # 提取预处理后的图像、中心点和尺度
        imgs, centers, scales = zip(*preprocessed_data)

        # 将所有预处理后的图像堆叠成一个batch
        batch_imgs = np.stack(imgs, axis=0)

        outputs = []
        # 模型推理
        simcc_x, simcc_y = self.inference(batch_imgs)
        # 按行遍历
        for row_x,row_y in zip(simcc_x,simcc_y):
            row_y = np.expand_dims(row_y,axis=0)
            row_x = np.expand_dims(row_x,axis=0)
            outputs.append([row_x,row_y])
            # print(row_x,row_y)
        # 后处理所有输出
        for i, (output, center, scale) in enumerate(zip(outputs, centers, scales)):
            kpts, score = self.postprocess(output, center, scale)
            keypoints.append(kpts)
            scores.append(score)

        # 合并所有关键点和分数
        keypoints = np.concatenate(keypoints, axis=0)
        scores = np.concatenate(scores, axis=0)

        if self.to_openpose:
            keypoints, scores = convert_coco_to_openpose(keypoints, scores)

        return keypoints, scores

    def preprocess(self, img: np.ndarray, bbox: list):
        """Do preprocessing for RTMPose model inference.

        Args:
            img (np.ndarray): Input image in shape.
            bbox (list):  xyxy-format bounding box of target.

        Returns:
            tuple:
            - resized_img (np.ndarray): Preprocessed image.
            - center (np.ndarray): Center of image.
            - scale (np.ndarray): Scale of image.
        """
        bbox = np.array(bbox)

        # get center and scale
        center, scale = bbox_xyxy2cs(bbox, padding=1.25)
        # do affine transformation
        resized_img, scale = top_down_affine(self.model_input_size, scale,
                                             center, img)
        # normalize image
        if self.mean is not None:
            self.mean = np.array(self.mean)
            self.std = np.array(self.std)
            resized_img = (resized_img - self.mean) / self.std

        return resized_img, center, scale

    def postprocess(
            self,
            outputs: List[np.ndarray],
            center: Tuple[int, int],
            scale: Tuple[int, int],
            simcc_split_ratio: float = 2.0) -> Tuple[np.ndarray, np.ndarray]:
        """Postprocess for RTMPose model output.

        Args:
            outputs (np.ndarray): Output of RTMPose model.
            model_input_size (tuple): RTMPose model Input image size.
            center (tuple): Center of bbox in shape (x, y).
            scale (tuple): Scale of bbox in shape (w, h).
            simcc_split_ratio (float): Split ratio of simcc.

        Returns:
            tuple:
            - keypoints (np.ndarray): Rescaled keypoints.
            - scores (np.ndarray): Model predict scores.
        """
        # decode simcc
        simcc_x, simcc_y = outputs
        locs, scores = get_simcc_maximum(simcc_x, simcc_y)
        keypoints = locs / simcc_split_ratio

        # rescale keypoints
        keypoints = keypoints / self.model_input_size * scale
        keypoints = keypoints + center - scale / 2

        return keypoints, scores
