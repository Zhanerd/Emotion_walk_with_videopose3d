import numpy as np
import torch
import cv2
from videopose3d.common.camera import *
from videopose3d.common.utils import deterministic_random
from videopose3d.common.model import *
from videopose3d.common.generators import ChunkedGenerator, UnchunkedGenerator

class VideoPose3D:
    def __init__(self,res_w_h=(640, 640), pose_path=""):
        self.custom_camera_params = {
            'id': None,
            'res_w': res_w_h[0],  # Pulled from metadata
            'res_h': res_w_h[1],  # Pulled from metadata

            # Dummy camera parameters (taken from Human3.6M), only for visualization purposes
            'azimuth': 70,  # Only used for visualization
            'orientation': [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088],
            'translation': [1841.1070556640625, 4955.28466796875, 1563.4454345703125],
        }
        self.pose_path = pose_path
        self.pose_model = None

        self.joints_num = 17
        self.filter_widths = [3, 3, 3, 3, 3]
        self.receptive_field = 243
        self.pad = (self.receptive_field - 1) // 2
        self.causal_shift = 0

    def __call__(self, pose2d):
        pose2d = self.preprocessing(pose2d)

        ### 因输入为动态，所以需要重新初始化模型
        self._init_model()
        pose3d = self.inference(pose2d)
        return pose3d

    def _init_model(self):
        self.pose_model = TemporalModel(self.joints_num, 2, # in_feature 2 意为2维坐标的x，y
                                  self.joints_num,
                                  filter_widths=self.filter_widths, causal=False, dropout=0.25,
                                  channels=1024,
                                  dense=False)
        self.receptive_field = self.pose_model.receptive_field()
        self.pose_model = self.pose_model.cuda()
        checkpoint = torch.load(self.pose_path, map_location=lambda storage, loc: storage)
        self.pose_model.load_state_dict(checkpoint['model_pos'])
        self.pad = (self.receptive_field - 1) // 2

    def preprocessing(self, pose2d):
        pose2d[..., :2] = normalize_screen_coordinates(pose2d[..., :2], w=self.custom_camera_params['res_w'],
                                                    h=self.custom_camera_params['res_h'])
        self.joints_num = pose2d.shape[-2]
        return pose2d

    def inference(self, pose2d, test_augment=True):
        joints_left = [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        kps_left = [1, 3, 5, 7, 9, 11, 13, 15]
        kps_right = [2, 4, 6, 8, 10, 12, 14, 16]
        with torch.no_grad():
            self.pose_model.eval()
            inputs_2d = np.expand_dims(np.pad(pose2d,
                                             ((self.pad + self.causal_shift, self.pad - self.causal_shift), (0, 0),
                                              (0, 0)),
                                             'edge'), axis=0)

            if test_augment:
                inputs_2d = np.concatenate((inputs_2d, inputs_2d), axis=0)
                inputs_2d[1, :, :, 0] *= -1
                inputs_2d[1, :, kps_left + kps_right] = inputs_2d[1, :, kps_right + kps_left]

            inputs_2d = torch.from_numpy(inputs_2d.astype('float32'))
            if torch.cuda.is_available():
                inputs_2d = inputs_2d.cuda()

            predicted_3d_pos = self.pose_model(inputs_2d)

            if test_augment:
                # Undo flipping and take average with non-flipped version
                predicted_3d_pos[1, :, :, 0] *= -1
                predicted_3d_pos[1, :, joints_left + joints_right] = predicted_3d_pos[1, :, joints_right + joints_left]
                predicted_3d_pos = torch.mean(predicted_3d_pos, dim=0, keepdim=True)

        return predicted_3d_pos.squeeze(0)

    # def fetch(self, subjects, action_filter=None, subset=1, parse_3d_poses=True, downsample=1):
    #     out_poses_3d = []
    #     out_poses_2d = []
    #     out_camera_params = []
    #     for subject in subjects:
    #         for action in keypoints[subject].keys():
    #             if action_filter is not None:
    #                 found = False
    #                 for a in action_filter:
    #                     if action.startswith(a):
    #                         found = True
    #                         break
    #                 if not found:
    #                     continue
    #
    #             poses_2d = keypoints[subject][action]
    #             for i in range(len(poses_2d)):  # Iterate across cameras
    #                 out_poses_2d.append(poses_2d[i])
    #
    #             if subject in dataset.cameras():
    #                 cams = dataset.cameras()[subject]
    #                 assert len(cams) == len(poses_2d), 'Camera count mismatch'
    #                 for cam in cams:
    #                     if 'intrinsic' in cam:
    #                         out_camera_params.append(cam['intrinsic'])
    #
    #             if parse_3d_poses and 'positions_3d' in dataset[subject][action]:
    #                 poses_3d = dataset[subject][action]['positions_3d']
    #                 assert len(poses_3d) == len(poses_2d), 'Camera count mismatch'
    #                 for i in range(len(poses_3d)):  # Iterate across cameras
    #                     out_poses_3d.append(poses_3d[i])
    #
    #     if len(out_camera_params) == 0:
    #         out_camera_params = None
    #     if len(out_poses_3d) == 0:
    #         out_poses_3d = None
    #
    #     stride = downsample
    #     if subset < 1:
    #         for i in range(len(out_poses_2d)):
    #             n_frames = int(round(len(out_poses_2d[i]) // stride * subset) * stride)
    #             start = deterministic_random(0, len(out_poses_2d[i]) - n_frames + 1, str(len(out_poses_2d[i])))
    #             out_poses_2d[i] = out_poses_2d[i][start:start + n_frames:stride]
    #             if out_poses_3d is not None:
    #                 out_poses_3d[i] = out_poses_3d[i][start:start + n_frames:stride]
    #     elif stride > 1:
    #         # Downsample as requested
    #         for i in range(len(out_poses_2d)):
    #             out_poses_2d[i] = out_poses_2d[i][::stride]
    #             if out_poses_3d is not None:
    #                 out_poses_3d[i] = out_poses_3d[i][::stride]
    #
    #     return out_camera_params, out_poses_3d, out_poses_2d