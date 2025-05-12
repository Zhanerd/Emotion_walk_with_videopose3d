import numpy as np
import torch
from emotion_walk.models.model_loader import load_dgnn, load_stgcn

class Pose2emotion:
    def __init__(self, weights_path=""):
        if 'stgcn' in weights_path:
            self.model = load_stgcn(weights_path)
        elif 'dgnn' in weights_path:
            self.model = load_dgnn(weights_path)
        else:
            print("INVALID PATH!")

    def __call__(self, joint_data):
        pred = self.model(joint_data.unsqueeze(-1))
        return pred