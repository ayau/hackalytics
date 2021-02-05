import torch
import numpy as np
import cv2
from opts import opts
from model import create_model
# from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d

from utils.debugger import show_2d, mpii_edges

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose


mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

class PoseModel(object):

    def __init__(self):
        # Remove --gpus -1 to use gpu
        if (torch.cuda.is_available()):
            self.opt = opts().parse(['--load_model', 'models/fusion_3d_var.pth'])
            self.opt.device = torch.device('cuda:{}'.format(self.opt.gpus[0]))
        else:
            self.opt = opts().parse(['--load_model', 'models/fusion_3d_var.pth', '--gpus', '-1'])
            self.opt.device = torch.device('cpu')

        self.opt.heads['depth'] = self.opt.num_output

        model, _, _ = create_model(self.opt)
        self.model = model.to(self.opt.device)

    def open_media_pipe(self):
        self.pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    def close_media_pipe(self):
        self.pose.close()

    def predict(self, image):
        s = max(image.shape[0], image.shape[1]) * 1.0
        c = np.array([image.shape[1] / 2., image.shape[0] / 2.], dtype=np.float32)
        trans_input = get_affine_transform(
            c, s, 0, [self.opt.input_w, self.opt.input_h])
        inp = cv2.warpAffine(image, trans_input, (self.opt.input_w, self.opt.input_h),
                             flags=cv2.INTER_LINEAR)
        inp = (inp / 255. - mean) / std
        inp = inp.transpose(2, 0, 1)[np.newaxis, ...].astype(np.float32)
        inp = torch.from_numpy(inp).to(self.opt.device)
        out = self.model(inp)[-1]
        pred = get_preds(out['hm'].detach().cpu().numpy())[0]
        pred = transform_preds(pred, c, s, (self.opt.output_w, self.opt.output_h))
        # pred_3d = get_preds_3d(out['hm'].detach().cpu().numpy(), out['depth'].detach().cpu().numpy())[0]

        # Overlay points on top fo image
        #return show_2d(image, pred, (255, 0, 0), mpii_edges)

        return image, pred, mpii_edges

        # debugger = Debugger()
        # debugger.add_img(image)
        # debugger.add_point_2d(pred, (255, 0, 0))
        # debugger.add_point_3d(pred_3d, 'b')
        # debugger.show_all_imgs(pause=False)
        # debugger.show_3d()

    # based on https://google.github.io/mediapipe/solutions/holistic#python-solution-api
    def predict_with_mediapipe(self, image):
        # Flip the image horizontally for a later selfie-view display, and convert
        # the BGR image to RGB.
        image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
        # To improve performance, optionally mark the image as not writeable to
        # pass by reference.
        image.flags.writeable = False
        results = self.pose.process(image)

        # Draw landmark annotation on the image.
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        return image


