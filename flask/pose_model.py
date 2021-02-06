import torch
import numpy as np
import cv2
from opts import opts
from model import create_model
# from utils.debugger import Debugger
from utils.image import get_affine_transform, transform_preds
from utils.eval import get_preds, get_preds_3d
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import models
from utils.debugger import show_2d, mpii_edges

import mediapipe as mp
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

import torchvision.transforms as T



mean = np.array([0.485, 0.456, 0.406], np.float32).reshape(1, 1, 3)
std = np.array([0.229, 0.224, 0.225], np.float32).reshape(1, 1, 3)

class PoseModel(object):

    def __init__(self):
        # Remove --gpus -1 to use gpu
        #if (torch.cuda.is_available()):
        if(False):
            self.opt = opts().parse(['--load_model', 'models/fusion_3d_var.pth'])
            self.opt.device = torch.device('cuda:{}'.format(self.opt.gpus[0]))
        else:
            self.opt = opts().parse(['--load_model', 'models/fusion_3d_var.pth', '--gpus', '-1'])
            self.opt.device = torch.device('cpu')

        self.opt.heads['depth'] = self.opt.num_output

        model, _, _ = create_model(self.opt)
        self.model = model.to(self.opt.device)

    def open_media_pipe(self):
        self.pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

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


    def predict_segment(self, image, dev='cpu'):
        img = image
        #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)
        nc=21
        net = models.segmentation.deeplabv3_resnet101(pretrained=1).eval()
        #if show_orig: plt.imshow(img); plt.axis('off'); plt.show()
        # Comment the Resize and CenterCrop for better inference results
        trf = T.Compose([T.Resize(640), 
                        #T.CenterCrop(224), 
                        T.ToTensor(), 
                        T.Normalize(mean = [0.485, 0.456, 0.406], 
                                    std = [0.229, 0.224, 0.225])])
        inp = trf(img).unsqueeze(0).to(dev)

        #print(inp)

        out = net.to(dev)(inp)['out']
        om = torch.argmax(out.squeeze(), dim=0).detach().cpu().numpy()
        
        label_colors = np.array([(0, 0, 0),  # 0=background
                    # 1=aeroplane, 2=bicycle, 3=bird, 4=boat, 5=bottle
                    (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
                    # 6=bus, 7=car, 8=cat, 9=chair, 10=cow
                    (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
                    # 11=dining table, 12=dog, 13=horse, 14=motorbike, 15=person
                    (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
                    # 16=potted plant, 17=sheep, 18=sofa, 19=train, 20=tv/monitor
                    (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)])

        r = np.zeros_like(om).astype(np.uint8)
        g = np.zeros_like(om).astype(np.uint8)
        b = np.zeros_like(om).astype(np.uint8)

        
        for l in range(0, nc):
            idx = om == l
            r[idx] = label_colors[l, 0]
            g[idx] = label_colors[l, 1]
            b[idx] = label_colors[l, 2]
            
        rgb = np.stack([r, g, b], axis=2)
        print(rgb)
        img = Image.fromarray(rgb, 'RGB')
        return cv2.cvtColor(np.float32(rgb), cv2.COLOR_RGB2BGR)


    def process_video_into_frames_and_poses(self, video_path):
        vidcap = cv2.VideoCapture(video_path)
        success, image = vidcap.read()

        frames_and_poses = []
        while success:
            image.flags.writeable = False
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            pose_output = self.pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            frames_and_poses.append({'image': image, 'pose_results': pose_output})

            print(pose_output.pose_landmarks)
            success,image = vidcap.read()

        vidcap.release()


