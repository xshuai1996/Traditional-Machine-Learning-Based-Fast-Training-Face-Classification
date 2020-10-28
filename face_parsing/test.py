from face_parsing.model import BiSeNet
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


class Face_parsing:
    def __init__(self):
        n_classes = 19
        model = './face_parsing/79999_iter.pth'
        self.net = BiSeNet(n_classes=n_classes)
        self.net.cuda()
        self.net.load_state_dict(torch.load(model))
        self.net.eval()
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

    def transform(self, img):
        with torch.no_grad():
            image = img.resize((512, 512), Image.BILINEAR)
            img = self.to_tensor(image)
            img = torch.unsqueeze(img, 0)
            img = img.cuda()
            out = self.net(img)[0]
            parsing = out.squeeze(0).cpu().numpy().argmax(0)

            # atts = [1 'skin', 2 'l_brow', 3 'r_brow', 4 'l_eye', 5 'r_eye', 6 'eye_g', 7 'l_ear', 8 'r_ear', 9 'ear_r',
            #         10 'nose', 11 'mouth', 12 'u_lip', 13 'l_lip', 14 'neck', 15 'neck_l', 16 'cloth', 17 'hair', 18 'hat']
            discard = {0, 7, 8, 9, 14, 15, 16, 17, 18}
            mask = np.ones((parsing.shape[:2]))
            for i in discard:
                mask = np.where(parsing == i, 0, mask)
            return image, mask





