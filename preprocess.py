import os
import shutil
from collections import Counter
from random import shuffle
from PIL import Image, ImageOps
from face_parsing.test import Face_parsing
import numpy as np
import cv2


class Preprocess:
    def __init__(self):
        self.min_images = 18
        self.balance = False
        self.test_ratio = 0.25
        self.train_set_path = "./dataset/train_set/"
        self.test_set_path = "./dataset/test_set/"
        self.LFW_path = "./dataset/lfw"
        self.preprocessed_subset = "./dataset/preprocessed_subset"
        self.horizontal_flip = True
        self.img_size = 64
        num_images_in_LFW, discard = self.preprocess_original_dataset()
        num_people, statistics = self.build_train_test_set()
        self.summary(num_images_in_LFW, discard, num_people, statistics)


    def preprocess_original_dataset(self):
        if os.path.exists(self.preprocessed_subset):
            shutil.rmtree(self.preprocessed_subset)
        os.makedirs(self.preprocessed_subset)

        pars = Face_parsing()
        num_images_in_LFW = 0
        discard = 0
        for folder in os.listdir(self.LFW_path):
            for img_name in os.listdir(os.path.join(self.LFW_path,folder)):
                num_images_in_LFW += 1
                image = Image.open(os.path.join(self.LFW_path, folder, img_name))
                img, mask = pars.transform(image)
                img = np.array(img)
                image.close()

                crop_box = self.count_faces(mask)
                if crop_box is False:      # only one face in image. otherwise discard
                    discard += 1
                else:
                    crop_box = [int(ind) for ind in crop_box]
                    blank_area = np.where(mask == 0)
                    blank_color = np.zeros(3)
                    img[blank_area[0], blank_area[1]] = blank_color
                    # mask = mask[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
                    img = img[crop_box[0]:crop_box[1], crop_box[2]:crop_box[3]]
                    img = Image.fromarray(img)
                    img = img.resize((self.img_size, self.img_size), Image.LANCZOS)
                    if not os.path.exists(os.path.join(self.preprocessed_subset, folder)):
                        os.makedirs(os.path.join(self.preprocessed_subset, folder))
                    img.save(os.path.join(self.preprocessed_subset, folder, img_name))
                    if self.horizontal_flip is True:
                        flip = ImageOps.mirror(img)
                        flip_name = img_name.split('.')
                        flip_name = flip_name[0] + '_F.' + flip_name[1]
                        flip.save(os.path.join(self.preprocessed_subset, folder, flip_name))
        return num_images_in_LFW, discard


    def build_train_test_set(self):
        ''' In LFW dataset most people only have few images (e.g. one), here I discard the people
            with less than min_images samples, and build train and test set with given params. '''
        if os.path.exists(self.train_set_path):
            shutil.rmtree(self.train_set_path)
        os.makedirs(self.train_set_path)
        if os.path.exists(self.test_set_path):
            shutil.rmtree(self.test_set_path)
        os.makedirs(self.test_set_path)

        person_id = 0
        statistics = []
        for folder in os.listdir(self.preprocessed_subset):
            samples = os.listdir(os.path.join(self.preprocessed_subset, folder))
            if len(samples) >= self.min_images:
                person_id += 1
                shuffle(samples)
                if self.balance == False:
                    statistics.append(len(samples))
                    train = samples[:int(len(samples) * (1 - self.test_ratio))]
                    test = samples[int(len(samples) * (1 - self.test_ratio)):]
                else:
                    statistics.append(self.min_images)
                    train = samples[:int(self.min_images * (1 - self.test_ratio))]
                    test = samples[int(self.min_images * (1 - self.test_ratio)):self.min_images]
                for img_id, img_name in enumerate(train):
                    self.histogram_equalization(os.path.join(self.preprocessed_subset, folder, img_name),
                        os.path.join(self.train_set_path, str(person_id) + '_' + str(img_id) + '.jpg'))
                for img_id, img_name in enumerate(test):
                    self.histogram_equalization(os.path.join(self.preprocessed_subset, folder, img_name),
                        os.path.join(self.test_set_path, str(person_id) + '_' + str(img_id) + '.jpg'))

        statistics = Counter(statistics).items()
        statistics = sorted(statistics, key=lambda x: x[0], reverse=True)
        return person_id, statistics


    def count_faces(self, mask):
        '''return False if more than one face is detected, otherwise return the crop box of the face'''
        cnt = 0
        mask_copy = mask.copy()
        crop_box = [float('inf'), -float('inf'), float('inf'), -float('inf')]   # boundary of face
        for x in range(mask_copy.shape[0]):
            for y in range(mask_copy.shape[1]):
                if mask_copy[x, y] == 1:
                    cnt += 1
                    if cnt == 2:
                        return False    # More than one face
                    expand = [(x, y)]
                    while expand:
                        i, j = expand[0]
                        crop_box[0] = min(crop_box[0], i)
                        crop_box[1] = max(crop_box[1], i)
                        crop_box[2] = min(crop_box[2], j)
                        crop_box[3] = max(crop_box[3], j)
                        if mask_copy[i][j] == 1:
                            mask_copy[i][j] = 0
                            if i > 0 and mask_copy[i - 1, j] == 1:
                                expand.append((i - 1, j))
                            if j > 0 and mask_copy[i][j - 1] == 1:
                                expand.append((i, j - 1))
                            if i + 1 < mask.shape[0] and mask_copy[i + 1][j] == 1:
                                expand.append((i + 1, j))
                            if j + 1 < mask.shape[1] and mask_copy[i][j + 1] == 1:
                                expand.append((i, j + 1))
                        del expand[0]
        if cnt == 0:
            return False
        else:
            return crop_box # only one face


    def histogram_equalization(self, img_read_path, img_save_path):
        img = cv2.imread(img_read_path)
        res = []
        for c in range(3):
            channel = img[:, :, c]
            equ_channel = cv2.equalizeHist(channel)
            res.append(equ_channel.reshape([equ_channel.shape[0], equ_channel.shape[1], 1]))
        res = np.concatenate(res, axis=2)
        cv2.imwrite(img_save_path, res)


    def summary(self, num_images_in_LFW, discard, num_people, statistics):
        print("PREPROCESS --------------------------------------------------------------------------------")
        print("Number of classes in LFW:            {}".format(len(os.listdir(self.LFW_path))))
        print("Number of samples in LFW:            {}".format(num_images_in_LFW))
        print("Number of discard samples in LFW:    {}".format(discard))
        print("Horizontal flip:                     {}".format(self.horizontal_flip))
        num_samples_in_subset = num_images_in_LFW - discard
        if self.horizontal_flip is True:
            num_samples_in_subset *= 2
        print("Number of samples in subset:         {}".format(num_samples_in_subset))
        print("Balance among all classes:           {}".format(self.balance))
        print("minimum number of sample of a class: {}".format(self.min_images))
        print("number of selected classes:          {}".format(num_people))
        print("(# samples, # classes): {}".format(statistics))
        print("train / test ratio:                  {} / 1".format((1-self.test_ratio)/self.test_ratio))
        print("train set size:                      {}".format(len(os.listdir(self.train_set_path))))
        print("test set size:                       {}".format(len(os.listdir(self.test_set_path))))
        print()

