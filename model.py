import os
from PIL import Image
import numpy as np
from skimage.util import view_as_windows
from sklearn.decomposition import PCA
import time


class Model:
    def __init__(self, local_cuboid_sizes=None, components_per_channel=90):
        self.local_cuboid_sizes = [(1, 8, 8, 1), (1, 4, 4, 16), (1, 2, 2, 64)] \
            if local_cuboid_sizes is None else local_cuboid_sizes
        self.components_per_channel = components_per_channel
        self.Exps = [[] for _ in range(3)]
        self.PCAs = [[] for _ in range(3)]
        self.ws = None
        self.bs = None

    def load_data_set(self, data_set_path):
        data_set = [[] for _ in range(3)]       # separate into 3 channels RGB
        labels = []
        for img_name in os.listdir(data_set_path):
            img = Image.open(os.path.join(data_set_path, img_name))
            img = np.array(img, dtype=np.float)
            for c in range(3):
                channel = img[:, :, c].reshape((1, img.shape[0], img.shape[1], 1))
                data_set[c].append(channel)
            labels.append(int(img_name.split('_')[0]))
        for c in range(3):
            data_set[c] = np.concatenate(data_set[c], axis=0)
        labels = np.array(labels)
        return data_set, labels


    def global_to_local_cuboids(self, dataset, local_cuboid_size):
        dataset = view_as_windows(dataset, local_cuboid_size, local_cuboid_size)
        shape = dataset.shape
        dataset = np.reshape(dataset, (shape[0], shape[1] * shape[2], shape[5] * shape[6] * shape[7]))
        return dataset


    def dimension_reduction(self, dataset, num_component):
        Expectations, PCAs = [], []
        res = []
        for i in range(dataset.shape[1]):
            features = dataset[:, i, :]
            exp = np.mean(features, axis=0)
            Expectations.append(exp)
            features -= exp
            pca = PCA(n_components=num_component)
            pca.fit(features)
            PCAs.append(pca)
            features = pca.transform(features)
            features = np.reshape(features, (features.shape[0], 1, features.shape[1]))
            res.append(features)
        dataset = np.concatenate(res, axis=1)
        shape = dataset.shape
        dataset = np.reshape(dataset, (shape[0], int(pow(shape[1], 0.5)), int(pow(shape[1], 0.5)), shape[2]))
        return Expectations, PCAs, dataset


    def train_classifier(self, train_set, train_labels):
        mus, covs, ratios = [], [], []       # ratio = # belong to a class / # total samples
        num_classes = np.unique(train_labels).shape[0]
        for i in range(1, num_classes+1):
            mask = np.where(train_labels == i)
            samples = train_set[mask]
            mu = np.mean(samples, axis=0)
            mu = np.reshape(mu, (1, mu.shape[0]))
            cov = np.cov(samples.T)
            cov = np.reshape(cov, (1, cov.shape[0], cov.shape[1]))
            ratio = samples.shape[0] / train_labels.shape[0]
            mus.append(mu)
            covs.append(ratio * cov)
            ratios.append(ratio)

        pooled_cov = np.sum(covs, axis=0)
        pooled_cov = np.reshape(pooled_cov, (pooled_cov.shape[1], pooled_cov.shape[2]))
        pooled_cov_inv = np.linalg.inv(pooled_cov)

        ws, bs = [], []
        for i in range(num_classes):
            w = mus[i] @ pooled_cov_inv
            b = -0.5 * w @ mus[i].T + np.log(ratios[i])
            ws.append(w)
            bs.append(b)

        self.ws = np.concatenate(ws, axis=0)
        self.bs = np.concatenate(bs, axis=0).reshape((-1, 1))


    def train(self, train_set_path="./dataset/train_set/"):
        train_set, train_labels = self.load_data_set(train_set_path)

        start_time = time.time()

        for c in range(3):
            train_set[c] = self.global_to_local_cuboids(train_set[c], self.local_cuboid_sizes[0])
            Expectations, PCAs, train_set[c] = self.dimension_reduction(train_set[c], self.local_cuboid_sizes[1][3])
            self.Exps[c].append(Expectations)
            self.PCAs[c].append(PCAs)
            train_set[c] = self.global_to_local_cuboids(train_set[c], self.local_cuboid_sizes[1])
            Expectations, PCAs, train_set[c] = self.dimension_reduction(train_set[c], self.local_cuboid_sizes[2][3])
            self.Exps[c].append(Expectations)
            self.PCAs[c].append(PCAs)
            train_set[c] = self.global_to_local_cuboids(train_set[c], self.local_cuboid_sizes[2])
            Expectations, PCAs, train_set[c] = self.dimension_reduction(train_set[c], self.components_per_channel)
            self.Exps[c].append(Expectations)
            self.PCAs[c].append(PCAs)
            train_set[c] = np.reshape(train_set[c], (train_set[c].shape[0], train_set[c].shape[-1]))

        train_set = np.concatenate(train_set, axis=1)
        self.train_classifier(train_set, train_labels)

        end_time = time.time()
        pred = self.predict_with_classifier(train_set)
        print("TRAIN: ", end='')
        self.calculate_accuracy(pred, train_labels)
        print("Time consuming:               {} sec".format(format(end_time - start_time, '.2f')))


    def PCA_transform(self, test_set, Exp, PCA):
        res = []
        for i in range(test_set.shape[1]):
            features = test_set[:, i, :]
            features -= Exp[i]
            features = PCA[i].transform(features)
            features = np.reshape(features, (features.shape[0], 1, features.shape[1]))
            res.append(features)
        test_set = np.concatenate(res, axis=1)
        shape = test_set.shape
        test_set = np.reshape(test_set, (shape[0], int(pow(shape[1], 0.5)), int(pow(shape[1], 0.5)), shape[2]))
        return test_set


    def predict_with_classifier(self, test_set):
        predict = self.ws @ test_set.T + self.bs
        predict = np.argmax(predict, axis=0)
        predict += 1    # index start from 0 but class labels start from 1
        return predict


    def test(self, test_set_path="./dataset/test_set/"):
        test_set, test_labels = self.load_data_set(test_set_path)
        start_time = time.time()
        for c in range(3):
            test_set[c] = self.global_to_local_cuboids(test_set[c], self.local_cuboid_sizes[0])
            test_set[c] = self.PCA_transform(test_set[c], self.Exps[c][0], self.PCAs[c][0])
            test_set[c] = self.global_to_local_cuboids(test_set[c], self.local_cuboid_sizes[1])
            test_set[c] = self.PCA_transform(test_set[c], self.Exps[c][1], self.PCAs[c][1])
            test_set[c] = self.global_to_local_cuboids(test_set[c], self.local_cuboid_sizes[2])
            test_set[c] = self.PCA_transform(test_set[c], self.Exps[c][2], self.PCAs[c][2])
            test_set[c] = np.reshape(test_set[c], (test_set[c].shape[0], test_set[c].shape[-1]))

        test_set = np.concatenate(test_set, axis=1)
        predict = self.predict_with_classifier(test_set)
        end_time = time.time()
        print("TEST: ", end='')
        self.calculate_accuracy(predict, test_labels)
        print("Time consuming:               {} sec".format(format(end_time - start_time, '.2f')))


    def calculate_accuracy(self, predict, labels):
        predict = predict.reshape((-1))
        labels = labels.reshape((-1))
        correct = np.where(predict == labels)[0].shape[0]
        print("CALCULATE ACCURACY --------------------------------------------------------------------")
        print("Number of correct prediction: {}".format(correct))
        print("Number of test samples:       {}".format(labels.shape[0]))
        print("Accuracy:                     {}%".format(format(correct/labels.shape[0]*100, '.2f')))
