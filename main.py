from preprocess import Preprocess
from model import Model

if __name__ == '__main__':
    preprocess = Preprocess()
    model = Model()
    model.train()
    model.test()
