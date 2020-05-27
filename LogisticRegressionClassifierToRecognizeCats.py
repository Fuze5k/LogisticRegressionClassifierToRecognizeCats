import numpy as np
import matplotlib.pyplot as plt
import h5py
#import scipy
#from PIL import Image
class Helper:
    def sigmoid(self, f):
        return 1 / (1 + np.exp(-f))
    
    def initialize_w_and_b(self, k):
        w = np.zeros((k,1))
        b = 0
        return w, b

    def forward_and_backward_propagate(w, b, X, Y):
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        dw = (1 / m) * np.dot(X,(A - Y).T)
        db = (1 / m) * np.sum(A - Y,axis=1)
        grads = {"dw": dw,"db": db}
        return grads

def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig


train_set_x_orig, train_set_y = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten / 255.



