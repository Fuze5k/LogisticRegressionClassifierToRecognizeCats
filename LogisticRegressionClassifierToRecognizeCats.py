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

class Logic:
    def forward_and_backward_propagate(self, w, b, X, Y):
        m = X.shape[1]
        A = sigmoid(np.dot(w.T,X) + b)
        dw = (1 / m) * np.dot(X,(A - Y).T)
        db = (1 / m) * np.sum(A - Y,axis=1)
        grads = {"dw": dw,"db": db}
        return grads

    def optimize(self, w, b, X, Y, k_iterations, step_learning):

        for i in range(k_iterations):
            grads = forward_and_backward_propagate(w,b,X,Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - step_learning * dw
            b = b - step_learning * db

        params = {"w": w,"b": b}
        grads = {"dw": dw,"db": db}
    
        return params, grads

    def predict(w, b, X):
        m = X.shape[1]
        Y_p = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        A = sigmoid(np.dot(w.T,X)+b)
   
        for i in range(A.shape[1]):
            if A[0,i]<0.5:
                Y_prediction[0,i]=0
            else:
                Y_prediction[0,i]=1
        return Y_prediction



def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig


train_set_x_orig, train_set_y = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten / 255.



