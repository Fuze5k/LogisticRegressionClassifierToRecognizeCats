import numpy as np
import matplotlib.pyplot
import h5py
import os
from skimage.transform import resize
import _pickle  as pickle
class Helper:
    def sigmoid(self, f):
        return 1 / (1 + np.exp(-f))
    
    def initialize_w_and_b(self, k):
        w = np.zeros((k,1))
        b = 0
        return w, b

class Logic:
    def __init__(self):
        self.helper = Helper()

    def forward_and_backward_propagate(self, w, b, X, Y):
        m = X.shape[1]
        A = self.helper.sigmoid(np.dot(w.T,X) + b)
        dw = (1 / m) * np.dot(X,(A - Y).T)
        db = (1 / m) * np.sum(A - Y,axis=1)
        return dw,db

    def optimize(self, w, b, X, Y, k_iterations, step_learning):
        for i in range(k_iterations):
            dw,db = self.forward_and_backward_propagate(w,b,X,Y)
            w = w - step_learning * dw
            b = b - step_learning * db
        return w,b

    def predict(self, w, b, X):
        m = X.shape[1]
        w = w.reshape(X.shape[0], 1)
        A = self.helper.sigmoid(np.dot(w.T,X) + b)
        return A

    def training(self, X_train, Y_train, num_iterations=2000, learning_rate=0.05):
        w, b = self.helper.initialize_w_and_b(X_train.shape[0])
        w_train,b_train = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
        return w_train,b_train


def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig


logic = Logic()
data = dict(w_train=0,b_train=0,num_px=0)
make_train = input("train logistick regresiom y/n: ")
if make_train == "y":
    num_iterations = int(input("num iterations(default 1000): "))
    learning_rate = float(input("learning rate(default 0.1): "))
    train_set_x_orig, train_set_y = load_dataset()
    train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T
    train_set_x = train_set_x_flatten / 255.
    data[3] = train_set_x_orig.shape[1]
    data[1],data[2] = logic.training(train_set_x, train_set_y, num_iterations, learning_rate)
    output = open('train_data.pkl', 'wb')
    pickle.dump(data, output)
    output.close()
else:
    input = open('train_data.pkl', 'rb')
    data = pickle.load(input)
    input.close()

directory = "images"
files = os.listdir(directory)
for file in files:
    image = np.array(matplotlib.pyplot.imread(directory + "/" + file))
    image = image / 255.
    my_image = resize(image,(data[3],data[3])).reshape((1, data[3] * data[3] * 3)).T
    my_predicted_image = logic.predict(data[1],data[2], my_image)
    print(str(file) + " cat on: " + str(np.squeeze(my_predicted_image) * 100) + "%")