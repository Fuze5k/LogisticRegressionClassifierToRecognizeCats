import numpy as np
import matplotlib.pyplot
import h5py
import PIL
import scipy.misc
from skimage.transform import resize
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
        grads = {"dw": dw,"db": db}
        return grads

    def optimize(self, w, b, X, Y, k_iterations, step_learning):

        for i in range(k_iterations):
            grads = self.forward_and_backward_propagate(w,b,X,Y)
            dw = grads["dw"]
            db = grads["db"]
            w = w - step_learning * dw
            b = b - step_learning * db

        params = {"w": w,"b": b}
        grads = {"dw": dw,"db": db}
    
        return params, grads

    def predict(self, w, b, X):
        m = X.shape[1]
        Y_p = np.zeros((1,m))
        w = w.reshape(X.shape[0], 1)

        A = self.helper.sigmoid(np.dot(w.T,X) + b)
   
        for i in range(A.shape[1]):
                Y_p[0,i] = A[0,i]

        return Y_p
   
    def training(self, X_train, Y_train, num_iterations=2000, learning_rate=0.5, print_cost=False):
        w, b = self.helper.initialize_w_and_b(X_train.shape[0])
        parameters, grads = self.optimize(w, b, X_train, Y_train, num_iterations, learning_rate)
        w = parameters["w"]
        b = parameters["b"]
        return w,b


def load_dataset():
    train_dataset = h5py.File('dataset/train_catvnoncat.h5', "r")
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    return train_set_x_orig, train_set_y_orig


train_set_x_orig, train_set_y = load_dataset()

train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0],-1).T

train_set_x = train_set_x_flatten / 255.

num_px = train_set_x_orig.shape[1]

logic = Logic();
w,b= logic.training(train_set_x, train_set_y,2000, 0.005)


fname = "images/my_image.jpg"
image = np.array(matplotlib.pyplot.imread(fname))
image = image / 255.
my_image = resize(image,(num_px,num_px)).reshape((1, num_px * num_px * 3)).T
my_predicted_image = logic.predict(w,b, my_image)

matplotlib.pyplot.imshow(image)
print("It is cat on: " + str(np.squeeze(my_predicted_image)*100) + "%")