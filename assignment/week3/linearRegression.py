'''
1. Reorganize Linear Regression in Python mode.
You have seen what we are doing in class to do linear regression. That is not bad in C++. But it's not a good
idea in Python because we were not using Python's features at all.
So, your first task is: rewrite linear regression code in Python. You are not allowed to use "Too Many For
Loops", especially when doing calculations.
Write the code in "Python's way". Go ahead and good luck.
2. Logistic regression:
Logistic regression is widely used. We derived the cost function and it's gradient in class. Please complete
the logistic regression code in "Python's way" as well.
Tips: It's almost like the linear regression code. The only difference is you need to complete a sigmoid
function and use the result of that as your "new X" and also you need to generate your own training data.

'''

import numpy as np
import random


class LinearRegression(object):

    def __init__(self, lr, max_iter):
        self.X = []
        self.y_list = []
        self.lr = lr
        self.max_iter = max_iter
        self.sample_ratio = 0.5

    def inference(self, w, b, X):
        return np.dot(w.T, X)+b #w * x + b


    def eval_loss(self, w, b, X, y):
        return np.square((np.dot(w.T, X)+b - y)).mean() * 0.5

    def gradient(self, pred_y, gt_y, x):
        diff = pred_y - gt_y
        dw = diff * x
        db = diff
        return dw, db

    def cal_step_gradient(self, X, y, w, b):

        dz = self.inference(w, b, X) - y

        avg_dw = np.sum(np.dot(X, dz.T), axis=0) / len(y)
        avg_db = np.sum(dz) * 1.0/len(y)

        w -= self.lr * avg_dw
        b -= self.lr * avg_db
        return w, b



    def train(self):

        b = np.zeros(1)
        num_samples = self.X.shape[1]
        num_factors = self.X.shape[0]
        w = np.zeros((num_factors,1))
        for i in range(self.max_iter):
            batch_idxs = np.random.choice(num_samples, int(num_samples * self.sample_ratio))
            batch_x = self.X[..., batch_idxs]
            batch_y = self.y_list[batch_idxs]
            #print(batch_x)
            w, b = self.cal_step_gradient(batch_x, batch_y, w, b)
            print('w:{0}, b:{1}'.format(w, b))
            print('loss is {0}'.format(self.eval_loss(w, b, self.X, self.y_list)))

    def gen_sample_data(self, num_samples=100):
        w = np.random.randint(0, 10, 1) + random.random()  # for noise random.random[0, 1)
        b = np.random.randint(0, 5, 1) + random.random()
        X = np.random.randint(0, 100, size=(1, num_samples)) * random.random()
        y_list = self.inference(w, b, X) + np.random.random(num_samples) * random.randint(-1, 1)
        return X, y_list, w, b

    def start_test(self):
        num_samples = 100
        self.X, self.y_list, w, b = self.gen_sample_data(num_samples)
        self.train()
        print('start train with w is {0} ,b is {1}'.format(w, b))




class LogisticRegression(LinearRegression):

    def inference(self, w, b, X):
        return self.sigmoid(np.dot(w.T, X)+b)

    def sigmoid(self, z):
        return 1/(1 + np.exp(-z))


    def gen_sample_data(self, num_samples=100):
        w = np.random.randint(0, 10, 1) + random.random()  # for noise random.random[0, 1)
        b = np.random.randint(0, 5, 1) + random.random()
        X = np.random.randint(0, 100, size=(1, num_samples)) * random.random()
        err = np.multiply(np.random.random(num_samples), np.random.randint(-1, 1,num_samples))
        y_list = np.dot(w.T, X)+ b  + err
        r_list = np.array([1 if y >= 0 else 0 for y in err])
        X = np.array([X[0],y_list])

        return X, r_list, w, b



if __name__ == '__main__':
    liner_regression = LinearRegression(0.001, 10000)
    liner_regression.start_test()

    print('-------------logistic regression-----------')
    logistic_regression = LogisticRegression(0.001, 10000)
    logistic_regression.start_test()

