import numpy as np
import random


class LinearRegression(object):

    def __init__(self, x_list, y_list, lr, max_iter):
        self.x_list = x_list
        self.y_list = y_list
        self.lr = lr
        self.max_iter = max_iter
        self.sample_ratio = 0.5

    def inference(self, w, b, x):
        return  w * x + b


    def eval_loss(self, w, b, x_list, gt_y_list):
        return np.square((w * x_list + b - gt_y_list)).mean() * 0.5

    def gradient(self, pred_y, gt_y, x):
        diff = pred_y - gt_y
        dw = diff * x
        db = diff
        return dw, db

    def cal_step_gradient(self, batch_x_list, batch_gt_y_list, w, b):
        avg_dw, avg_db = 0, 0
        batch_size = len(batch_x_list)
        for i in range(batch_size):
            pred_y = self.inference(w, b, batch_x_list[i])	# get label data
            dw, db = self.gradient(pred_y, batch_gt_y_list[i], batch_x_list[i])
            avg_dw += dw
            avg_db += db
        avg_dw /= batch_size
        avg_db /= batch_size
        w -= self.lr * avg_dw
        b -= self.lr * avg_db
        return w, b

    def train(self):
        w = 0
        b = 0
        num_samples = len(self.x_list)
        for i in range(self.max_iter):
            batch_idxs = np.random.choice(num_samples, int(num_samples * self.sample_ratio))
            batch_x = self.x_list[batch_idxs]
            batch_y = self.y_list[batch_idxs]
            w, b = self.cal_step_gradient(batch_x, batch_y, w, b)
            print('w:{0}, b:{1}'.format(w, b))
            print('loss is {0}'.format(self.eval_loss(w, b, self.x_list, self.y_list)))



def gen_sample_data(num_samples=100):
    w = random.randint(0, 10) + random.random()		# for noise random.random[0, 1)
    b = random.randint(0, 5) + random.random()
    x_list = np.random.randint(0, 100, num_samples) * random.random()
    y_list = w * x_list + b + np.random.random(num_samples) * random.randint(-1, 1)
    return x_list, y_list, w, b

if __name__ == '__main__':
    num_samples = 100
    x_list, y_list, w, b = gen_sample_data(num_samples)
    print('set w:%f,b:%f' %(w, b))
    liner_regression = LinearRegression(x_list, y_list, 0.001, 10000)
    liner_regression.train()