# -*- coding:utf-8 -*-

"""全连接前馈神经网络训练和测试实现。学习算法采用小批量随机梯度下降算法。
输出层采用softmax函数，代价函数采用对数似然函数，有正则化"""

import numpy as np
import random
import mnist_loader
import matplotlib.pyplot as plt


class Network(object):

    def __init__(self, struct, w=None, b=None, batsize=5, i_max=30, c_min=1e-3, rate=1.0, lam=10):
        # 初始化神经网络
        self.struct = struct
        self.batsize = batsize
        self.nlayer = len(struct) # 神经网络层数，不包含输入层
        self.rate = rate
        self.lam = lam
        self.c = []
        if w is None:
            w, b = self.initwb()
        self.w = w
        self.b = b
        self.i_max = i_max
        self.c_min = c_min
        self.nin = struct[0]
        self.nout = struct[-1]

    def initwb(self):
        # 初始化权重和偏置
        w = [None] + [np.random.randn(a,b)/np.sqrt(b) for a,b in zip(self.struct[1:], self.struct[:-1])]
        b = [None] + [np.random.randn(a, 1) for a in self.struct[1:]]
        # for i in range(0, self.nlayer-1):
        #     w.append(np.ones((self.struct[i+1], self.struct[i]))/self.struct[i])
        #     b.append(np.zeros((self.struct[i+1], 1)))
        return w, b

    def batpro(self, num):
        # 随机分组
        ind = list(range(num))
        # random.shuffle(ind)
        self.nbat = num//self.batsize+1
        t = self.batsize-num%self.batsize
        tt = random.sample(ind, t)
        ind_2 = ind + tt
        random.shuffle(ind_2)
        return [ind_2[i:i+self.batsize] for i in range(0, num, self.batsize)]
        # return np.array(ind_2).reshape((self.nbat, self.batsize))

    def train(self, x, y, x_t, y_t):
        # 训练
        num = x.shape[1]
        self.t = 1 - self.rate*self.lam/num
        epoch = 0
        c = float('inf')
        while epoch < self.i_max:
            bat_inds = self.batpro(num)
            for inds in bat_inds:
                x_bat = x[:, inds]
                y_bat = y[:, inds]
                self.update(x_bat, y_bat)
            epoch += 1
            _, aa = self.forword(x)
            al = aa[-1]
            c = self.calcc(al, y)
            print('c', c)
            self.c.append(c)
            print('Epoch:', epoch)
            self.test(x_t, y_t)
        plt.plot(self.c)
        plt.show()

    def test(self, x, y):
        # 测试
        a = x
        for i in range(1, self.nlayer):
            z = self.w[i]@a+self.b[i]
            a = sigmoid(z)
        ind_p = np.argmax(a, 0)
        ind = np.argmax(y, 0)
        print(sum(ind_p == ind)/y.shape[1])

    def forword(self, x_bat):
        aa = [x_bat] + [np.zeros((t.shape[0], self.batsize)) for t in self.b[1:]]
        zz = [None] + [np.zeros((t.shape[0], self.batsize)) for t in self.b[1:]]
        for i in range(1, self.nlayer - 1):
            zz[i] = self.w[i] @ aa[i - 1] + self.b[i]
            aa[i] = sigmoid(zz[i])
        zz[-1] = self.w[-1] @ aa[-2] + self.b[-1]
        aa[-1] = softmax(zz[-1])
        return zz, aa

    def backward(self, zz, aa, y_bat):
        # 对小批量数据计算各个参数的平局梯度，反向传播算法的关键所在
        delt = [None] + [np.zeros((t.shape[0], self.batsize)) for t in self.b[1:]]
        dw = [None] + [np.zeros(w.shape) for w in self.w[1:]]
        db = [None] + [np.zeros((t.shape[0], self.batsize)) for t in self.b[1:]]

        delt[-1] = aa[-1] - y_bat  # 采用对数似然函数时的 delta_L
        db[-1] = np.mean(delt[-1], 1)
        db[-1].shape = (len(db[-1]), 1)
        dw[-1] = delt[-1]@(aa[-2].T)/self.batsize
        for i in range(self.nlayer-2, 0, -1):
            d_l = ((self.w[i+1].T)@delt[i+1])*sigmoid_d(zz[i])
            delt[i] = d_l
            b_l = np.mean(d_l, 1)
            b_l.shape = (len(b_l), 1)
            db[i] = b_l
            w_l = d_l@(aa[i-1].T)/self.batsize
            dw[i] = w_l
        return dw, db

    def update(self, x_bat, y_bat):
        # 更新参数
        zz, aa = self.forword(x_bat)
        dw, db = self.backward(zz, aa, y_bat)
        self.w = [None]+[self.t*w1-self.rate*w2 for w1, w2 in zip(self.w[1:], dw[1:])]
        self.b = [None]+[b1-self.rate*b2 for b1, b2 in zip(self.b[1:], db[1:])]

    def calcc(self, al, y):
        ind = np.argmax(y, 0)
        return - np.mean(np.log(al[ind, list(range(y.shape[1]))]))

def softmax(z):
    return np.exp(z)/sum(np.exp(z))

def sigmoid(z):
    # 激活函数
    return 1.0/(1.0 + np.exp(-z))

def sigmoid_d(z):
    # 激活函数导数
    return sigmoid(z) * (1 - sigmoid(z))

def parse_data(data):
    # 解析数据
    x_list = [sample[0] for sample in data]
    y_list = [sample[1] for sample in data]
    x = np.array(x_list)
    y = np.array(y_list)
    x = x.T
    y = y.T
    x.shape = x.shape[1:]
    y.shape = y.shape[1:]
    return x, y


if __name__ == '__main__':
    training_data, validation_data, test_data = mnist_loader.load_data_wrapper()
    training_data = list(training_data)
    test_data = list(test_data)
    test_data1 = []
    for i, data in enumerate(test_data):
        t = np.zeros((10, 1))
        t[data[1], 0] = 1.0
        test_data1.append([data[0], t])

    x, y = parse_data(training_data)
    x_t, y_t = parse_data(test_data1)
    nn = Network([784, 100, 10], batsize=10, rate=0.5, i_max=30, lam=1)
    nn.train(x, y, x_t, y_t)
