# encoding:utf-8
__author__ = 'Administrator'

"""
Created on Tue Mar 22 13:08:05 2016

@author: Administrator
"""

# !/usr/bin/env python

'''
MOSSE sample

'''

import os
import cv2
import shutil
import numpy as np
from time import time
import matplotlib.pyplot as plt
from commonOperation import draw_str
import scipy

# import ipdb

eps = 1e-5

# EXT_DICT = ['.jpg', '.bmp', '.JPG', '.BMP']
EXT_DICT = ['.bmp', '.BMP']


# SIZE = (256, 128)
# RECT_U_V_100  = ()
# RECT_R_FEAT   = (50, 50, 50, 80)
# RECT_U_V_STAR = (10, 80, 50, 70)
# RECT_R_100 = (95, 5, 30, 35)

def divSpec(A, B):
    Ar, Ai = A[..., 0], A[..., 1]
    Br, Bi = B[..., 0], B[..., 1]
    C = (Ar + 1j * Ai) / (Br + 1j * Bi)
    C = np.dstack([np.real(C), np.imag(C)]).copy()
    return C


class MOSSE:
    def __init__(self):
        self.RECT_U_V_100 = (100, 10, 50, 30)  # U_V 100
        self.RECT_R_FEAT = (60, 50, 100, 50)  # R GHP
        self.RECT_U_V_STAR = (95, 10, 90, 50)  # U_V STAR
        self.RECT_R_100 = (10, 105, 40, 20)  # R 100

    def train_H(self, train_data):
        h, w = train_data[0].shape
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)
        g = np.zeros((h, w), np.float32)
        g[h // 2, w // 2] = 1
        g = cv2.GaussianBlur(g, (-1, -1), 2.0)
        g /= g.max()
        self.G = cv2.dft(g, flags=cv2.DFT_COMPLEX_OUTPUT)
        self.H1 = np.zeros_like(self.G)
        self.H2 = np.zeros_like(self.G)
        for elem in train_data:
            a = self.preprocess(elem)
            A = cv2.dft(a, flags=cv2.DFT_COMPLEX_OUTPUT)
            self.H1 += cv2.mulSpectrums(self.G, A, 0, conjB=True)  # conjB=True 表示在做乘法之前取第二个输入数组的共轭.
            self.H2 += cv2.mulSpectrums(A, A, 0, conjB=True)
        self.update_kernel()

    def load_H(self, H_file):
        self.H = np.load(H_file)
        h, w = self.H.shape[:2]
        self.win = cv2.createHanningWindow((w, h), cv2.CV_32F)

    def update(self, data, thr):
        start = time()
        self.img = data
        data = self.preprocess(data)
        self.resp, self.psr = self.correlate(data)
        self.good = self.psr > thr
        end = time()
        print "cnf spend time %f" % (end - start)
        return self.psr, self.resp

    @property
    def state_vis(self):
        f = cv2.idft(self.H, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        h, w = f.shape
        f = np.roll(f, -h // 2, 0)
        f = np.roll(f, -w // 2, 1)
        kernel = np.uint8((f - f.min()) / f.ptp() * 255)
        resp = self.resp
        resp = np.uint8(np.clip(resp / resp.max(), 0, 1) * 255)
        vis = np.hstack([self.img, kernel, resp])
        return vis

    def draw_state(self, vis):
        draw_str(vis, (10, 10), 'PSR: %.2f' % self.psr)

    def preprocess(self, data):
        data = np.log(np.float32(data) + 1.0)
        data = (data - data.mean()) / (data.std() + eps)
        return data * self.win

    def correlate(self, img):
        C = cv2.mulSpectrums(
            cv2.dft(img, flags=cv2.DFT_COMPLEX_OUTPUT), self.H, 0, conjB=True)
        resp = cv2.idft(C, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
        _, mval, _, _ = cv2.minMaxLoc(resp)
        side_resp = resp.copy()
        smean, sstd = side_resp.mean(), side_resp.std()
        psr = (mval - smean) / (sstd + eps)
        return resp, psr

    def update_kernel(self):
        self.H = divSpec(self.H1, self.H2)
        self.H[..., 1] *= -1

    def save_H(self, path):
        np.save(path, self.H)

    def save_H_R(self, path):
        np.save(path, self.H[..., 0])

    def save_H_I(self, path):
        np.save(path, self.H[..., 1])

    # modify lbl
    def save_mat_H(self, path):
        scipy.io.savemat(os.path.join(path, 'mosse_H.mat'), {'array': self.H})

    def save_mat_H_R(self, path):
        scipy.io.savemat(os.path.join(path, 'mosse_H_R.mat'), {'array': self.H})

    def save_mat_H_I(self, path):
        scipy.io.savemat(os.path.join(path, 'mosse_H_I.mat'), {'array': self.H})


if __name__ == '__main__':
    # sample_path = r'E:\boshi\U_V_100\1'
    sample_path = r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\Sample_Sort\V05_100_Z_Z'
    mosse = MOSSE()
    files = os.listdir(sample_path)
    data = [cv2.imread(os.path.join(sample_path, file))[:, :, 0]
            for file in files if os.path.splitext(file)[-1] in EXT_DICT]
    mosse.train_H(train_data=data)
#    mosse.save_H(os.path.join(sample_path, 'mosse_H'))
#    mosse.save_mat_H(sample_path)
#    mosse.save_mat_H_R(sample_path)
#    mosse.save_mat_H_I(sample_path)
'''
    dir = []
    for x in data:
        dir.append(mosse.update(x, 0)[0])
'''

testimage = cv2.imread(r'E:\LiuBingLe_Data\Start_Programme\Face_Recognition\Sample\Sample_Sort\V05_5_Z_Z\38CROP_ZG.bmp')[:,:,0]
imagepsr, imageresp = mosse.update(testimage,0.8)
#smean, sstd = imageresp.mean(), imageresp.std()


cv2.imshow('test',testimage)
cv2.waitKey(0)
cv2.destroyAllWindows()



