import os
import cv2
import sys
# sys.path.append(os.path.join(os.path.dirname(__file__), 'mnist'))
import mnist.model as model
import operator

import numpy as np
import tensorflow as tf

x = tf.placeholder("float", [None, 784])
sess = tf.Session()


with tf.variable_scope("convolutional"):
    keep_prob = tf.placeholder("float")
    y2, variables = model.convolutional(x, keep_prob)
saver = tf.train.Saver(variables)
saver.restore(sess, "mnist/data/convolutional.ckpt")


def convolutional(input):
    return sess.run(y2, feed_dict={x: input, keep_prob: 1.0}).flatten().tolist()


def main():
    cv2.namedWindow("image")
    im = cv2.imread('test.png')
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)
    ret, th = cv2.threshold(blur, 100, 300, cv2.THRESH_BINARY_INV)
    _, ctrs, hier = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    ctrs
    rects = [cv2.boundingRect(ctr) for ctr in ctrs]
    rects = sorted(rects, key=lambda x:x[0] + x[1] * 10)
    imgs = []
    im2 = im.copy()
    for rect in rects:
        if rect[2] * rect[3] > 50 and rect[2]*rect[3] < 200:
            cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
            imgs.append(im2[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
            print(rect)
    cv2.imshow("image", im)
    numlist = []
    for i in range(9):
        img2 = np.zeros((28, 28, 3),np.uint8)
        img2[:,:] = (255,255,255)
        if imgs[i].shape[0] > imgs[i].shape[1]:
            img = cv2.resize(imgs[i], (int(imgs[i].shape[1] * 20 / imgs[i].shape[0]), 20),interpolation = cv2.INTER_AREA)
            rest = 28 - img.shape[1]
            img2[4:24,int(rest/2-1):int(rest/2-1+img.shape[1])] = img
        else:
            img = cv2.resize(imgs[i], (20, int(imgs[i].shape[0] * 20 / imgs[i].shape[1])),interpolation = cv2.INTER_AREA)
            rest = 28 - img.shape[0]
            img2[int(rest/2-1):int(rest/2-1+img.shape[0]),4:24] = img
        # img2 = cv2.resize(imgs[i], (28, 28),interpolation = cv2.INTER_AREA)
        # img2[:,:] = (255,255,255)
        # if img.shape[0] > img.shape[1]:
        #     img = cv2.resize(img, (int(img.shape[1] * 20 / img.shape[0]), 20),interpolation = cv2.INTER_AREA)
        #     rest = 28 - img.shape[1]
        #     img2[4:24,rest/2-1:rest/2-1+img.shape[1]] = img
        # else:
        #     img = cv2.resize(img, (20, int(img.shape[0] * 20 / img.shape[1])),interpolation = cv2.INTER_AREA)
        #     rest = 28 - img.shape[0]
        #     img2[rest/2-1:rest/2-1+img.shape[0],4:24] = img
        # cv2.imshow("image", im)
        # cv2.waitKey(2000)
        # cv2.imshow("image", img2)
        # cv2.waitKey(2000)
        # cv2.imwrite("test" + str(i) + "-resize.png", img2)
        gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (3,3),0)
        ret, th = cv2.threshold(gray, 100, 230, cv2.THRESH_BINARY)
        input = ((255 - th) / 255.0).reshape(1, 784)
        output = convolutional(input)
        ind = np.argpartition(output, -2)[-2:]
        print(ind)
        index, value = max(enumerate(output), key=operator.itemgetter(1))
        if (value >= 0.9) :
            numlist.append(index)
        else:
            if 7 in np.argpartition(output, -3)[-3:]:
                numlist.append(7)
            elif 6 in np.argpartition(output, -3)[-3:]:
                numlist.append(6)
            else:
                for e in ind:
                    if (e != index):
                        numlist.append(e)
        print(output)
        print(index, value)
    print(numlist)
if __name__ == '__main__':
    main()
