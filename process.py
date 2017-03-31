import os
import cv2
import sys
import mnist.model as model
import params
import numpy as np
import rospy
from std_msgs.msg import String
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from scipy import ndimage
import operator
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
    # Creating a window for later use
    # cv2.namedWindow('result')
    cv2.namedWindow('image')
    rospy.init_node('image_publisher', anonymous=True)
    pub = rospy.Publisher('board', Float32MultiArray, queue_size=1)
    pub2 = rospy.Publisher('result', String, queue_size=1)
    r = rospy.Rate(0.5)
    # for i in range(len(params.drop_color)):
    #     cv2.namedWindow(params.drop_color[i]['name'])
    cap = cv2.VideoCapture(0)
    while(cap.isOpened()):
        ret, frame = cap.read()
        im = frame[170:430, 63:533]
        # img = frame[97:286, 414:571]
        # cv2.imshow("image", im)
        im = ndimage.rotate(im, 90, reshape=True)
        cv2.imshow("image", im)
        cv2.waitKey(1000)
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        ret, th = cv2.threshold(blur, 100, 300, cv2.THRESH_BINARY_INV)
        _, ctrs, hier = cv2.findContours(th.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        ctrs
        rects = [cv2.boundingRect(ctr) for ctr in ctrs]
        rects = sorted(rects, key=lambda x:x[0] + x[1] * 10)
        imgs = []
        im2 = im.copy()
        hsv = cv2.cvtColor(im2,cv2.COLOR_BGR2HSV)
        for rect in rects:
            if rect[2] > 3 and rect[3] < 30 and rect[2] * rect[3] > 100 and rect[2]*rect[3] < 400:
                # cv2.rectangle(im, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
                imgs.append(im2[rect[1]:rect[1]+rect[3], rect[0]:rect[0]+rect[2]])
        mask = cv2.inRange(hsv, np.array([0, 160, 160]), np.array([15, 255, 255]))
        print(cv2.countNonZero(mask))
        if (cv2.countNonZero(mask) > 70000):
            pub2.publish(String("fail"))
        mask2 = cv2.inRange(hsv, np.array([109, 0, 0]), np.array([179, 255, 255]))
        print(cv2.countNonZero(mask2))
        if (cv2.countNonZero(mask2) > 70000):
            pub2.publish(String("success"))
        # print(len(imgs))
        # for i in range(len(imgs)):
        #     cv2.imshow("result", imgs[i])
        #     cv2.waitKey(1000)
        if (len(imgs) == 9):
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
                gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(gray, (3,3),0)
                ret, th = cv2.threshold(gray, 100, 230, cv2.THRESH_BINARY)
                input = ((255 - th) / 255.0).reshape(1, 784)
                output = convolutional(input)
                ind = np.argpartition(output, -2)[-2:]
                index, value = max(enumerate(output), key=operator.itemgetter(1))

                if (value >= 0.85) :
                    numlist.append(index)
                else:
                    if 7 in np.argpartition(output, -3)[-3:] and output[7] > 0.01:
                        numlist.append(7)
                    elif 6 in np.argpartition(output, -3)[-3:] and output[6] > 0.01:
                        numlist.append(6)
                    else:
                        for e in ind:
                            if (e != index):
                                numlist.append(e)

                # print(i, output)
                # cv2.imshow("result", img2)
                # cv2.waitKey(1000)
            a = [x for x in range(1, 10)]
            b = numlist.copy()
            b.sort()
            print(numlist)
            if (a == b):
                print(numlist)
                mat = Float32MultiArray()
                mat.layout.dim.append(MultiArrayDimension())
                mat.layout.dim.append(MultiArrayDimension())
                mat.layout.dim[0].label = "height"
                mat.layout.dim[1].label = "width"
                mat.layout.dim[0].size = 3
                mat.layout.dim[1].size = 3
                mat.data = [0]*9
                for y in range(3):
                    for x in range(3):
                        mat.data[x + 3 * y] = numlist[x + 3 * y]

                pub.publish(mat)
                r.sleep()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
