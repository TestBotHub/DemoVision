import cv2
import params
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from std_msgs.msg import String
from scipy import ndimage

def getRectByPoints(points):
    points = list(map(lambda x:x[0], points))
    points = sorted(points, key=lambda x:x[1])
    top_points = sorted(points[:2], key=lambda x:x[0])
    bottom_points = sorted(points[2:4], key=lambda x:x[0])
    points = top_points + bottom_points

    left = min(points[0][0], points[2][0])
    right = max(points[1][0], points[3][0])
    top = min(points[0][1], points[1][1])
    bottom = max(points[2][1], points[3][1])
    return (top, bottom, left, right)

tmp = None
def generate(msg):
    global tmp

    cv2.imwrite("/home/mech-user/tmp.png", tmp)
def main():
    global tmp
    # Creating a window for later use
    cv2.namedWindow('result')
    cv2.namedWindow('image')
    rospy.init_node('image_publisher', anonymous=True)
    pub = rospy.Publisher('position', Float32MultiArray, queue_size=1)
    rospy.Subscriber('generate', String, generate)
    r = rospy.Rate(1.0)
    # for i in range(len(params.drop_color)):
    #     cv2.namedWindow(params.drop_color[i]['name'])
    cap = cv2.VideoCapture(params.cam_number)
    Y = np.ones((256, 1), dtype= 'uint8') * 0
    for i in range(256):
        Y[i][0] = 255 * pow(float(i)/255.0, 1.0 / .9)
    while(cap.isOpened()):
    # img = cv2.imread('images/'+ str(i) + '.png')
        ret, frame = cap.read()
        # frame = cv2.imread('images/1.png')
        # img = cv2.resize(frame, (300, 400))
        # img = img[185:, 6:294]
        img = frame[32:219, 181:463]
        # img = frame[97:286, 414:571]
        img = ndimage.rotate(img, -90, reshape=True)
        img = cv2.resize(img, (300, 250))
        img = cv2.LUT(img, Y)

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img_blur = cv2.GaussianBlur(img_gray, (11, 11), 0)
        th = cv2.threshold(img_blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
        contours, hierarchy = cv2.findContours(th, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        print(contours)
        print(hierarchy)
        th_area = img.shape[0] * img.shape[1] / 100
        print(th_area)
        contours_large = list(filter(lambda c:cv2.contourArea(c) > th_area, contours))

        rects = []
        approxes = []
        for (i, cnt) in enumerate(contours_large):
            arclen = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02* arclen, True)
            if len(approx) < 4:
                continue
            approxes.append(approx)
            rect = getRectByPoints(approx)
            rects.append(rect)
        if len(rects) == 4:
            cv2.line(img, (rects[2][2], rects[2][0]), (rects[2][3], rects[2][0]), (0,0,0), 5)
            cv2.line(img, (rects[2][3], rects[2][0]), (rects[2][3], rects[2][1]), (0,0,0), 5)
            cv2.line(img, (rects[2][3], rects[2][1]), (rects[2][2], rects[2][1]), (0,0,0), 5)
            cv2.line(img, (rects[2][2], rects[2][1]), (rects[2][2], rects[2][0]), (0,0,0), 5)
            tmp = np.empty_like(img)
            tmp[:] = img

        for rect in rects:
            cv2.line(img, (rect[2], rect[0]), (rect[3], rect[0]), (0,0,0), 5)
            cv2.line(img, (rect[3], rect[0]), (rect[3], rect[1]), (0,0,0), 5)
            cv2.line(img, (rect[3], rect[1]), (rect[2], rect[1]), (0,0,0), 5)
            cv2.line(img, (rect[2], rect[1]), (rect[2], rect[0]), (0,0,0), 5)


        mat = Float32MultiArray()
        mat.layout.dim.append(MultiArrayDimension())
        mat.layout.dim.append(MultiArrayDimension())
        mat.layout.dim[0].label = "height"
        mat.layout.dim[1].label = "width"
        mat.layout.dim[0].size = 2
        mat.layout.dim[1].size = 4
        mat.data = [0]*8
        print(len(rects))
        for i in range(len(rects)):
            mat.data[2*i] = 1.0 * (rects[i][2] + rects[i][3]) / 2.0
            mat.data[2*i + 1] = 1.0 * (rects[i][0] + rects[i][1]) / 2.0

        pub.publish(mat)
        r.sleep()

        cv2.imshow("image", img)
        cv2.waitKey(100)
        # if tmp != None and len(tmp) > 0:
        #     cv2.imshow("result", tmp)
        #     cv2.waitKey(100)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
