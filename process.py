import cv2
import params
import numpy as np
import rospy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import MultiArrayDimension
from scipy import ndimage
filtered_images = []

def nothing(x):
    pass

def get_drop_type(x, y):
    global filtered_images
    sx = x * 50
    sy = y * 50
    candidate = {}
    tmp = 0
    for k in range(len(filtered_images)):
        cnt = sum(1 for pix in filtered_images[k][sy+1:sy+49, sx+1:sx+49] if pix.any())
        if cnt > 0:
            if k == 3:
                cnt += 10
            candidate[cnt] = k
            tmp = max(tmp, cnt)
    if tmp == 0:
        return 2
    return candidate[tmp]
def draw_image(drops):
    img = np.zeros((250, 300, 3), np.uint8)
    for y in range(5):
        for x in range(6):
            img[y*50:(y+1)*50, x*50:(x+1)*50] = params.drop_color[drops[y][x]]['color']
    return img
    # cv2.imshow("result", img)
    # cv2.waitKey(0)
def main():
    global filtered_images
    # Creating a window for later use
    cv2.namedWindow('result')
    cv2.namedWindow('image')
    rospy.init_node('image_publisher', anonymous=True)
    pub = rospy.Publisher('board', Float32MultiArray, queue_size=1)
    r = rospy.Rate(10.0)
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
        img = frame[98:309, 322:496]
        # img = frame[97:286, 414:571]
        img = ndimage.rotate(img, -90, reshape=True)
        img = cv2.resize(img, (300, 250))
        img = cv2.LUT(img, Y)
        #converting to HSV
        hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
        filtered_images = [None for i in range(6)]
        for k in range(len(params.drop_color)):
            lower_blue = np.array([params.drop_color[k]['h'][0], params.drop_color[k]['s'][0], params.drop_color[k]['v'][0]])
            upper_blue = np.array([params.drop_color[k]['h'][1], params.drop_color[k]['s'][1], params.drop_color[k]['v'][1]])
            mask = cv2.inRange(hsv,lower_blue, upper_blue)

            result = cv2.bitwise_and(img,img,mask = mask)
            kernel = np.ones((3,3),np.uint8)
            opening = cv2.morphologyEx(result, cv2.MORPH_OPEN, kernel)
            for y in range(5):
                cv2.line(img, (0, 50*y), (300, 50*y), (255, 255, 255))
                cv2.line(opening, (0, 50*y), (300, 50*y), (255, 255, 255))
            for x in range(6):
                cv2.line(img, (50*x, 0), (50*x, 250), (255, 255, 255))
                cv2.line(opening, (50*x, 0), (50*x, 250), (255, 255, 255))
            # cv2.imshow(params.drop_color[k]['name'], opening)
            # cv2.waitKey(100)
            # cv2.imwrite('results/' + params.drop_color[k]['name'] + '.png', opening)
            filtered_images[k] = opening

        mat = Float32MultiArray()
        mat.layout.dim.append(MultiArrayDimension())
        mat.layout.dim.append(MultiArrayDimension())
        mat.layout.dim[0].label = "height"
        mat.layout.dim[1].label = "width"
        mat.layout.dim[0].size = 5
        mat.layout.dim[1].size = 6
        mat.data = [0]*30
        drops = [['' for i in range(6)] for j in range(5)]
        for y in range(5):
            for x in range(6):
                drops[y][x] = get_drop_type(x, y)
                mat.data[x + 6 * y] = drops[y][x] + 1

        pub.publish(mat)
        r.sleep()
        image = draw_image(drops)
        cv2.imshow("image", img)
        cv2.waitKey(100)
        cv2.imshow("result", image)
        cv2.waitKey(100)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
