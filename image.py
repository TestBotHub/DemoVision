import cv2
import numpy as np
import numpy.linalg as la

def ang(v1, v2):
    cosang = np.dot(v1, v2)
    sinang = la.norm(np.cross(v1, v2))
    return np.arctan2(sinang, cosang) * 360.0 / (2 * np.pi)

def nothing(x):
    pass
def main():
    camera = cv2.VideoCapture(0)
    cv2.namedWindow('gray')
    cv2.createTrackbar('th_u', 'gray', 0, 255, nothing)
    cv2.createTrackbar('th_l', 'gray', 0, 255, nothing)
    while True:
        # size 640 * 480
        retval, img = camera.read()
        img = cv2.resize(img, (640, 480))#[80:400,50:590]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        th_u = cv2.getTrackbarPos('th_u','gray')
        th_l = cv2.getTrackbarPos('th_l','gray')
        cv2.imshow("gray", gray)
        edged = cv2.Canny(gray, th_l, th_u, None, 3)
        cv2.imshow("Edged", edged)
        # lines = cv2.HoughLinesP(edged, 1, np.pi, threshold=10, minLineLength=40, maxLineGap=1)
        # if len(lines):
        #     for x1, y1, x2, y2 in lines[0]:
        #         cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 1)
        #
        #
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        closed = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
        cv2.imshow("Closed", closed)
        (cnts, _) = cv2.findContours(closed.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        cnts.sort(key=cv2.contourArea, reverse=True)
        contours = list(filter(lambda c:cv2.contourArea(c) > 50000 and cv2.contourArea(c) < 200000, cnts))
        for c in contours:
            peri = cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, 0.02 * peri, True)
            if len(approx) == 4:
                angles = [ang(approx[i][0] - approx[(i+3) % 4][0], approx[i][0] - approx[(i+1) % 4][0]) for i in range(4)]
                if 80 <= min(angles) and max(angles) <= 120:
                    print("detected!!", cv2.contourArea(c))
                    cv2.drawContours(img, [approx], -1, (0, 255, 0), 4)
                    # cv2.imshow("Result", img[min(approx[])])
        cv2.imshow("Main", img)
        k = cv2.waitKey(50) & 0xFF
        if k == 27:
            break

if __name__ == "__main__":
    main()
