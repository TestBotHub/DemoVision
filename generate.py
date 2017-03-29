import cv2
import time
def main():
    camera = cv2.VideoCapture(0)
    cnt = 1
    while True:
        ret, img = camera.read()

        cv2.imwrite("images/pd/" + str(cnt) + ".png", img)
        time.sleep(0.5)
        k = cv2.waitKey(1) & 0xFF
        if k == 27:
            break
        cnt += 1
if __name__ == "__main__":
    main()
