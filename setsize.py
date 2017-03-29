import cv2
import numpy as np
import params
import json

selecting = False
selected = False
cnt = 0
tx, ty, curx, cury = 0, 0, 0, 0
area = []
def onmouse(event, x, y, flags, params):
    global selecting, area, tx, ty, curx, cury, selected, cnt
    curx, cury = x, y
    if event == cv2.EVENT_LBUTTONDOWN:
        selecting = True
        tx, ty = x, y
    elif event == cv2.EVENT_MOUSEMOVE:
        if selecting:
            selected = True
            if tx != x and ty != y:
                area = (min(tx, x), min(ty, y), abs(tx-x), abs(ty-y))
    elif event == cv2.EVENT_LBUTTONUP:
        if tx != x and ty != y:
            area = (min(tx, x), min(ty, y), abs(tx-x), abs(ty-y))
def main():
    # Creating a window for later use
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', onmouse)
    cv2.namedWindow('result')
    # Starting with 100's to prevent error while masking

    cap = cv2.VideoCapture(params.cam_number)
    while(cap.isOpened()):
    # frame = cv2.imread('images/'+ str(i) + '.png')
        ret, frame = cap.read()
        if selected:
            frame_copy = frame.copy()
            cv2.rectangle(frame_copy, (tx, ty), (curx, cury), (0, 255, 0), 1)
            cv2.imshow('image', frame_copy)
            result = frame[area[1]:area[1] + area[3], area[0]:area[0] + area[2]]
            result = cv2.resize(result, (400, 200))
            for y in range(25):
                cv2.line(result, (0, 8*y), (400, 8*y), (255, 255, 255))
            for x in range(50):
                cv2.line(result, (8*x, 0), (8*x, 200), (255, 255, 255))
            cv2.imshow('result', result)
        else:
            cv2.imshow('image', frame)
            cv2.imshow('result', frame)
        k = cv2.waitKey(10) & 0xFF
        # cv2.waitKey(0)
        if k == 27:
            break
        elif k == ord('s'):
            frame_size = [area[1], area[1] + area[3], area[0], area[0] + area[2]]
            with open('../config.json', 'r') as fp:
                data = json.load(fp)
            print(data)
            if data == None:
                data = {}
            data['frame'] = frame_size
            with open('../config.json', 'w') as fp:
                json.dump(data, fp)
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
