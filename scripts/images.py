from PIL import Image, ImageDraw
import numpy as np
import cv2

if __name__ == '__main__':

    img = Image.open('/home/user/Downloads/5bbec42e4b226-20181011090158.jpg')
    mser = cv2.MSER()

    while True:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        vis = img.copy()

        regions = mser.detect(gray, None)
        hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
        cv2.polylines(vis, hulls, 1, (0, 255, 0))

        cv2.imshow('img', vis)
        if 0xFF & cv2.waitKey(5) == 27:
            break
    cv2.destroyAllWindows()
