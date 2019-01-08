from PIL import Image, ImageDraw
import numpy as np
import cv2
import math


class ImageProcessing(object):
    RED = (0, 0, 255)
    YELLOW = (0, 255, 255)
    BLUE = (255, 0, 0)
    GREEN = (0, 255, 0)

    def __init__(self, image_list=[]):
        self.image_list = image_list
        self.size = None

    def get_images(self):
        if not self.image_list:
            print "Please provide image to extract picture."
            return False
        for img in self.image_list:
            print img
            image = cv2.imread(img)
            if not image.any():
                print "Couldn't load image %s".format(img)
                continue
            squares_list, rects, img_copy = self.find_squares(image)
            count = 1
            print "squares_list", squares_list, rects
            for seq in rects:

                # Find bounding rectangle for mouth coordinates
                x, y, w, h = cv2.boundingRect(seq)

                new_image = image[y:y + h, x:x + w]

                h, w, channels = new_image.shape
                # If the cropped region is very small, ignore this case.
                if h < 10 or w < 10:
                    continue
                img_name = "image_" + str(count) + ".jpg"
                count += 1

                resized = cv2.resize(new_image, 32, 32)
                cv2.imwrite(img_name, resized)
                cv2.imshow(img_name, resized)

                # # r = cv2.boundingRect(seq)
                # r = np.zeros(seq, dtype=np.uint8)
                # print r, seq
                # s = image(r)
                #
                # cv2.imshow(img_name, s)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def find_squares(self, img):
        img = cv2.GaussianBlur(img, (5, 5), 0)
        squares = []
        for gray in cv2.split(img):
            for thrs in xrange(0, 255, 26):
                if thrs == 0:
                    bin = cv2.Canny(gray, 0, 50, apertureSize=5)
                    bin = cv2.dilate(bin, None)
                else:
                    _retval, bin = cv2.threshold(gray, thrs, 255, cv2.THRESH_BINARY)
                contours, _hierarchy = self.find_contours(bin)
                for cnt in contours:
                    x, y, w, h = cv2.boundingRect(cnt)
                    cnt_len = cv2.arcLength(cnt, True)
                    cnt = cv2.approxPolyDP(cnt, 0.02*cnt_len, True)
                    area = cv2.contourArea(cnt)
                    if len(cnt) == 4 and 20 < area < 1000 and cv2.isContourConvex(cnt):
                        cnt = cnt.reshape(-1, 2)
                        max_cos = np.max([self.angle_cos(cnt[i], cnt[(i+1) % 4], cnt[(i+2) % 4] ) for i in xrange(4)])
                        if max_cos < 0.1:
                            if (1 - (float(w) / float(h)) <= 0.07 and 1 - (float(h) / float(w)) <= 0.07):
                                squares.append(cnt)
        return squares

    def preallocate(self, img):
        if self.size is None or self.size[0] != img.shape[0] or self.size[1] != img.shape[1]:
            h, w = img.shape[:2]
            self.size = (h, w)

            self.img = np.empty((h, w, 3), dtype=np.uint8)

            self.hsv = np.empty((h, w, 3), dtype=np.uint8)
            self.bin = np.empty((h, w, 1), dtype=np.uint8)
            self.bin2 = np.empty((h, w, 1), dtype=np.uint8)

            self.out = np.empty((h, w, 3), dtype=np.uint8)

            # for overlays
            self.zeros = np.zeros((h, w, 1), dtype=np.bool)
            self.black = np.zeros((h, w, 3), dtype=np.uint8)

            self.morphKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2), anchor=(0, 0))

        cv2.copyMakeBorder(img, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)

    def threshold(self, img):
        cv2.cvtColor(img, cv2.COLOR_BGR2HSV, dst=self.hsv)
        cv2.inRange(self.hsv, self.thresh_low, self.thresh_high, dst=self.bin)

        cv2.morphologyEx(self.bin, cv2.MORPH_CLOSE, self.morphKernel, dst=self.bin2, iterations=1)

        if self.draw_thresh:
            b = (self.bin2 != 0)
            cv2.copyMakeBorder(self.black, 0, 0, 0, 0, cv2.BORDER_CONSTANT, value=self.RED, dst=self.out)
            self.out[np.dstack((b, b, b))] = 255

        return self.bin2

    def find_contours(self, img):
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.preallocate(img)
        thresh_img = self.threshold(img)

        _, contours, _ = cv2.findContours(thresh_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        result = []
        for cnt in contours:
            approx = cv2.approxPolyDP(cnt, 0.01 * cv2.arcLength(cnt, True), True)

            if self.draw_approx:
                cv2.drawContours(self.out, [approx], -1, self.BLUE, 2, lineType=8)

            if len(approx) > 3 and len(approx) < 15:
                _, _, w, h = cv2.boundingRect(approx)
                if h > self.min_height and w > self.min_width:
                    hull = cv2.convexHull(cnt)
                    approx2 = cv2.approxPolyDP(hull, 0.01 * cv2.arcLength(hull, True), True)

                    if self.draw_approx2:
                        cv2.drawContours(self.out, [approx2], -1, self.GREEN, 2, lineType=8)

                    result.append(approx2)
        return result

    def angle_cos(self, p0, p1, p2):
        d1, d2 = (p0 - p1).astype('float'), (p2 - p1).astype('float')
        return abs(np.dot(d1, d2) / np.sqrt(np.dot(d1, d1) * np.dot(d2, d2)))

        # # returns sequence of squares detected on the image.
    # def find_squares(self, image, squares):
    #     img_copy = image.copy()
    #     # print image
    #     # (h, w, d) = image.shape
    #     # print "width={}, height={}, depth={}".format(w, h, d)
    #     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #     ret, thresh = cv2.threshold(gray, 127, 255, 0)
    #     im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     nts = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    #
    #     rects = []
    #     for c in contours:
    #         peri = cv2.arcLength(c, True)
    #         approx = cv2.approxPolyDP(c, 0.02 * peri, True)
    #         x, y, w, h = cv2.boundingRect(approx)
    #         if h >= 15:
    #             # if height is enough
    #             # create rectangle for bounding
    #             rect = (x, y, w, h)
    #             rects.append(rect)
    #             cv2.rectangle(img_copy, (x, y), (x + w, y + h), (0, 255, 0), 1);
    #
    #     for cnt in contours:
    #         canvas = np.zeros(image.shape, np.uint8)
    #
    #         # define main island contour approx. and hull
    #         perimeter = cv2.arcLength(cnt, True)
    #         epsilon = 0.01 * cv2.arcLength(cnt, True)
    #         approx = cv2.approxPolyDP(cnt, epsilon, 10, True)
    #
    #         if math.fabs(cv2.contourArea(cnt)) > 500:
    #             squares.append(approx)
    #         if len(approx) == 4 and math.fabs(cv2.contourArea(cnt)) > 500 and cv2.isContourConvex(cnt):
    #             print "ddddddddd", len(approx) == 4, math.fabs(cv2.contourArea(cnt)), cv2.isContourConvex(cnt)
    #             max_cosine = 0
    #             for i in range(2, 5):
    #                 cosine = math.fabs(self.angle(approx[i%4], approx[i-2], approx[i-1]))
    #                 max_cosine = max(max_cosine, cosine)
    #                 print "max_cosine  ::", max_cosine, squares
    #                 if max_cosine < 0.3:
    #                     squares.append(approx)
    #     return squares, rects, img_copy

    def angle(self, pt1, pt2, pt0):
        dx1 = pt1.x - pt0.x;
        dy1 = pt1.y - pt0.y;
        dx2 = pt2.x - pt0.x;
        dy2 = pt2.y - pt0.y;
        return (dx1*dx2 + dy1*dy2)/math.sqrt((dx1*dx1 + dy1*dy1)*(dx2*dx2 + dy2*dy2) + 1e-10);


if __name__ == '__main__':
    image_list = ['/home/user/Downloads/5bbec42e4b226-20181011090158.jpg']
    imgp = ImageProcessing(image_list)
    imgp.get_images()


# python scripts/image_processing.py
