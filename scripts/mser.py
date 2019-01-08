# import cv2
# import numpy as np
#
# #Create MSER object
# mser = cv2.MSER_create()
#
# #Your image path i-e receipt path
# img = cv2.imread('/home/user/Downloads/5bbec42e4b226-20181011090158.jpg')
#
# #Convert to gray scale
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#
# vis = img.copy()
#
# #detect regions in gray scale image
# regions, _ = mser.detectRegions(gray)
#
# hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
#
# cv2.polylines(vis, hulls, 1, (0, 255, 0))
#
# cv2.imshow('img', vis)
#
# cv2.waitKey(0)
#
# mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)
#
# for contour in hulls:
#
#     cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)
#
# #this is used to find only text regions, remaining are ignored
# text_only = cv2.bitwise_and(img, img, mask=mask)
# print text_only
# cv2.imshow("text only", text_only)
#
# cv2.waitKey(0)

import cv2
img = cv2.imread('/home/user/Downloads/5bbec42e4b226-20181011090158.jpg');
print img
(h, w, d) = img.shape
print "width={}, height={}, depth={}".format(w, h, d)
vis = img.copy()
mser = cv2.MSER_create()
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
regions, _ = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]
cv2.polylines(vis, hulls, 1, (0, 255, 0))
cv2.imshow('img', vis)
cv2.waitKey(0)
cv2.destroyAllWindows()

roi = img[60:160, 320:420]
cv2.imshow("ROI", roi)
cv2.waitKey(0)
