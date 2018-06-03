import cv2
import numpy as np

frame = cv2.imread("examples/apple_01.jpg")
# Convert BGR to HSV
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV
lower_blue = np.array([0,43,46])
upper_blue = np.array([10,255,255])
# Threshold the HSV image to get only blue colors

mask = cv2.inRange(hsv, lower_blue, upper_blue)
# Bitwise-AND mask and original image
res = cv2.bitwise_and(frame,frame, mask=mask)
resgray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
resgraycanny=cv2.Canny(resgray,50,100)
image, contours, hier = cv2.findContours(resgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    x, y, w, h = cv2.boundingRect(c)
    cv2.rectangle(res, (x, y), (x + w, y + h), (255, 0, 0), 2)
cv2.imshow('resgraycanny',resgraycanny)
cv2.imshow('resgray',resgray)
cv2.imshow('frame',frame)
cv2.imshow('mask',mask)
cv2.imshow('res',res)
k = cv2.waitKey(5) & 0xFF
cv2.waitKey(0)
cv2.destroyAllWindows()