import cv2
img = cv2.imread("examples/apple_01.jpg")
cv2.imshow('res',img)
cv2.waitKey(0)
cv2.destroyAllWindows()