# USAGE
# python test_network.py --model tomato_not_tomato.model --image images/examples/tomato_01.jpg

# import the necessary packages
from keras.preprocessing.image import img_to_array
from keras.models import load_model
import numpy as np
import argparse
import imutils
import cv2

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
	help="path to trained model model")
ap.add_argument("-i", "--image", required=True,
	help="path to input image")
args = vars(ap.parse_args())

# load the image
image = cv2.imread(args["image"])
orig = image.copy()

# pre-process the image for classification
image = cv2.resize(image, (28, 28))
image = image.astype("float") / 255.0
image = img_to_array(image)
image = np.expand_dims(image, axis=0)

# load the trained convolutional neural network
print("[INFO] loading network...")
model = load_model(args["model"])

# classify the input image
(nottomato, tomato) = model.predict(image)[0]

# build the label
label = "tomato" if tomato > nottomato else "Not tomato"
proba = tomato if tomato > nottomato else nottomato
label2 = "{}: {:.2f}%".format(label, proba * 100)

# draw the label on the image
output = cv2.resize(orig,(600,350))
height, width, channels = output.shape
if label=="tomato":
    # Convert BGR to HSV
    hsv = cv2.cvtColor(output, cv2.COLOR_BGR2HSV)
    # define range of blue color in HSV
    lower_blue = np.array([0,43,46])
    upper_blue = np.array([10,255,255])
    # Threshold the HSV image to get only blue colors
    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    # Bitwise-AND mask and original image
    res = cv2.bitwise_and(output,output, mask=mask)
    resgray=cv2.cvtColor(res,cv2.COLOR_BGR2GRAY)
    image, contours, hier = cv2.findContours(resgray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if(w and h)>(height/10):
            cv2.rectangle(output, (x, y), (x + w, y + h), (255, 0, 0), 5)
cv2.putText(output, label2, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,
	0.7, (0, 255, 0), 2)

# show the output image
cv2.imshow("Output", output)
cv2.waitKey(0)