import cv2
from cvzone.HandTrackingModule import HandDetector
import numpy as np
import math


cap = cv2.VideoCapture(0)

#As we want only One hand to be recognized by the camera for input - (maxHANDS=1)
detector = HandDetector(maxHands=1)

#for the bounding box to have more space at the edge or border
offset = 20
imgSize = 300 #we are setting the size of the image to 500px

while True:
    success, img = cap.read()
    hands, img = detector.findHands(img)

    #Cropping the Image
    if hands:
        hand =hands[0]
        x,y,w,h = hand['bbox']

        imgWhite = np.ones((imgSize,imgSize,3),np.uint8)*255 #the 255 is for the white color
        imgCrop = img[y-offset:y+h+offset, x-offset:x+w+offset]

        imgCropShape = imgCrop.shape

        aspectRatio = h / w

        if aspectRatio > 1:
            k = imgSize / h
            wCal = math.ceil(k * w)
            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
            imgResizeShape = imgResize.shape
            wGap = math.ceil((imgSize - wCal) / 2)
            imgWhite[:, wGap:wCal + wGap] = imgResize

        else:
            k = imgSize / w
            hCal = math.ceil(k * h)
            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
            imgResizeShape = imgResize.shape
            hGap = math.ceil((imgSize - hCal) / 2)
            imgWhite[hGap:hCal + hGap, :] = imgResize



        cv2.imshow("ImageCrop", imgCrop)
        cv2.imshow("ImageWhite", imgWhite)

    cv2.imshow("Image",img)

    #for 1 milli second DELAY - (cv2.waitkey)
    cv2.waitKey(1)
