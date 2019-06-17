try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import time

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\MinhND34\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

imagePath = 'C:/Users/MinhND34/Desktop/ImageTest/1cmnd.jpg'
east = 'C:/Users/MinhND34/Desktop/EASTTEST/frozen_east_text_detection.pb'
min_confidence = 0.5
width = 320
height = 640
paddingX = 0.05
paddingY = 0.2

start = time.time()
image = cv2.imread(imagePath)
orig = image.copy()
(origH, origW) = image.shape[:2]

#set new dimensions for pictures and destermine the ratio
(newH, newW) = (height, width)
rH = origH / float(newH)
rW = origW / float(newW)

#resize actual image to new dimensions and save them to dimension variables
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

#layerNames for EAST detector model - probabilities of text - derive box coords
layerNames = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]

#load EAST text detector
print("loading EAST")
net = cv2.dnn.readNet(east)

#convert to blob and forward pass of the model to obtain the prob and geometry sets
#AKA MAGIC

blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)

#get number of rows and cols from scores array
#rects is a array of coords for text regions
#confidences sores prob inrespect to boxes from rects
(numRows, numCols) = scores.shape[2:4]
rects=[]
confidences=[]

#loop over the rows
for y in range(0, numRows):
    #get scores and geometry to derive coords to surround text
    scoresData = scores[0,0,y]
    xData0 = geometry[0,0,y]
    xData1 = geometry[0,1,y]
    xData2 = geometry[0,2,y]
    xData3 = geometry[0,3,y]
    anglesData = geometry[0,4,y]
    for x in range(0, numCols):
        #ignore data lower then min confidence
        if scoresData[x] < min_confidence:
            continue
        #compute offset factor because the resulting map is 4 tiems smaller than the input image
        (offsetX, offsetY) = (x*4.0, y*4.0)

        #get rotational angle for the prediction than get sin and cos
        angle = anglesData[x]
        cos = np.cos(angle)
        sin = np.sin(angle)

        #get width and length of bounding boixes for texts
        h = xData0[x] + xData2[x]
        w = xData1[x] + xData3[x]

        #Get starting and ending coords of bounding boxes
        endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
        endY = int(offsetY + (sin * xData1[x]) + (cos * xData2[x]))
        startX = int(endX - w)
        startY = int(endY - h)

        #add coords and prob scores to list
        rects.append((startX, startY, endX, endY))
        confidences.append(scoresData[x])

#apply non-maxima suppression to weak and overlapping boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)
#loop over the boxes
for (startX, startY, endX, endY) in boxes:
    #scale the boxes\ coords based on ratio found in the start
    startX = int(startX * rW)
    startY = int(startY * rH)
    endX = int(endX * rW)
    endY = int(endY * rH)

    #make padding
    dX = int((endX - startX) * paddingX)
    dY = int((endY - startY) * paddingY)
    startX = max(0, startX - dX)
    startY = max(0, startY - dY)
    endX = min(origW, endX + (dX*2))
    endY = min(origH, endY + (dY*2))

    #Get the ROI
    roi = orig[startY:endY, startX:endX]

    #draw boxes
    cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,0), 2)

    #Apply Tesseract
    config = ("-l vie --oem 1 --psm 7")
    text = pytesseract.image_to_string(roi, config=config)
    print(text, " ", "(", startX, ", ", startY, ", ", endX, ", ", endY, ")")

end = time.time()
print("TIME:", (end-start))

#show image
cv2.imshow("Text Detection", orig)
cv2.waitKey(0)







