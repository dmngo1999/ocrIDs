try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
from imutils.object_detection import non_max_suppression
import time
import re
import os

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\MinhND34\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'


def readText(filename):
    imagePath = filename
    east = 'C:/Users/MinhND34/Desktop/EASTTEST/frozen_east_text_detection.pb'
    min_confidence = 0.3
    width = 640
    height = 320
    paddingX = 0.05
    paddingY = 0.15

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
    #print("loading EAST")
    net = cv2.dnn.readNet(east)

    #convert to blob and forward pass of the model to obtain the prob and geometry sets
    #AKA MAGIC

    blob = cv2.dnn.blobFromImage(image, 1.0, (W, H), (123.68, 116.78, 103.94), swapRB=True, crop=False)
    net.setInput(blob)
    (scores, geometry) = net.forward(layerNames)

    def decode(scores, geometry):
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

                #get rotational angle for the prediction then get sin and cos
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
        return (rects, confidences)
        
    (rects, confidences) = decode(scores, geometry)
    #apply non-maxima suppression to weak and overlapping boxes
    boxes = non_max_suppression(np.array(rects), probs=confidences)
    yCoordStart = []
    yCoordEnd = []
    xCoordStart = []
    xCoordEnd = []

    target = []
    birthday = []
    gender = []
    nationality = []
    idNumber = []
    gender = []
    genderSplit = 0
    expire = []

    #loop over the boxes
    for (startX, startY, endX, endY) in boxes:
        yCoordStart.append(startY)
        xCoordStart.append(startX)
        yCoordEnd.append(endY)
        xCoordEnd.append(endX)

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
        #cv2.rectangle(orig, (startX, startY), (endX, endY), (0,255,0), 2)

        #Apply Tesseract
        config = ("-l vie --oem 1 --psm 7")
        text = pytesseract.image_to_string(roi, config=config)

        print(text, " ", "(", startX, ", ", startY, ", ", endX, ", ", endY, ")")
        #find ho va ten and break
        if text.find('Họ') != -1 or text.find('và') != -1 or text.find('tên') != -1:
            target.append(text)
            target.append(startY)
            target.append(endY)
            target.append(startX)

        if text.find('Ngày') != -1 or text.find('tháng') != -1 or text.find('năm') != -1 or text.find('sinh') != -1:
            birthday.append(text)
            birthday.append(startY)
            birthday.append(endY)
            birthday.append(startX)

        if text.find('Quốc') != -1:
            genderSplit = startX

        if text.find('Giới') != -1 or text.find('tính') != -1:
            gender.append(text)
            gender.append(startY)
            gender.append(endY)
            gender.append(startX)

        if text.find('Quốc') != -1 or text.find('tịch') != -1:
            nationality.append(text)
            nationality.append(startY)
            nationality.append(endY)
            nationality.append(startX)
            
        if text.find('Số') != -1:
            idNumber.append(text)
            idNumber.append(startY)
            idNumber.append(endY)
            idNumber.append(endX)

        if text.find('đến') != -1 or text.find('Có') != -1 or text.find('giá') != -1 or text.find('trị') != -1:
            expire.append(text)
            expire.append(startY)
            expire.append(endY)
            expire.append(endX)


    def detail_func(target, is_digit, is_gender, is_expire):
        officialStartY = target[1] - 10
        officialEndY = target[2] + 8
        officialStartX = target[3]
        if is_expire:
            officialStartY = target[1] - 5
            officialEndY = target[2]

        officialImg = orig[officialStartY:officialEndY, officialStartX:origW]
        if is_gender:
            officialImg = orig[officialStartY:officialEndY, officialStartX:genderSplit]

        official2 = officialImg.copy()
        (offH, offW) = officialImg.shape[:2]

        #set new dimensions for pictures and destermine the ratio
        (newerH, newerW) = (160, 960)
        if is_gender:
            (newerH, newerW) = (160, 320)
        r2H = offH / float(newerH)
        r2W = offW / float(newerW)

        #resize actual image to new dimensions and save them to dimension variables
        officialImg = cv2.resize(officialImg, (newerW, newerH))
        (H2, W2) = officialImg.shape[:2]

        #load EAST text detector
        #print("loading EAST again")

        #convert to blob and forward pass of the model to obtain the prob and geometry sets
        #AKA MAGIC
        layerNames2 = ["feature_fusion/Conv_7/Sigmoid","feature_fusion/concat_3"]


        idName = []

        #loop over the boxes
        if(is_digit == True):
            
            hsv = cv2.cvtColor(official2, cv2.COLOR_RGB2GRAY)
   
            if is_expire:
                config = ("-l vie --oem 1 --psm 11")
                text2 = pytesseract.image_to_string(hsv, config=config)
                result = re.sub("[^1234567890/]Khônghạnthời", "", text2)
                result = result.split('\n')[0]

            else:
                config = ("--oem 1 --psm 11")
                text2 = pytesseract.image_to_string(hsv, config=config)
                result = re.sub("[^1234567890/]", "", text2)
            
            #cv2.imshow("rwgr", hsv)
            #cv2.imwrite(os.path.join('C:/Users/MinhND34/Pictures/Screenshots', 'testImgBD.jpg'), officialImg)
            #cv2.imshow("ewrg", official2)

            return result

        else:
            
            net2 = cv2.dnn.readNet(east)
            blob2 = cv2.dnn.blobFromImage(officialImg, 1.0, (W2, H2), (123.68, 116.78, 103.94), swapRB=True, crop=False)
            net2.setInput(blob2)
            (scores2, geometry2) = net2.forward(layerNames2)

            (rects2, confidences2) = decode(scores2, geometry2)

            #apply non-maxima suppression to weak and overlapping boxes
            boxes2 = non_max_suppression(np.array(rects2), probs=confidences2)

            paddingX2 = 0.07
            paddingY2 = 0.25

            
            for (startX2, startY2, endX2, endY2) in reversed(boxes2):

                #scale the boxes\ coords based on ratio found in the start
                startX2 = int(startX2 * r2W)
                startY2 = int(startY2 * r2H)
                endX2 = int(endX2 * r2W)
                endY2 = int(endY2 * r2H)

                #make padding
                dX2 = int((endX2 - startX2) * paddingX2)
                dY2 = int((endY2 - startY2) * paddingY2)
                startX2 = max(0, startX2 - dX2)
                startY2 = max(0, startY2 - dY2)
                endX2 = min(offW, endX2 + (dX2*2))
                endY2 = min(offH, endY2 + (dY2*2))


                #Get the ROI
                roi2 = official2[startY2:endY2, startX2:endX2]

                #draw boxes
                cv2.rectangle(official2, (startX2, startY2), (endX2, endY2), (0,255,0), 2)

                #Apply Tesseract
                config = ("-l vie --oem 1 --psm 7")
                text2 = pytesseract.image_to_string(roi2, config=config)
                print(text2)


                #find ho va ten and break
                lineRead = open("C:/Users/MinhND34/AppData/Local/Programs/Python/Python37/work/wordTest.txt", mode="r", encoding="utf-8-sig")
                thisList = lineRead.read()
                line = thisList.split(", ")

                keep = True
                for i in line:
                    if (text2.find(i) >= 0):
                        keep = False
                        print(text2, "FOUND")
                if(keep == True):
                    idName.append(text2)
            #cv2.imshow("rwgr", officialImg)
            #cv2.imshow("ewrg", official2)
            idNamestr = ' '.join(idName)
            idNamestr = re.sub(r"^\s+", "", idNamestr)
            print(idNamestr)
            return idNamestr

    final_nationality = detail_func(nationality, False, False, False)
    final_name = detail_func(target, False,False, False)
    final_birthday = detail_func(birthday, True, False, False)
    final_id = detail_func(idNumber, True, False, False)
    final_gender = detail_func(gender, False, True, False)
    final_expire = detail_func(expire, True, False, True)

    cv2.waitKey(0)


    return (final_name, final_nationality, final_birthday, final_id, final_gender, final_expire)


def main(filename):
    start = time.time()
    (name, nationality, birthday, idNum, gender, expire) = readText(filename)
    end = time.time()
    print("Code takes:", end - start, "seconds")
    print("Name:", name)
    print("ID Number:", idNum)
    print("Gender:", gender)
    print("Nationality:", nationality)
    print("Birthday:", birthday)
    print("Expire date:", expire)

    
    return (name, nationality, birthday, idNum, gender, end-start, expire)



#main('C:/Users/MinhND34/Desktop/ImageTest/1cmnd.jpg')