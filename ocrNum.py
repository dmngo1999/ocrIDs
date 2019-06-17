try:
    from PIL import Image
except ImportError:
    import Image
import pytesseract
import cv2
import numpy as np
import re

pytesseract.pytesseract.tesseract_cmd = r'C:\\Users\\MinhND34\\AppData\\Local\\Tesseract-OCR\\tesseract.exe'

def readID(filename):
    image = cv2.imread(filename)

    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

    lowerNum = (110, 42, 19)
    upperNum = (251, 238, 233)

    maskNum = cv2.inRange(hsv, lowerNum, upperNum)
    config = ("--oem 1 --psm 3")
    resultNum = pytesseract.image_to_string(maskNum, config=config)
    result = re.sub("[^1234567890]", "", resultNum)
    print(result)

    cv2.imshow("PicNum", maskNum)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
readID('C:/Users/MinhND34/Desktop/ImageTest/cmnd.jpg')