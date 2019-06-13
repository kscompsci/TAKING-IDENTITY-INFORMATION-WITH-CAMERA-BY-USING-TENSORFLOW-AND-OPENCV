import numpy as np
import cv2
def auto_contrast(img):


    """img = cv2.imread('aaa.jpg')
    img = cv2.resize(img,(1000,600))"""

    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    #cv2.imshow("asdasd", img)
    # create a CLAHE o, bject (Arguments are optional).
    clahe = cv2.createCLAHE(clipLimit=0.5, tileGridSize=(8,8))
    cl1 = clahe.apply(img)
    return cl1
"""cv2.imshow('clahe_2.jpg',cl1)
cv2.waitKey(0)
cv2.destroyAllWindows()"""