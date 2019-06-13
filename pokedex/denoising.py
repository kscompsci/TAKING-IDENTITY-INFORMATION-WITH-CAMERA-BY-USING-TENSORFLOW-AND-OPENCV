import numpy as np
import cv2
from matplotlib import pyplot as plt
def denoising(img):
    #img = cv2.imread('aaa.jpg')
    #img = cv2.resize(img,(1000,600))# Denoising

    #cv2.imshow("asdad", img)
    """b,g,r = cv2.split(img)           # get b,g,r
    rgb_img = cv2.merge([r,g,b])     # switch it to rgb"""
    #img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    dst = cv2.fastNlMeansDenoising(img,None,30,5,21)

    return dst
    #cv2.imshow('asd',dst)
    #cv2.imshow("asdf",rgb_dst)
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()