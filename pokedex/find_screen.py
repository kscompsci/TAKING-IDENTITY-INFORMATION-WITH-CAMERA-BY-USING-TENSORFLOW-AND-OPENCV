# USAGE
# python find_screen.py --query queries/query_marowak.jpg
import pytesseract
from PIL import Image
# import the necessary packages
from skimage import exposure
import numpy as np
import argparse
import imutils
from text_detection.ctpn import demo_pb
import cv2
from .denoising import denoising
import matplotlib.pyplot as plt
from .auto_contrast import auto_contrast
# construct the argument parser and parse the arguments
"""ap = argparse.ArgumentParser()
ap.add_argument("-q", "--query", required = True,
	help = "Path to the query image")
args = vars(ap.parse_args())"""

# load the query image, compute the ratio of the old height
# to the new height, clone it, and resize it
def pokedex_find_screen(image, temp, flagForSteps):
	#image = cv2.imread("queries/asss.jpg")
	#image = auto_contrast(image)
	ratio = image.shape[0] / 300.0
	orig = image.copy()
	image = imutils.resize(image, height = 300)
	tcnum = "tc bulunamadi"
	# convert the image to grayscale, blur it, and find edges
	# in the image
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	gray = cv2.bilateralFilter(gray, 11, 17, 17)
	edged = cv2.Canny(gray, 30, 200)

	# find contours in the edged image, keep only the largest
	# ones, and initialize our screen contour
	cnts = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
	cnts = imutils.grab_contours(cnts)
	cnts = sorted(cnts, key = cv2.contourArea, reverse = True)[:10]
	screenCnt = None

	# loop over our contours
	for c in cnts:
		# approximate the contour
		peri = cv2.arcLength(c, True)
		approx = cv2.approxPolyDP(c, 0.015 * peri, True)

		# if our approximated contour has four points, then
		# we can assume that we have found our screen
		if len(approx) == 4:
			screenCnt = approx
			break

	# now that we have our screen contour, we need to determine
	# the top-left, top-right, bottom-right, and bottom-left
	# points so that we can later warp the image -- we'll start
	# by reshaping our contour to be our finals and initializing
	# our output rectangle in top-left, top-right, bottom-right,
	# and bottom-left order
	try:
		pts = screenCnt.reshape(4, 2)
		rect = np.zeros((4, 2), dtype = "float32")
	except:
		print("The photo is not good enough for processing! Please try again with a better one.")
		return "okunamadi"
		cv2.waitKey(0)
	# the top-left point has the smallest sum whereas the
	# bottom-right has the largest sum

	s = pts.sum(axis = 1)

	rect[0] = pts[np.argmin(s)]
	rect[2] = pts[np.argmax(s)]

	# compute the difference between the points -- the top-right
	# will have the minumum difference and the bottom-left will
	# have the maximum difference
	diff = np.diff(pts, axis = 1)
	rect[1] = pts[np.argmin(diff)]
	rect[3] = pts[np.argmax(diff)]

	# multiply the rectangle by the original ratio
	rect *= ratio

	# now that we have our rectangle of points, let's compute
	# the width of our new image
	(tl, tr, br, bl) = rect
	widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
	widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))

	# ...and now for the height of our new image
	heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
	heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))

	# take the maximum of the width and height values to reach
	# our final dimensions
	maxWidth = max(int(widthA), int(widthB))
	maxHeight = max(int(heightA), int(heightB))

	# construct our destination points which will be used to
	# map the screen to a top-down, "birds eye" view
	dst = np.array([
		[0, 0],
		[maxWidth - 1, 0],
		[maxWidth - 1, maxHeight - 1],
		[0, maxHeight - 1]], dtype = "float32")

	# calculate the perspective transform matrix and warp
	# the perspective to grab the screen
	M = cv2.getPerspectiveTransform(rect, dst)
	warp = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
	img_rgb = warp
	# convert the warped image to grayscale and then adjust
	# the intensity of the pixels to have minimum and maximum
	# values of 0 and 255, respectively
	if (flagForSteps == 1):
		cv2.imshow("before", warp)
	warp = denoising(warp)
	if (flagForSteps == 1):
		cv2.imshow("after denoising", warp)
	warp = auto_contrast(warp)
	if (flagForSteps == 1):
		cv2.imshow("after contrast", warp)
	warp = cv2.resize(warp, (1000, 500))
	img_rgb = cv2.resize(img_rgb, (1000, 500))


	"""f temp == 1:
		crop_img = warp[103:133, 747:961]
		cv2.imshow("cropped" , crop_img)
		cropped_text = pytesseract.image_to_string(crop_img, lang="tur")
		print("tc kimlik numarası" + cropped_text)"""

	text = pytesseract.image_to_string(warp, lang="tur")
	flag=0
	temp1=""
	temp2=""
	flag2 = 0
	flag3=0
	for i in text:
		if i=="0" or i=="1" or i=="2" or i=="3" or i=="4" or i=="5" or i=="6" or i=="7" or i=="8" or i == "9":
			temp1+=i
			flag+=1
			if flag==11:
				tcnum = temp1
				print ("TC Kimlik Numarası = "+temp1)
				flag2 = 1
		elif flag2==1:
			if flag3==1 :
				if i != "\n":
					temp2 += i
				else:
					flag2 = 0
					flag3 = 0
					if temp == 1:
						print("isim : "+temp2)
			else :
				flag3=1



		else:
			flag =0
			temp1=""
	flag4 = 0
	flag5 = 0
	textForSurname = ""
	surname = ""
	flag6 = 0

	counter = 0
	textForSurname2=[]
	for i in text:
		if i != " " and i!="\n":
			textForSurname +=i

		else:
			if textForSurname != "":
				textForSurname2.append(textForSurname)

			textForSurname = ""
	flag7=0
	for i in textForSurname2:
		if (flag7 == 1 and temp == 1):
			print("Soyadı : " + i)
			flag7 = 0
		if i=="SURNAME":
			flag7 = 1




	#print(text)

	#demo_pb.text_detection_ctpn(img_rgb)

	"""warp = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
	warp = exposure.rescale_intensity(warp, out_range = (0, 255))

	# the pokemon we want to identify will be in the top-right
	# corner of the warped image -- let's crop this region out
	(h, w) = warp.shape
	(dX, dY) = (int(w * 0.4), int(h * 0.45))
	crop = warp[10:dY, w - dX:w - 10]

	# save the cropped image to file
	cv2.imwrite("cropped.png", crop)"""

	# show our images
	#cv2.imshow("image", image)
	#cv2.imshow("edge", edged)
	#cv2.imshow("warp", imutils.resize(warp, height = 300))
	#cv2.imwrite("asddsa.png",warp)
	"""text = pytesseract.image_to_string(image, lang="tur")
	print(text)"""
	#cv2.imshow("crop", imutils.resize(crop, height = 300))
	#cv2.imshow("crop", imutils.resize(crop, height = 300))
	#cv2.waitKey(0)
	return tcnum