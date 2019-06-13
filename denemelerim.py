import cv2
import urllib
import numpy as np



url = "http://192.168.1.30:8080/shot.jpg"

# Initialize webcam feed
#video = cv2.VideoCapture(1)
#if video.isOpened() :
#    print("opened")
#ret = video.set(3,1080)
#ret = video.set(4,720)
xy = 0
ax=0
while(True):
    imgResp = urllib.request.urlopen(url)

    # Numpy to convert into a arrayqq
    imgNp = np.array(bytearray(imgResp.read()), dtype=np.uint8)

    # Finally decode the array to OpenCV usable format ;)

    frame = cv2.imdecode(imgNp, -1)
    # Acquire frame and expand frame dimensions to have shape: [1, None, None, 3]
    # i.e. a single-column array, where each item in the column has the pixel RGB value
    #ret, frame = video.read(0)
    #frame = cv2.flip(frame, 1)



    cv2.imshow('Object detector', frame)
    # Press 'q' to quit
    if chr(cv2.waitKey(0)&255) == 'q':
            break



#find_screen.pokedex_find_screen(crop_img)
# Clean up
#video.release()
cv2.destroyAllWindows()