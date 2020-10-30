import cv2
import imutils
import numpy as np
import urllib
import requests
import json
'''

THE PASSPORT PICTURE SHOULD BE CLEAR !!!


'''
PASSPORT_PICTURE = "" # THE INPUT IMAGE
OUTPUT_NAME = "" # THE OUTPUT NAME
#Read the image
img = cv2.imread(PASSPORT_PICTURE)

# shape of the image
h,w_,_ = img.shape

# appling blurs , thresholds and some transformations to detect the MRZ CODE
img = cv2.resize(img,(600,500))
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (3, 3), 0)
rectKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (21, 21))
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKernel)
gradX = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
gradX = np.absolute(gradX)
(minVal, maxVal) = (np.min(gradX), np.max(gradX))
gradX = (255 * ((gradX - minVal) / (maxVal - minVal))).astype("uint8")
gradX = cv2.morphologyEx(gradX, cv2.MORPH_CLOSE, sqKernel)
threshold = cv2.threshold(gradX, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, sqKernel)
thresh = cv2.erode(thresh, None, iterations=2)
p = int(img.shape[1] * 0.05)
thresh[:, 0:p] = 0
thresh[:, img.shape[1] - p:] = 0
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv2.contourArea, reverse=True)

#Extract bounding box of the MRZ CODE
for c in cnts:
	(x, y, w, h) = cv2.boundingRect(c)
	ar = w / float(h)
	crWidth = w / float(gray.shape[1])
	if ar > 5 and crWidth > 0.75:
		pX = int((x + w) * 0.03)
		pY = int((y + h) * 0.03)
		(x, y) = (x - pX, y - pY)
		(w, h) = (w + (pX * 2), h + (pY * 2))

		#roi variable contains the croped image (MRZ CODE)
		roi = img[y:y + h, x:x + w].copy()
		cv2.imwrite(OUTPUT_NAME,roi)
		#if you want to draw a bounding rectangle
		#cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
		break
OUTPUT_PATH_URL = "" #CHANGE IT TO THE URL OF THE OUTPUT IMAGE
#GET REQUEST
params ={"apikey":"03a0ae683a88957",
        "url":OUTPUT_PATH_URL,
		"isOverlayRequired":"true",
		"OCREngine":2}
api_url = "https://api.ocr.space/parse/imageurl?"
params = urllib.parse.urlencode(params)
response = urllib.request.urlopen(api_url+params)
#READ THE DATA
data = json.load(response)
first_line = data['ParsedResults'][0]['TextOverlay']['Lines'][0]['LineText']
second_line = data['ParsedResults'][0]['TextOverlay']['Lines'][1]['LineText']
# MRZ CODE
print(first_line)
print(second_line)

