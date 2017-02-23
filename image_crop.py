import cv2
import sys

# Get user supplied values
imagePath = input("Hey there, Please enter the image name: ")
cascPath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "nose_cascade.xml"
eyeCascadeFilePath = "frontal_eyes.xml"

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cascPath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
eyeCascade = cv2.CascadeClassifier(eyeCascadeFilePath)
# Read the image
image = cv2.imread(imagePath)
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Detect faces in the image
faces = faceCascade.detectMultiScale(
    gray,
    scaleFactor=1.1,
    minNeighbors=5,
    minSize=(30, 30),
    flags = cv2.cv.CV_HAAR_SCALE_IMAGE
)

# Draw a rectangle around the faces
for (x, y, w, h) in faces:
	cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 1)
	roi_color = image[y:y+h, x:x+w]
	roi_gray = gray[y:y+h, x:x+w]
	cv2.imwrite("roi.png", roi_color)
	nose = noseCascade.detectMultiScale(roi_gray)
	eyes = eyeCascade.detectMultiScale(roi_gray)
	
	for (nx,ny,nw,nh) in nose:
		cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),1)
		roi_nose = roi_color[ny:ny+nh, nx:nx+nw]
		cv2.imwrite("roi_nose.png", roi_nose)
	
	for (ex,ey,ew,eh) in eyes:
		cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(255,0,0),1)
		roi_eyes = roi_color[ey:ey+eh, ex:ex+ew]
		cv2.imwrite("roi_eyes.png", roi_eyes)
	
cv2.waitKey(0)