import cv2

faceCascadeFilePath = "haarcascade_frontalface_default.xml"
noseCascadeFilePath = "nose_cascade.xml"
imagePath = "kavi2.jpg"
 
faceCascade = cv2.CascadeClassifier(faceCascadeFilePath)
noseCascade = cv2.CascadeClassifier(noseCascadeFilePath)
 

imgMustache = cv2.imread('mustache.png',-1)
 

orig_mask = imgMustache[:,:,3]
 
orig_mask_inv = cv2.bitwise_not(orig_mask)
 

imgMustache = imgMustache[:,:,0:3]
origMustacheHeight, origMustacheWidth = imgMustache.shape[:2]
 

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

print "Found {0} faces!".format(len(faces))


for (x, y, w, h) in faces:
        # Un-comment the next line for debug (draw box around all faces)
        face = cv2.rectangle(image,(x,y),(x+w,y+h),(0,234,7),1)
 
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = image[y:y+h, x:x+w]
 
        # Detect a nose within the region bounded by each face (the ROI)
        nose = noseCascade.detectMultiScale(roi_gray)
 
        for (nx,ny,nw,nh) in nose:
            #cv2.rectangle(roi_color,(nx,ny),(nx+nw,ny+nh),(255,0,0),2)
 
            # The mustache is kept three times the width of the nose
            mustacheWidth =  3 * nw
            mustacheHeight = mustacheWidth * origMustacheHeight / origMustacheWidth
 
            # To Center the mustache on the bottom of the nose
            x1 = nx - (mustacheWidth/4)
            x2 = nx + nw + (mustacheWidth/4)
            y1 = ny + nh - (mustacheHeight/2)
            y2 = ny + nh + (mustacheHeight/2)
 
            # Re-calculate the width and height of the mustache image
            mustacheWidth = x2 - x1
            mustacheHeight = y2 - y1
 
            # Re-size the original image and the masks to the mustache sizes calcualted above
            mustache = cv2.resize(imgMustache, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask = cv2.resize(orig_mask, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
            mask_inv = cv2.resize(orig_mask_inv, (mustacheWidth,mustacheHeight), interpolation = cv2.INTER_AREA)
 
            # take ROI for mustache from background equal to size of mustache image
            roi = roi_color[y1:y2, x1:x2]
 
            # roi_bg contains the original image only where the mustache is not in the region that is the size of the mustache.
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
 
            # roi_fg contains the image of the mustache only where the mustache is
            roi_fg = cv2.bitwise_and(mustache,mustache,mask = mask)
 
            # join the roi_bg and roi_fg
            dst = cv2.add(roi_bg,roi_fg)
 
            # place the joined image, saved to dst back over the original image
            roi_color[y1:y2, x1:x2] = dst
 
            break
 
    # Display the resulting frame
cv2.imshow('Video', image)
cv2.waitKey(0)	