import cv2 as cv

#demo image
img=cv.imread('Photos/group 1.jpg')
cv.imshow('Original Picture',img)

#face detection is done on gray scaled image
gray=cv.cvtColor(img,cv.COLOR_BGR2GRAY)

#using the classifier
haar_cascade=cv.CascadeClassifier('haar_face.xml')

faces_rect=haar_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=1)
print(f"Number of faces found = {len(faces_rect)}")

#drawing a rectangle on the detected face
for (x,y,w,h) in faces_rect:
    cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
cv.imshow('Detected Face',img)


cv.waitKey(0)