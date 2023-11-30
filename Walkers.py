import cv2


# Create our body classifier
body_classifier = cv2.CascadeClassifier("haarcascade_fullbody.xml")

# Initiate video capture for video file
cap = cv2.VideoCapture('walking.avi')


# Loop once video is successfully loaded
while True:
    
    # Read first frame
    ret, frame = cap.read()

    #Convert Each Frame into Grayscale
    for i in range(100):
        grey = cv2.cvtColor(cap,cv2.COLOR_BGR2GRAY)
    # Pass frame to our body classifier
    bodies=body_classifier.detectMultiScale(grey,1.2,3)
    
    # Extract bounding boxes for any bodies identified
    for (x,y,w,h) in bodies:
        cv2.rectangle(cap,(x,y),(x+w,y+h),(128,42,240),2)
        image=cap[y:y+h,x:x+w]
        cv2.imwrite("PEDESTRIAN.jpg",image)


    if cv2.waitKey(1) == 32: #32 is the Space Key
        break

cv2.imshow("rectangle",cap)
cv2.waitKey(0)
cap.release()
cv2.destroyAllWindows()
