import cv2

faceObject = cv2.CascadeClassifier('resource/haarcascade_eye.xml')

webcam = cv2.VideoCapture(0)
webcam.set(3,520)
webcam.set(4,600)

while True:
    suc,img = webcam.read()
    grayImg = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    eyes = faceObject.detectMultiScale(grayImg,1.3,4)
    eyeNum = "Number of Eye is: {}".format(len(eyes))
    for (x,y,w,h) in eyes:
        cv2.rectangle(img,(x,y),(x+w,y+h),(0,123,123),2)
        cv2.putText(img,eyeNum,(40,40),cv2.FONT_HERSHEY_COMPLEX,1,(230,32,234),3)
    cv2.imshow('Eyes',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break