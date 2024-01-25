import cv2

video = cv2.VideoCapture('static/bb2.mp4')
while True:
    check,image = video.read()
    print(check)
    if(check):
        cv2.imshow("myvideo", image)
        if cv2.waitKey(1) == 27:
           break
    

video.release()