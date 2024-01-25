import cv2
import numpy as np    # Numerical python to create an array

modelConfiguration="cfg/yolov3.cfg"
modelWeights="yolov3.weights"

yoloNetwork=cv2.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
print(yoloNetwork)

labels= open("coco.names").read().strip().split('\n')


confidenceThreshold=0.5
NMSThreshold=0.3

video = cv2.VideoCapture('static/bb2.mp4')
while True:
    check,image = video.read()
    print(check)
    if(check):
        image=cv2.resize(image, (700,500),fx=1,fy=1)
        dimensions=image.shape[:2]
        print(dimensions)
        H=dimensions[0]
        W=dimensions[1]
        # Blob
        blob = cv2.dnn.blobFromImage(image, 1/255, (416,416))

        # Input the image blob to the model
        yoloNetwork.setInput(blob)

        # Get names of unconnected output layers
        layerName=yoloNetwork.getUnconnectedOutLayersNames()
        # print("layername: ", layerName)

        # Forward the input data through network(neural network)
        layerOutputs=yoloNetwork.forward(layerName)

        boxes=[]
        confidences=[]
        classIds=[]
        for output in layerOutputs:
            #get class score and id of class with the highest score
            for detection in output:
                score=detection[5:]
                classId=np.argmax(score)
                confidence=score[classId]
                if confidence>confidenceThreshold:
                    box=detection[0:4]*np.array([W,H,W,H])
                    (centerX,centerY,width,height)=box.astype('int')
                    x=int(centerX-(width)/2)
                    y=int(centerY-(height)/2)

                    boxes.append([x,y,int(width),int(height)])
                    confidences.append(float(confidence))
                    classIds.append(classId)

        print(len(boxes))
        indexes=cv2.dnn.NMSBoxes(boxes,confidences,confidenceThreshold,NMSThreshold)
        font=cv2.FONT_HERSHEY_SIMPLEX
        cv2.imshow("myvideo", image)
        if cv2.waitKey(1) == 27:
           break
    

video.release()