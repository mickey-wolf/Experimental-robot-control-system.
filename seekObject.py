
from ultralytics import YOLO
import cv2
import math
import nltk
nltk.download('wordnet')
import time
from nltk.corpus import wordnet
def seekObject(objectName):
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    synonyms = []
    for syn in wordnet.synsets(objectName):
        for i in syn.lemmas():
            synonyms.append(i.name())

    syns = synonyms
    syns = [f"{objectName}"] if len(syns) == 0 else syns
    try:
        for syn in syns:
            if (syn in classNames) or (objectName in classNames):
                object = syn if syn in classNames else objectName
                cap = cv2.VideoCapture(0)
                cap.set(3, 640)
                cap.set(4, 480)

                model = YOLO("yolo-Weights/yolov8n.pt")

                timeout_start = time.time()
                object_last_seen_x = -1
                while time.time() < timeout_start + 10:
                    success, img = cap.read()
                    results = model(img, stream=True)
                    for r in results:
                        boxes = r.boxes
                        for box in boxes:
                            # bounding box
                            x1, y1, x2, y2 = box.xyxy[0]
                            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                            # put box in cam
                            cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                            # class name
                            cls = int(box.cls[0])
                            if object == classNames[cls]:
                                timeout_start = time.time()
                                print(f"Found {object}!" )
                                object_last_seen_x = x1+(abs(x2-x1)/2)
                                print(object_last_seen_x)
                                move([[0,20,""]])
                                if abs(x2-x1)*abs(y2-y1) > 200000:
                                    print("Object Reached!")
                                    return(True)

                            else:
                                if object_last_seen_x > 320:
                                    move([[30, -1, ""]])
                                    print("Rotating Right")
                                elif object_last_seen_x < 320 and object_last_seen_x>0:
                                    move([[-30, -1, ""]])
                                    print("Rotating Left")

                            # object details
                            org = [x1, y1]
                            font = cv2.FONT_HERSHEY_SIMPLEX
                            fontScale = 1
                            color = (255, 0, 0)
                            thickness = 2

                            cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)

                    cv2.imshow('Webcam', img)
                    if cv2.waitKey(1) == ord('q'):
                        break

                cap.release()
                cv2.destroyAllWindows()
            else:
                raise Exception("Object Unknown To System.")

            break
    except Exception as e:
        print(e)
        return False
def move(movement):
    pass
seekObject("orange")
