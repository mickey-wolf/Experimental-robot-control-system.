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

    syns = {}
    for syn in wordnet.synsets(objectName):
        for i in syn.lemmas():
            syns.append(i.name())
    syns = set(syns)
    syns = {objectName} if len(syns) == 0 else syns
    objectKnown = False
    print(syns)
    try:
        for syn in syns:
            if (syn in classNames) or (objectName in classNames):
                object = syn if syn in classNames else objectName
                objectKnown = True
        if objectKnown == False:
            raise Exception("Object Unknown To System.")
        cap = cv2.VideoCapture(0)
        resolution = (224,224)
        total_area = resolution[0]*resolution[1]
        center_zone_for_object = (0.25*resolution[0],0.75*resolution[0])
        desired_pixel_occupancy = 2/3
        cap.set(3, resolution[0])
        cap.set(4, resolution[1])

        model = YOLO("yolo-Weights/yolov3-tinyu.pt")

        timeout_start = time.time()
        object_last_seen_x = None
        object_reached = False
        forward_gain = 0
        rotation_gain = 0

        started = time.time()
        last_logged = time.time()
        frame_count = 0

        while time.time() < timeout_start + 10:
            object_in_frame = False
            success, img = cap.read()
            results = model(img, stream=True)
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    if object == classNames[cls]:
                        object_in_frame = True
                        timeout_start = time.time()
                        print(f"Found {object}!" )
                        x1, y1, x2, y2 = box.xyxy[0]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values
                        object_box_area = (abs(x2 - x1) * abs(y2 - y1))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        object_last_seen_x = x1+(abs(x2-x1)/2)
                        org = [x1, y1]
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        fontScale = 1
                        color = (255, 0, 0)
                        thickness = 2
                        cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        if object_box_area > resolution[1]*resolution[0]*desired_pixel_occupancy:
                            object_reached = True
                            print("Object Reached!")
                            return True

                        elif object_last_seen_x < center_zone_for_object[1] and object_last_seen_x > center_zone_for_object[0]:
                            forward_gain = (total_area*desired_pixel_occupancy-object_box_area)/total_area*desired_pixel_occupancy
                            print(f"Object ahead! \n"
                                  f"Moving forward gain is: {forward_gain}")
                        else:
                            rotation_gain = -(resolution[0]/2-object_last_seen_x)/(resolution[0]/2)
                            print(f"Rotating towards object.\n"
                                  f"Rotation gain is: {rotation_gain}")
                        break
                    # object details

            if object_in_frame == False:
                if object_last_seen_x is not None and object_last_seen_x > resolution[0]/2:
                    forward_gain = 0
                    rotation_gain = 1
                    print("Searching for object: Rotating Right")
                elif object_last_seen_x is not None and object_last_seen_x < resolution[0]/2 and object_last_seen_x > 0:
                    moving_forward = 0
                    rotation_gain = -1
                    print("Searching for object : Rotating Left")
                elif object_last_seen_x is None:
                    rotation_gain = 1
            cv2.imshow('Webcam', img)
            if cv2.waitKey(1) == ord('q'):
                break
            frame_count += 1
            now = time.time()
            if now - last_logged > 1:
                print(f"{frame_count / (now - last_logged)} fps")
                last_logged = now
                frame_count = 0

        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        return False