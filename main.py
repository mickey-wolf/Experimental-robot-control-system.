import torch
from playsound import playsound
import speech_recognition as sr
from gpt4all import GPT4All
import pyttsx3
import time
engine = pyttsx3.init()
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
import string
from ultralytics import YOLO
import cv2
import math
import nltk
nltk.download('wordnet')
import time
from nltk.corpus import wordnet

def getSpeech():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)

    detectedSpeech = ((r.recognize_google(audio, language="english")).lower()).translate(str.maketrans('', '', string.punctuation))
    print("Recognized:" + detectedSpeech)
    return (detectedSpeech)



def processVoiceMovementInstruction(instruct):  ## this model is shitty, gotta replace it.
    with model.chat_session(
            system_prompt="If an instruction requires you to move somewhere, you must only return the vector (DIRECTION, DISTANCE, "
                          "TGT) where DIRECTION is an integer that represents the angle in degrees in which you are "
                          "required to move. It can be between -180 to 180. DISTANCE is the absolute value of the "
                          "distance converted to centimeters that you are required to move. TGT should be "
                          "an empty string unless you are required to proceed towards an object. In that "
                          "case, you must write the name of the object in this field. If TGT represents a person, and not an object, replace the TGT string with the string 'person'"
                          ". The default DISTANCE is 0, And the default DIRECTION is 0."
                          " You are not allowed to say anything but the "
                          "required vector. If multiple operations are required, write a Python list "
                          "which you append with a vector for each operation. If one operation is "
                          "required, return it as [[DIRECTION, DISTANCE, TGT]]."):
        response1 = model.generate(prompt=instruct, temp=0)

        rawMovementCommand = (model.current_chat_session[2]["content"])
        print("Model returned: " + rawMovementCommand)
        return (rawMovementCommand)


def processRawMovementInstruction(rawMovementCommand):
    movementCommand = eval(rawMovementCommand)
    try:
        for movement in movementCommand:
            if type(int(movement[0])) is int and type(int(movement[1])) is int and type(movement[2]) is str:
                pass
            else:
                raise TypeError
        print("Movement Command Recieved is: " + str(movementCommand))
        return (movementCommand, True)
    except TypeError:
        print("rawMovementCommand Invalid")
        return ([[[0, -1, ""]]], False)


def speak(text):
    engine.say(text)
    engine.runAndWait()

def seekObject(objectName):
    classNames = ["person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck", "boat",
                  "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat",
                  "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
                  "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
                  "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup",
                  "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli",
                  "carrot", "hot dog", "pizza", "cake", "chair", "sofa", "pottedplant", "bed",
                  "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone",
                  "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
                  "teddy bear", "hair drier", "toothbrush"
                  ]

    syns = []
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
        cap = cv2.VideoCapture(0)
        resolution = (640, 480)
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
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        object_box_area = (abs(x2 - x1) * abs(y2 - y1))
                        cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 255), 3)
                        object_last_seen_x = x1+(abs(x2-x1)/2)
                        # org = [x1, y1]
                        # font = cv2.FONT_HERSHEY_SIMPLEX
                        # fontScale = 1
                        # color = (255, 0, 0)
                        # thickness = 2
                        # cv2.putText(img, classNames[cls], org, font, fontScale, color, thickness)
                        if object_box_area > resolution[1]*resolution[0]*desired_pixel_occupancy:
                            object_reached = True
                            print("Object Reached!")
                            cap.release()
                            # cv2.destroyAllWindows()
                            return True

                        elif object_last_seen_x < center_zone_for_object[1] and object_last_seen_x > center_zone_for_object[0]:
                            forward_gain = (total_area*desired_pixel_occupancy-object_box_area)/total_area*desired_pixel_occupancy
                            rotation_gain = 0

                        else:
                            rotation_gain = -(resolution[0]/2-object_last_seen_x)/(resolution[0]/2)
                            forward_gain = (
                                                       total_area * desired_pixel_occupancy - object_box_area) / total_area * desired_pixel_occupancy
                        break


            if object_in_frame == False:
                if object_last_seen_x is None:
                    rotation_gain = 1
                    print(f"Looking for {object}")
            # cv2.imshow('Webcam', img)
            # if cv2.waitKey(1) == ord('q'):
            #     break
            print(f"Forward Gain is {forward_gain}\n"
                  f"Rotation Gain is {rotation_gain}")

        if object_last_seen_x == None:
            raise Exception("Object Not Found")
        cap.release()
        cv2.destroyAllWindows()
    except Exception as e:
        print(e)
        return False

def move(gain_vector):
    pass
def movementToGain(movement):
    pass

#####CODE STARTS HERE!!
playsound("startup new.wav")
while True:
    try:
        while True:
            playsound("ready.wav")
            wake_up_call = getSpeech()
            if wake_up_call == " wake up":
                print('Woke Up!')
                playsound('Epiano Wake Up heard.wav')
                voiceCommand = getSpeech()
                # answer = verbalResponseGenerator Need to define a function that responds to the question, with gpt.
                #speak()
                rawMovementInstruction = processVoiceMovementInstruction(voiceCommand)
                movementInstruction = processRawMovementInstruction(rawMovementInstruction)
                if movementInstruction[1]:
                    for movement in movementInstruction[0]:
                        objectName = movement[2]
                        if objectName != "":
                            print(objectName)
                            if seekObject(objectName):
                                playsound("task complete.wav")
                            else:
                                playsound("task failed.wav")
                        else:
                            move(movementToGain(movement))



    except Exception as e:
        playsound("Epiano Error.wav")
        print(e)
        time.sleep(5)
        pass
