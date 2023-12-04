import torch
from playsound import playsound
import speech_recognition as sr
from gpt4all import GPT4All
import pyttsx3
import time
engine = pyttsx3.init()
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf", device="gpu")
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
    try:
        detectedSpeech = ((r.recognize_whisper(audio, language="english")).lower()).translate(str.maketrans('', '', string.punctuation))
        print("Recognized:" + detectedSpeech)
        return (detectedSpeech)
    except sr.UnknownValueError:
        return -1
    except sr.RequestError as e:
        return -1


def processVoiceMovementInstruction(instruct):  ## this model is shitty, gotta replace it.
    with model.chat_session(
            system_prompt="If an instruction requires you to move somewhere, you must only return the vector (DIRECTION, DISTANCE, "
                          "TGT) where DIRECTION is an integer that represents the angle in degrees in which you are "
                          "required to move. DISTANCE is the absolute value of the "
                          "distance converted to centimeters that you are required to move. TGT should be "
                          "an empty string unless you are required to proceed towards an object. In that "
                          "case, you must write the name of the object in this field. The default distance is -1,And the default direction is -1."
                          " You are not allowed to say anything but the "
                          "required vector. If multiple operations are required, write a Python list "
                          "which you append with a vector for each operation. If one operation is "
                          "required, return it as [[DIRECTION, DISTANCE, TGT]]."):
        response1 = model.generate(prompt=instruct, temp=0)

        rawMovementCommand = (model.current_chat_session[2]["content"])
        print("Model returned: " + rawMovementCommand)
        return (rawMovementCommand)


def processRawMovementInstruction(rawMovementCommand):
    movementCommand = []
    try:
        for movement in eval(rawMovementCommand.rstrip()):
            print(f"Current movement is {movement}")
            if type(int(movement[0])) is int and type(int(movement[1])) is int and type(movement[2]) is str:
                movementCommand.append(movement)
            else:
                raise TypeError
        print("Movement Command Recieved is: "+ movementCommand)
        return (movementCommand, True)
    except TypeError as e:
        print("rawMovementCommand Invalid"+e)
        return ([[[-1, -1, ""]]], False)


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
                    object_in_frame = False
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
                                object_in_frame = True
                                timeout_start = time.time()
                                print(f"Found {object}!" )
                                object_last_seen_x = x1+(abs(x2-x1)/2)
                                move([[0,20,""]])
                                if abs(x2-x1)*abs(y2-y1) > 200000:
                                    print("Object Reached!")
                                    return True

                            else:
                                if object_in_frame ==  False:
                                    if object_last_seen_x > 320:
                                        move([[35, -1, ""]])
                                        print("Rotating Right")
                                    elif object_last_seen_x < 320 and object_last_seen_x>0:
                                        move([[-35, -1, ""]])
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
    return 0


#####CODE STARTS HERE
playsound("Epiano Startup.wav")
while True:
    try:
        while True:
            wake_up_call = getSpeech()
            if wake_up_call == " command":
                try:
                    playsound('Epiano Wake Up heard.wav')
                    voiceCommand = getSpeech()
                    # answer = verbalResponseGenerator Need to define a function that responds to the question, with gpt.
                    #speak()
                    rawMovementInstruction = processVoiceMovementInstruction(voiceCommand)
                    movementInstruction = processRawMovementInstruction(rawMovementInstruction)
                    if movementInstruction[1] is True:
                        for movement in movementInstruction[0]:
                            if movement[2] != "":
                                seekObject(movement[2])
                            else:
                                move((movement))
                except Exception:
                    break

    except Exception as e:
        playsound('Epiano Error.wav')
        print(e)
        time.sleep(5)
        pass
