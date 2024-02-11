import torch
from playsound import playsound
import speech_recognition as sr
from gpt4all import GPT4All
import pyttsx3
import time
engine = pyttsx3.init()
model = GPT4All("mistral-7b-instruct-v0.1.Q4_0.gguf")
import string
import time
import seekObject

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



def movementToGain(movement):
    pass

#####CODE STARTS HERE!
playsound("startup new.wav")
while True:
    try:
        while True:
            playsound("ready.wav")
            wake_up_call = getSpeech()
            if wake_up_call == "wake up":
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
                            # move(movementToGain(movement))
                            pass



    except Exception as e:
        playsound("Epiano Error.wav")
        print(e)
        time.sleep(5)
        pass
