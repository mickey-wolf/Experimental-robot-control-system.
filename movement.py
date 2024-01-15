from pyfirmata import ArduinoMega, util
import time
board = ArduinoMega('COM6')
it = util.Iterator(board)
it.start()
leftForward = board.get_pin('d:3:o')
leftBackward = board.get_pin('d:2:o')
rightForward = board.get_pin('d:5:o')
rightBackward = board.get_pin('d:4:o')
def move(forwardgain,rotationgain): #Only temporary, doesnt enable continous movement.
    if rotationgain > 0:
        leftForward.write(1)
        leftBackward.write(0)
        rightForward.write(0)
        rightBackward.write(1)
        print('moving right')
    elif rotationgain < 0:
        leftForward.write(0)
        leftBackward.write(1)
        rightForward.write(1)
        rightBackward.write(0)
        print('moving left')
    elif forwardgain > 0:
        leftForward.write(1)
        leftBackward.write(0)
        rightForward.write(1)
        rightBackward.write(0)
        print('moving forward')
    else:
        leftBackward.write(0)
        leftForward.write(0)
        rightForward.write(0)
        rightBackward.write(0)
        print("not moving")
