from pyfirmata import Arduino, util
import time

board= Arduino('COM3')
it = util.Iterator(board)
it.start()

pin_10 = board.get_pin('d:10:o')
pin_9 = board.get_pin('d:9:o')
pin_10.write(1)