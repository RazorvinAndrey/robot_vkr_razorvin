import RPi.GPIO as GPIO
from i2c_itg3205 import *
from time import sleep
import atexit
import math


#  ------------------------------------------Constants----------------------------------

ENA = 13
ENB = 12
IN1 = 26
IN2 = 21
IN3 = 16
IN4 = 19

# Set the type of GPIO
GPIO.setmode(GPIO.BCM)
GPIO.setwarnings(False)
# Motor initialized to LOW
GPIO.setup(ENA, GPIO.OUT)
GPIO.setup(IN1, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN2, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(ENB, GPIO.OUT)
GPIO.setup(IN3, GPIO.OUT, initial=GPIO.LOW)
GPIO.setup(IN4, GPIO.OUT, initial=GPIO.LOW)
pwmA = GPIO.PWM(ENA, 100)
pwmB = GPIO.PWM(ENB, 100)
pwmA.start(0)
pwmB.start(0)


#  ------------------------------------------Close function----------------------------------
def MotorStop():
    print('motor stop')
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)


def MotorOn(speed):
    print('motor forward')
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)


def on_esc():
    print("end")
    MotorStop()


atexit.register(on_esc)


#  ------------------------------------------Base function----------------------------------

def turn_by_angle(angle_g):
    itg3205 = i2c_itg3205(1)
    angle = angle_g*math.pi/180
    dt = 0.2
    spe = 60
    pwmA.ChangeDutyCycle(spe)
    pwmB.ChangeDutyCycle(spe)
    x, y, z = itg3205.getDegPerSecAxes()
    start_error = z
    this_angle = 0
    while abs(angle - this_angle) > 0.3:
        if angle - this_angle > 0:
            GPIO.output(IN4, True)
            GPIO.output(IN3, False)
            GPIO.output(IN1, True)
            GPIO.output(IN2, False)
        else:
            GPIO.output(IN3, True)
            GPIO.output(IN4, False)
            GPIO.output(IN2, True)
            GPIO.output(IN1, False)
        sleep(dt)
        try:
            itgready, dataready = itg3205.getInterruptStatus()
            if dataready:
                x, y, z = itg3205.getDegPerSecAxes()
                this_angle += -(z - start_error) / 180 * math.pi * dt
        except OSError:
            pass
    MotorStop()


#  ------------------------------------------End program----------------------------------
MotorStop()
