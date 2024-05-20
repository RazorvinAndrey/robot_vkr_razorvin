import RPi.GPIO as GPIO
from i2c_itg3205 import *
from time import sleep
import atexit
import math
import socket
import threading
import json
import time
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf

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

class RobotP2P:
    def __init__(self, robot_name, service_type="_robot._tcp.local.", port=12345):
        self.robot_name = robot_name
        self.service_type = service_type
        self.port = port
        self.zeroconf = Zeroconf()
        self.address = self._get_local_ip()
        self.info = ServiceInfo(
            service_type,
            f"{robot_name}.{service_type}",
            addresses=[socket.inet_aton(self.address)],
            port=port,
            properties={"name": robot_name.encode("utf-8")},
        )
        self.known_robots = {}

    def _get_local_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except Exception:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP

    def update_service(self, zeroconf, type, name, info):
        pass

    def start(self):
        self.zeroconf.register_service(self.info)
        self.browser = ServiceBrowser(self.zeroconf, self.service_type, listener=self)
        threading.Thread(target=self.listen_for_messages).start()
        threading.Thread(target=self.send_updates).start()

    def add_service(self, zeroconf, type, name):
        info = zeroconf.get_service_info(type, name)
        if info and info.properties[b'name'] != self.robot_name.encode("utf-8"):
            robot_name = info.properties[b'name'].decode("utf-8")
            self.known_robots[robot_name] = (socket.inet_ntoa(info.addresses[0]), info.port)
            print(f"Discovered robot: {robot_name} at {self.known_robots[robot_name]}")

    def remove_service(self, zeroconf, type, name):
        robot_name = name.split('.')[0]
        if robot_name in self.known_robots:
            del self.known_robots[robot_name]
            print(f"Robot {robot_name} left the network")

    def listen_for_messages(self):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(('', self.port))
            s.listen()
            while True:
                conn, addr = s.accept()
                with conn:
                    while True:
                        data = conn.recv(1024)
                        if not data:
                            break
                        message = data.decode('utf-8')
                        print(f"Message from {addr}: {message}")

    def send_message(self, ip, port, message):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
                sock.connect((ip, port))
                sock.sendall(bytes(message, 'utf-8'))
        except ConnectionRefusedError:
            pass

    def send_updates(self):
        while True:
            for robot_name, (ip, port) in self.known_robots.items():
                message = json.dumps({"name": self.robot_name, "location": "x:100, y:200"})  # Customize this message
                self.send_message(ip, port, message)
            time.sleep(1)

    def stop(self):
        self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()


# ------------------------- Robot Movement Control ---------------------------
class RealRobot:
    def __init__(self, name, init_pos, target_pos):
        self.name = name
        self.x, self.y = init_pos
        self.target_x, self.target_y = target_pos
        self.angle = 0  # Начальный угол движения
        self.speed = 50  # Скорость движения

        # Инициализация P2P связи
        self.p2p = RobotP2P(self.name)
        self.p2p.start()

    def move_forward(self, duration):
        MotorOn(self.speed)
        sleep(duration)
        MotorStop()

    def turn_to(self, target_angle):
        current_angle = self.angle
        angle_diff = (target_angle - current_angle) % 360
        if angle_diff > 180:
            angle_diff -= 360
        turn_by_angle(angle_diff)
        self.angle = target_angle

    def move_to_target(self):
        while not self.is_at_target():
            target_angle = self.calculate_target_angle()
            self.turn_to(target_angle)
            distance = self.calculate_distance_to_target()
            self.move_forward(distance / self.speed)
            self.update_position()
            self.send_position_update()

    def calculate_target_angle(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return math.degrees(math.atan2(dy, dx)) % 360

    def calculate_distance_to_target(self):
        dx = self.target_x - self.x
        dy = self.target_y - self.y
        return math.sqrt(dx ** 2 + dy ** 2)

    def is_at_target(self):
        return self.calculate_distance_to_target() < 0.1

    def update_position(self):
        self.x = self.target_x
        self.y = self.target_y

    def send_position_update(self):
        message = json.dumps({"name": self.name, "position": (self.x, self.y)})
        for robot_name, (ip, port) in self.p2p.known_robots.items():
            self.p2p.send_message(ip, port, message)

    def stop(self):
        self.p2p.stop()


if __name__ == "__main__":
    init_positions = {"Robot1": (0, 1), "Robot2": (2, 0), "Robot3": (1, 1), "Robot4": (0, 0)}
    target_positions = {"Robot1": (8, 7), "Robot2": (7, 7), "Robot3": (8, 8), "Robot4": (8.5, 8)}

    robot_name = "Robot2"  # Change this for each robot
    robot = RealRobot(robot_name, init_positions[robot_name], target_positions[robot_name])
    try:
        robot.move_to_target()
    except KeyboardInterrupt:
        robot.stop()