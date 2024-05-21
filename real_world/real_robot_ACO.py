import numpy as np
import time
import json
import threading
from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
import socket
from i2c_itg3205 import *
import RPi.GPIO as GPIO
from time import sleep

# Конфигурация для удаленного запуска
SERVER_IP = "192.168.1.100"  # IP адрес сервера
SERVER_PORT = 12345  # Порт сервера


def listen_for_start():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect((SERVER_IP, SERVER_PORT))
        print("Connected to the server. Waiting for the start command...")
        start_command = s.recv(1024).decode('utf-8')
        init_positions = {"Robot1": (0, 1), "Robot2": (2, 0), "Robot3": (1, 1), "Robot4": (0, 0)}
        target_positions = {"Robot1": (8, 7), "Robot2": (7, 7), "Robot3": (8, 8), "Robot4": (8.5, 8)}

        robot_name = "Robot2"  # Изменить имя робота
        robot = ACORobot(robot_name, init_positions[robot_name], target_positions[robot_name])
        if start_command == "START":
            print("Received start command. Starting the robot...")
            start_time = time.time()
            try:
                robot.move_to_target()
            except KeyboardInterrupt:
                robot.stop()
            end_time = time.time()
            duration = end_time - start_time
            s.sendall(f"FINISHED {robot.name} {duration:.2f}".encode('utf-8'))
            print(f"Sent completion message to the server: {duration:.2f} seconds")


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


def MotorStop():
    pwmA.ChangeDutyCycle(0)
    pwmB.ChangeDutyCycle(0)
    GPIO.output(IN1, False)
    GPIO.output(IN2, False)
    GPIO.output(IN3, False)
    GPIO.output(IN4, False)


def MotorOn(speed):
    pwmA.ChangeDutyCycle(speed)
    pwmB.ChangeDutyCycle(speed)
    GPIO.output(IN1, False)
    GPIO.output(IN2, True)
    GPIO.output(IN3, False)
    GPIO.output(IN4, True)


def turn_by_angle(angle_g):
    itg3205 = i2c_itg3205(1)
    angle = angle_g * math.pi / 180
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
                message = json.dumps(
                    {"name": self.robot_name, "position": (self.x, self.y)})  # Update with real position
                self.send_message(ip, port, message)
            time.sleep(1)

    def stop(self):
        self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()


class PheromoneMap:
    def __init__(self, width, height, decay_rate=0.1):
        self.map = np.zeros((height, width))
        self.decay_rate = decay_rate

    def update(self):
        self.map *= (1 - self.decay_rate)

    def add_pheromone(self, x, y, amount=1.0):
        if 0 <= int(x) < self.map.shape[1] and 0 <= int(y) < self.map.shape[0]:
            self.map[int(y), int(x)] += amount

    def get_pheromone_level(self, x, y):
        if 0 <= int(x) < self.map.shape[1] and 0 <= int(y) < self.map.shape[0]:
            return self.map[int(y), int(x)]
        return 0


class ACORobot:
    def __init__(self, name, init_pos, target_pos, pheromone_map):
        self.name = name
        self.position = np.array(init_pos, dtype=float)
        self.target = np.array(target_pos, dtype=float)
        self.pheromone_map = pheromone_map
        self.velocity = np.random.rand(2) * 0.1
        self.angle = np.arctan2(self.velocity[1], self.velocity[0])
        self.max_speed = 0.5
        self.width = 0.33
        self.path = [tuple(self.position)]
        self.at_target = False
        self.p2p = RobotP2P(self.name)
        self.p2p.start()

    def move_forward(self, duration):
        MotorOn(self.max_speed * 100)
        sleep(duration)
        MotorStop()

    def turn_to(self, target_angle):
        current_angle = self.angle
        angle_diff = (target_angle - current_angle) % (2 * np.pi)
        if angle_diff > np.pi:
            angle_diff -= 2 * np.pi
        turn_by_angle(np.degrees(angle_diff))
        self.angle = target_angle

    def update_velocity(self):
        if not self.at_target:
            pheromone_levels = [self.pheromone_map.get_pheromone_level(self.position[0] + dx, self.position[1] + dy)
                                for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]]
            best_direction = np.argmax(pheromone_levels)
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
            self.velocity = np.array(directions[best_direction]) * self.max_speed
            self.update_angle()

    def update_angle(self):
        if np.linalg.norm(self.velocity) > 0:
            target_angle = np.arctan2(self.velocity[1], self.velocity[0])
            angle_diff = (target_angle - self.angle + np.pi) % (2 * np.pi) - np.pi
            angle_diff = np.clip(angle_diff, -np.pi / 2, np.pi / 2)
            self.angle += angle_diff

    def move(self):
        if not self.at_target:
            target_angle = np.arctan2(self.velocity[1], self.velocity[0])
            self.turn_to(target_angle)
            distance = np.linalg.norm(self.velocity)
            self.move_forward(distance / self.max_speed)
            self.position += self.velocity
            self.path.append(tuple(self.position))
            self.pheromone_map.add_pheromone(self.position[0], self.position[1], 1.0)
            self.check_if_at_target()
            self.send_position_update()

    def check_if_at_target(self):
        target_radius = 0.3
        if np.linalg.norm(self.position - self.target) < target_radius:
            self.velocity = np.zeros(2)
            self.at_target = True

    def send_position_update(self):
        message = json.dumps({"name": self.name, "position": self.position.tolist()})
        for robot_name, (ip, port) in self.p2p.known_robots.items():
            self.p2p.send_message(ip, port, message)

    def move_to_target(self):
        try:
            while not self.at_target:
                self.update_velocity()
                self.move()
                time.sleep(0.1)
        except KeyboardInterrupt:
            self.stop()

    def stop(self):
        self.p2p.stop()
        MotorStop()


# Запуск прослушивания команды старта в отдельном потоке
threading.Thread(target=listen_for_start).start()
