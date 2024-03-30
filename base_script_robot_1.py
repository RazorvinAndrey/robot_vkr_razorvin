from zeroconf import ServiceBrowser, ServiceInfo, Zeroconf
import socket
import threading
import json
import time


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
        # Этот метод вызывается, когда информация о сервисе обновляется.
        # Если вам не нужно обрабатывать обновления, просто оставьте метод пустым.
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
            pass  # Ignore if the robot is currently unavailable

    def send_updates(self):
        while True:
            for robot_name, (ip, port) in self.known_robots.items():
                message = json.dumps({"name": self.robot_name, "location": "x:100, y:200"})  # Customize this message
                self.send_message(ip, port, message)
            time.sleep(1)

    def stop(self):
        self.zeroconf.unregister_service(self.info)
        self.zeroconf.close()


if __name__ == "__main__":
    robot_name = "Robot2"  # Change this for each robot
    robot = RobotP2P(robot_name)
    robot.start()

    try:
        while True:
            time.sleep(0.1)
    except KeyboardInterrupt:
        robot.stop()
