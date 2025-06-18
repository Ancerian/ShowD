import socket
import threading

import time


def search_tello():
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    sock.bind(('', 8889))

    tello_ip_list = []
    wait_for_tello = True


    def scan_port(host_port, port=8889):
        print(f"Scanning ports on {host_port}...")

        sock.sendto('command'.encode('utf-8'), (host_port, port))

        return False


    def scan_ips(prefix='192.168.35.'):
        open_ips = []
        for i in range(1, 255):
            ip = f"{prefix}{i}"
            if scan_port(ip):
                open_ips.append(ip)

        return open_ips



    def _receive_thread():
        """
        Listen to responses from the Tello.
        Runs as a thread, sets self.response to whatever the Tello last returned.

        :return: None.
        """

        while wait_for_tello:
            try:
                print(sock)
                response, ip = sock.recvfrom(1024)
                response = response.decode('utf-8')
                
                ip = ''.join(str(ip[0]))                
                
                if response.upper() == 'OK' and ip not in tello_ip_list:
                    tello_ip_list.append(ip)

            except socket.error as exc:
                # swallow exception
                print(f"[Exception_Error]Caught exception socket.error : {exc}")
                pass


    print(scan_ips())
    receive_thread = threading.Thread(target=_receive_thread)
    receive_thread.daemon = True
    receive_thread.start()

    time.sleep(3)
    wait_for_tello = False
    sock.close()
    return tello_ip_list


if __name__ == "__main__":
    print(search_tello())

