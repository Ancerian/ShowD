import socket


def scan_port(host_port, port=8889):
    print(f"Scanning ports on {host_port}...")

    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(0.1)  # Set a connection timeout

    result = sock.connect_ex((host_port, port))

    if result == 0:
        print(f"Port {port} is open")
        return True
    
    sock.close()
    return False


def scan_ips(prefix='192.168.35.'):
    open_ips = []
    for i in range(1, 255):
        ip = f"{prefix}{i}"
        if scan_port(ip):
            open_ips.append(ip)

    return open_ips


print(scan_ips())

