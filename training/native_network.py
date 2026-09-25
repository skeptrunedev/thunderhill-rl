"""Configure the container's own hostname for NVIDIA's native TCPStore."""

import fcntl
import ipaddress
import socket
import struct
from pathlib import Path


def configure_container_hostname() -> dict:
    """Advertise the actual default-route interface, without changing transport."""
    routes = []
    for line in Path("/proc/net/route").read_text().splitlines()[1:]:
        fields = line.split()
        if fields[1] == "00000000" and int(fields[3], 16) & 1:
            routes.append((int(fields[6]), fields[0]))
    if not routes:
        raise RuntimeError("No active IPv4 default route in container")
    _, interface = min(routes)
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
        response = fcntl.ioctl(
            probe.fileno(), 0x8915, struct.pack("256s", interface.encode()[:15])
        )  # Linux SIOCGIFADDR, address assigned to the inspected interface.
    address = socket.inet_ntoa(response[20:24])
    parsed = ipaddress.ip_address(address)
    if parsed.is_loopback or parsed.is_unspecified:
        raise RuntimeError(f"Default-route interface has unusable address: {address}")
    hostname = socket.gethostname()
    hosts = Path("/etc/hosts")
    original = hosts.read_text()
    lines = []
    for line in original.splitlines():
        entry, separator, comment = line.partition("#")
        fields = entry.split()
        if len(fields) > 1 and hostname in fields[1:]:
            aliases = [alias for alias in fields[1:] if alias != hostname]
            if aliases:
                lines.append("\t".join([fields[0], *aliases]) + (" #" + comment if separator else ""))
            elif separator:
                lines.append("#" + comment)
        else:
            lines.append(line)
    lines.append(f"{address}\t{hostname}")
    hosts.write_text("\n".join(lines) + "\n")
    resolved = sorted({
        item[4][0] for item in socket.getaddrinfo(hostname, None, proto=socket.IPPROTO_TCP)
    })
    if resolved != [address]:
        raise RuntimeError(f"Container hostname resolution mismatch: {resolved}, expected {address}")
    with socket.socket() as server:
        server.bind(("0.0.0.0", 0))
        server.listen(1)
        with socket.create_connection((hostname, server.getsockname()[1]), timeout=5):
            connection, _ = server.accept()
            connection.close()
    return {
        "hostname": hostname, "interface": interface, "address": address,
        "resolved_addresses": resolved, "hostname_tcp_connection_verified": True,
        "hosts_before": original, "hosts_after": hosts.read_text(),
    }
