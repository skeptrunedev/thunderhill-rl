"""Verify deployment hostname configuration on a CPU container before GPUs."""

from pathlib import Path

import modal

app = modal.App("thunderhill-native-network-check")
image = modal.Image.debian_slim(python_version="3.12").add_local_file(
    Path(__file__).with_name("native_network.py"), "/root/native_network.py", copy=True
)


@app.function(image=image, cpu=1, timeout=60, serialized=True, include_source=False)
def check_network():
    from native_network import configure_container_hostname

    return configure_container_hostname()


@app.local_entrypoint()
def main():
    import json

    print(json.dumps(check_network.remote(), indent=2))
