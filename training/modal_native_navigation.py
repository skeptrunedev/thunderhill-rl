"""CPU only native navigation verification on a real captured game observation."""

from pathlib import Path

import modal

from training.modal_native import REMOTE, ROOT, UPSTREAM, cache, image

app = modal.App("thunderhill-native-navigation-verification")
runtime_image = image
for relative in (
    "training/native_source.py",
    "tools/setup_alpagym.py",
    "tools/check_native_navigation.py",
    "tools/patches/alpagym-native-navigation.patch",
    "tools/patches/alpagym-native-navigation.json",
):
    runtime_image = runtime_image.add_local_file(
        str(ROOT / relative), REMOTE + "/" + relative
    )


@app.function(
    image=runtime_image,
    cpu=4,
    memory=16384,
    timeout=600,
    retries=0,
    volumes={"/model-cache": cache},
    include_source=False,
)
def verify_navigation(captured_payload: bytes):
    import json
    import os
    import subprocess
    import tempfile

    os.chdir(REMOTE)
    subprocess.run(
        [
            UPSTREAM + "/.venv/bin/python",
            "tools/setup_alpagym.py",
            "--checkout",
            UPSTREAM,
            "--source-only",
        ],
        check=True,
    )
    with tempfile.TemporaryDirectory(prefix="native-navigation-") as directory:
        source = Path(directory) / "native_model_input.pt"
        result = Path(directory) / "navigation_report.json"
        source.write_bytes(captured_payload)
        subprocess.run(
            [
                UPSTREAM + "/.venv/bin/python",
                "tools/check_native_navigation.py",
                "--model-input",
                str(source),
                "--checkpoint",
                "/model-cache/alpagym-converted-1.5",
                "--output",
                str(result),
                "--release-config",
                "/model-cache/hub/models--nvidia--Alpamayo-1.5-10B/snapshots/7aba8293c09993f2e125c6819df05d7fa3e873ea/config.json",
            ],
            check=True,
            timeout=540,
        )
        cache.commit()
        return json.loads(result.read_text())


@app.local_entrypoint()
def main(model_input: str, output: str):
    import json

    result = verify_navigation.remote(Path(model_input).read_bytes())
    result["real_game_model_input"] = str(Path(model_input).resolve())
    destination = Path(output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
