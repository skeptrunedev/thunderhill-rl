"""Install the pinned NVIDIA workspace without a separate training dependency stack."""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from training.native_source import verify_source

ALPAGYM_REVISION = "972d160eed0e23d388497851504a3a233fec5879"
ALPAGYM_URL = "https://github.com/NVlabs/alpagym.git"


def checkout_source(destination: Path) -> None:
    if not destination.exists():
        destination.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--no-checkout", ALPAGYM_URL, str(destination)], check=True
        )
        subprocess.run(
            ["git", "-C", str(destination), "checkout", "--detach", ALPAGYM_REVISION],
            check=True,
        )
    actual = subprocess.check_output(
        ["git", "-C", str(destination), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual != ALPAGYM_REVISION:
        raise RuntimeError(
            "Expected the pinned NVIDIA checkout; refusing to change its revision"
        )
    verify_source(destination, apply_patch=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Verify sources without installing the GPU runtime",
    )
    args = parser.parse_args()
    checkout = args.checkout.resolve()
    checkout_source(checkout)
    if not args.source_only:
        for executable in ("uv", "redis-server"):
            if shutil.which(executable) is None:
                raise FileNotFoundError(
                    f"Install required NVIDIA runtime tool: {executable}"
                )
        subprocess.run(
            ["uv", "sync", "--frozen", "--all-packages", "--project", str(checkout)],
            check=True,
        )
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("check_native_dependencies.py")), "--checkout", str(checkout)],
            check=True,
        )
    print(f"NVIDIA source: {checkout} at {ALPAGYM_REVISION}")
    if args.source_only:
        print("Source verified only; GPU runtime installation was not requested")
    else:
        print(f"Runtime Python: {checkout / '.venv/bin/python'}")
        print(
            "Dependencies checked; CUDA execution is validated by the training launcher"
        )


if __name__ == "__main__":
    main()
