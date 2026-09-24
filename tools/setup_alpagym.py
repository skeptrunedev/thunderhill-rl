"""Install the pinned NVIDIA workspace without a separate training dependency stack."""

import argparse
from pathlib import Path
import subprocess

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
    dirty = subprocess.check_output(
        ["git", "-C", str(destination), "status", "--porcelain"], text=True
    ).strip()
    if actual != ALPAGYM_REVISION or dirty:
        raise RuntimeError(
            "Expected a clean pinned NVIDIA checkout; refusing to change an existing checkout"
        )


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
        subprocess.run(
            ["uv", "sync", "--frozen", "--all-packages", "--project", str(checkout)],
            check=True,
        )
    print(f"NVIDIA source: {checkout} at {ALPAGYM_REVISION}")
    print(f"Runtime Python: {checkout / '.venv/bin/python'}")


if __name__ == "__main__":
    main()
