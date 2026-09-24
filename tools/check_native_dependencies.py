"""Check installed NVIDIA dependencies with its one documented exclusion.

AlpaGym intentionally excludes xformers although vLLM metadata requires it.
Preserve that upstream choice and reject every other pip check discrepancy.
uv sync --check requests reinstalling the same pinned FlashAttention URL directly
after a successful sync. Check the installed version and wheel origin explicitly.
"""

import argparse
import json
from pathlib import Path
import subprocess
import tomllib
from urllib.parse import unquote

REVISION = "972d160eed0e23d388497851504a3a233fec5879"
EXCLUDED_REQUIREMENT = (
    "The package `vllm` requires `xformers==0.0.32.post1 ; "
    "platform_machine == 'x86_64' and sys_platform == 'linux'`, but it's not installed"
)


def check(checkout: Path) -> None:
    revision = subprocess.check_output(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"], text=True
    ).strip()
    if revision != REVISION:
        raise ValueError("Dependency exception applies only to the reviewed NVIDIA revision")
    configuration = tomllib.loads((checkout / "pyproject.toml").read_text())
    uv = configuration["tool"]["uv"]
    if "xformers ; sys_platform == 'never'" not in uv["override-dependencies"]:
        raise ValueError("NVIDIA no longer explicitly excludes xformers")
    python = str(checkout / ".venv/bin/python")
    result = subprocess.run(
        ["uv", "pip", "check", "--python", python, "--color", "never"],
        text=True, capture_output=True,
    )
    report = result.stdout + result.stderr
    if result.returncode:
        discrepancies = [line for line in report.splitlines() if line.startswith("The package ")]
        if result.returncode != 1 or discrepancies != [EXCLUDED_REQUIREMENT] or "Found 1 incompatibility" not in report:
            raise RuntimeError(report)
        print("Accepted NVIDIA's documented xformers exclusion; no other dependency discrepancies")
    else:
        print(report, end="")
    installed = json.loads(subprocess.check_output(
        [python, "-c", (
            "import importlib.metadata as m,json; d=m.distribution('flash-attn'); "
            "print(json.dumps({'version':d.version,'source':json.loads(d.read_text('direct_url.json'))}))"
        )], text=True,
    ))
    if installed["version"] != "2.8.3":
        raise ValueError("Installed FlashAttention version differs from NVIDIA's pin")
    if unquote(installed["source"]["url"]) != unquote(uv["sources"]["flash-attn"]["url"]):
        raise ValueError("Installed FlashAttention wheel differs from NVIDIA's ABI-specific wheel")
    print("Verified installed FlashAttention version and exact NVIDIA wheel source")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=Path, required=True)
    check(parser.parse_args().checkout.resolve())


if __name__ == "__main__":
    main()
