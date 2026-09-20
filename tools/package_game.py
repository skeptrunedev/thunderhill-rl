# /// script
# requires-python = ">=3.11"
# dependencies = []
# ///
"""Export a provenance tagged native build with installed Godot templates.

Requires Godot 4.7.2 and matching official export templates. Development builds
may explicitly allow a dirty tree, and retain its exact content digest.
"""
import argparse
import hashlib
import json
from pathlib import Path
import plistlib
import subprocess
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def validate_scenery_bake():
    project = ROOT / "godot"
    manifest = json.loads((project / "data/scenery-bake.json").read_text())
    if manifest.get("schema_version") != 1 or not manifest.get("sources"):
        raise RuntimeError("Invalid scenery bake manifest")
    expected = dict(manifest["sources"])
    expected["assets/generated/scenery.scn"] = manifest["scene_sha256"]
    for relative, digest in expected.items():
        path = project / relative
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise RuntimeError(f"Stale scenery bake: {relative}. Run Godot --path godot --script res://tools/bake_scenery.gd")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--godot", required=True)
    parser.add_argument("--platform", choices=["macos", "linux"], default="macos")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args()
    validate_scenery_bake()
    dirty = bool(git("status", "--porcelain"))
    if dirty and not args.allow_dirty:
        parser.error("Commit the verified source first, or explicitly use --allow-dirty for a development package")
    version = subprocess.check_output([args.godot, "--version"], text=True).strip()
    if not version.startswith("4.7.2."):
        parser.error(f"Expected Godot 4.7.2, received {version}")
    commit = git("rev-parse", "HEAD")
    files = {}
    for path in sorted((ROOT / "godot").rglob("*")):
        if not path.is_file() or any(part in [".godot", "builds"] for part in path.parts):
            continue
        if path.name == "build-info.json":
            continue
        files[str(path.relative_to(ROOT / "godot"))] = hashlib.sha256(path.read_bytes()).hexdigest()
    content_hash = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    build_id = commit[:12] + "-" + content_hash[:12]
    provenance = {"schema_version": 1, "build_id": build_id, "source_commit": commit,
                  "source_dirty": dirty, "content_sha256": content_hash,
                  "godot_version": version, "platform": args.platform, "files": files,
                  "acceptance": "Development review candidate, not a realism or performance certification"}
    (ROOT / "godot/data/build-info.json").write_text(json.dumps(provenance, indent=2) + "\n")
    output = ROOT / "artifacts/builds" / build_id / args.platform
    output.mkdir(parents=True, exist_ok=True)
    preset, filename = ("macOS", "Thunderhill-macos.zip") if args.platform == "macos" else ("Linux", "Thunderhill.x86_64")
    target = output / filename
    log = output / "export.log"
    with log.open("w") as stream:
        result = subprocess.run([args.godot, "--headless", "--path", str(ROOT / "godot"),
                                 "--export-release", preset, str(target)], stdout=stream,
                                stderr=subprocess.STDOUT, text=True)
    text = log.read_text()
    if result.returncode or "SCRIPT ERROR:" in text or "ERROR:" in text or not target.exists():
        raise RuntimeError(f"Export failed; inspect {log}\n{text[-6000:]}")
    for name, expected in files.items():
        if hashlib.sha256((ROOT / "godot" / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError(f"Source changed during export: {name}")
    if args.platform == "macos":
        with zipfile.ZipFile(target) as archive:
            plist_name = next(n for n in archive.namelist() if n.endswith("Contents/Info.plist"))
            info = plistlib.loads(archive.read(plist_name))
            assert info["CFBundleIdentifier"] == "com.skeptrune.thunderhill"
            executable = plist_name.removesuffix("Info.plist") + "MacOS/" + info["CFBundleExecutable"]
            binary = archive.read(executable)
            assert binary[:4] in [b"\xca\xfe\xba\xbe", b"\xbe\xba\xfe\xca"], "Expected universal Mach-O"
            assert archive.getinfo(executable).external_attr >> 16 & 0o111, "Executable permissions missing"
            assert any(n.endswith(".pck") for n in archive.namelist()), "Game data pack missing"
            provenance["bundle_identifier"] = info["CFBundleIdentifier"]
            provenance["macos_runtime_verified"] = False
    provenance["artifact"] = {"name": filename, "bytes": target.stat().st_size,
                              "sha256": hashlib.sha256(target.read_bytes()).hexdigest()}
    (output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(json.dumps({"build_id": build_id, "artifact": str(target), "manifest": str(output / "manifest.json")}, indent=2))


if __name__ == "__main__":
    main()
