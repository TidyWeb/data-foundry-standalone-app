"""Build the downloadable standalone zip (for a GitHub Release).

    python tools/make_standalone_zip.py [output.zip]

The zip holds the app, the generator, the launchers for Windows, macOS and Linux, and START_HERE.txt.
Nothing is deleted or overwritten: it refuses to run if the output file already exists.
"""
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LAUNCHER_FILES = ["start.sh", "Start Data Foundry.command", "Start Data Foundry.bat", "START_HERE.txt", "install-linux-launcher.sh",
                  "icons/data-foundry.svg", "icons/data-foundry.png", "icons/data-foundry.ico"]
LAUNCHER_DIR = ROOT / "standalone_launchers" if (ROOT / "standalone_launchers").is_dir() else ROOT
VERSION = "0.1.0"
FOLDER = "data-foundry-standalone"
EXECUTABLE = {"start.sh", "Start Data Foundry.command", "install-linux-launcher.sh"}


def main():
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / f"data-foundry-standalone-{VERSION}.zip"
    if out.exists():
        sys.exit(f"{out} already exists; choose another name.")
    files = {"app.py": ROOT / "app.py", "requirements.txt": ROOT / "requirements.txt", "LICENSE": ROOT / "LICENSE"}
    for folder in ("foundry", "templates", "static"):
        for path in sorted((ROOT / folder).rglob("*")):
            if path.is_file() and "__pycache__" not in path.parts and path.suffix != ".pyc":
                files[path.relative_to(ROOT).as_posix()] = path
    for name in LAUNCHER_FILES:
        files[name] = LAUNCHER_DIR / name
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for name, path in sorted(files.items()):
            info = zipfile.ZipInfo(f"{FOLDER}/{name}")
            info.compress_type = zipfile.ZIP_DEFLATED
            mode = 0o755 if name in EXECUTABLE else 0o644
            info.external_attr = (mode | 0o100000) << 16
            data = path.read_bytes()
            if name.endswith(".bat"):
                data = data.replace(b"\r\n", b"\n").replace(b"\n", b"\r\n")
            z.writestr(info, data)
    print(f"Built {out} ({out.stat().st_size // 1024} KB, {len(files)} files)")


if __name__ == "__main__":
    main()
