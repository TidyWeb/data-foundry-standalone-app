"""Build the standalone ZIP, launch it on the current OS, and exercise the packaged app.

This is intended for GitHub Actions' Linux, Windows, and macOS runners.
It keeps its extracted package, virtual environment, and logs in a fresh temp
folder so it never changes the working source or a user's normal app data.
"""
from __future__ import annotations

import os
import re
import shutil
import shlex
import signal
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PACKAGE_FOLDER = "data-foundry-standalone"
URL_PATTERN = re.compile(r"Data Foundry is starting at (http://127\.0\.0\.1:\d+/)")


def prepare_isolated_environment(work_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    home = work_dir / "home"
    temp = work_dir / "temp"
    home.mkdir()
    temp.mkdir()
    env["HOME"] = str(home)
    env["USERPROFILE"] = str(home)
    env["APPDATA"] = str(home / "AppData" / "Roaming")
    env["LOCALAPPDATA"] = str(home / "AppData" / "Local")
    env["TEMP"] = str(temp)
    env["TMP"] = str(temp)
    env["PYTHONDONTWRITEBYTECODE"] = "1"

    command_stubs = work_dir / "command-stubs"
    command_stubs.mkdir()
    if os.name == "nt":
        python = subprocess.list2cmdline([sys.executable])
        (command_stubs / "py.cmd").write_text(
            f'@echo off\r\nif "%~1"=="-3" shift\r\n{python} %1 %2 %3 %4 %5 %6 %7 %8 %9\r\n',
            encoding="utf-8",
        )
        (command_stubs / "python.cmd").write_text(
            f'@echo off\r\n{python} %*\r\n', encoding="utf-8"
        )
    else:
        python = shlex.quote(sys.executable)
        for command in ("python3.14", "python3.13", "python3", "python"):
            stub = command_stubs / command
            stub.write_text(f'#!/bin/sh\nexec {python} "$@"\n', encoding="utf-8")
            stub.chmod(0o755)

        # The Linux and macOS launchers open the local page after starting Flask.
        # Stub those commands so CI verifies the server without opening a browser.
        for command in ("xdg-open", "open"):
            stub = command_stubs / command
            stub.write_text('#!/bin/sh\nexit 0\n', encoding="utf-8")
            stub.chmod(0o755)

    # Make the launcher use the explicit Python version set up by Actions.
    env["PATH"] = str(command_stubs) + os.pathsep + env.get("PATH", "")
    return env

def build_and_extract(work_dir: Path) -> Path:
    archive = work_dir / "data-foundry-ci.zip"
    subprocess.run(
        [sys.executable, str(ROOT / "tools" / "make_standalone_zip.py"), str(archive)],
        cwd=ROOT,
        check=True,
    )

    app_root = work_dir / "unpacked" / PACKAGE_FOLDER
    with zipfile.ZipFile(archive) as package:
        names = set(package.namelist())
        required = {
            f"{PACKAGE_FOLDER}/app.py",
            f"{PACKAGE_FOLDER}/requirements.txt",
            f"{PACKAGE_FOLDER}/START_HERE.txt",
            f"{PACKAGE_FOLDER}/start.sh",
            f"{PACKAGE_FOLDER}/Start Data Foundry.command",
            f"{PACKAGE_FOLDER}/Start Data Foundry.bat",
            f"{PACKAGE_FOLDER}/foundry/__init__.py",
            f"{PACKAGE_FOLDER}/templates/index.html",
        }
        missing = sorted(required - names)
        if missing:
            raise RuntimeError(f"Standalone ZIP is missing required files: {missing}")

        for launcher in ("start.sh", "Start Data Foundry.command"):
            info = package.getinfo(f"{PACKAGE_FOLDER}/{launcher}")
            archived_mode = (info.external_attr >> 16) & 0o777
            if archived_mode & 0o111 == 0:
                raise RuntimeError(f"{launcher} is not executable in the standalone ZIP")

        app_root.parent.mkdir(parents=True)
        package.extractall(app_root.parent)

    # zipfile extraction does not restore Unix executable bits on every Python.
    if os.name != "nt":
        for launcher in ("start.sh", "Start Data Foundry.command"):
            path = app_root / launcher
            path.chmod(path.stat().st_mode | 0o111)

    return app_root


def launcher_command(app_root: Path) -> str | list[str]:
    if os.name == "nt":
        batch_file = app_root / "Start Data Foundry.bat"
        return f'cmd.exe /d /c call "{batch_file}"'
    if sys.platform == "darwin":
        return ["./Start Data Foundry.command"]
    return ["./start.sh"]


def stop_process(process: subprocess.Popen[bytes]) -> None:
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(process.pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
    else:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass

    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        if os.name == "nt":
            process.kill()
        else:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        process.wait(timeout=10)


def run_packaged_launcher(app_root: Path, work_dir: Path, env: dict[str, str]) -> None:
    log_path = work_dir / "launcher.log"
    popen_options: dict[str, object] = {}
    if os.name == "nt":
        popen_options["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        popen_options["start_new_session"] = True

    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        process = subprocess.Popen(
            launcher_command(app_root),
            cwd=app_root,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            **popen_options,
        )
        try:
            deadline = time.monotonic() + 360
            url = None
            while time.monotonic() < deadline:
                output = log_path.read_text(encoding="utf-8", errors="replace")
                match = URL_PATTERN.search(output)
                if match:
                    url = match.group(1)
                    break
                if process.poll() is not None:
                    raise RuntimeError(
                        f"Launcher exited with {process.returncode}. Log:\n{output}"
                    )
                time.sleep(1)

            if url is None:
                output = log_path.read_text(encoding="utf-8", errors="replace")
                raise TimeoutError(f"Launcher did not report its local URL. Log:\n{output}")

            deadline = time.monotonic() + 60
            last_error: Exception | None = None
            while time.monotonic() < deadline:
                try:
                    with urllib.request.urlopen(url, timeout=5) as response:
                        body = response.read()
                        if response.status != 200 or b"Data Foundry" not in body:
                            raise RuntimeError(
                                f"Unexpected home page response: HTTP {response.status}"
                            )
                    break
                except (urllib.error.URLError, TimeoutError, OSError) as error:
                    last_error = error
                    if process.poll() is not None:
                        output = log_path.read_text(encoding="utf-8", errors="replace")
                        raise RuntimeError(
                            f"Launcher exited with {process.returncode}. Log:\n{output}"
                        ) from error
                    time.sleep(1)
            else:
                raise TimeoutError(f"App did not respond at {url}: {last_error}")

            print(f"Packaged launcher served the home page at {url}")

            venv_python = (
                app_root / "venv" / "Scripts" / "python.exe"
                if os.name == "nt"
                else app_root / "venv" / "bin" / "python"
            )
            version = subprocess.check_output(
                [
                    str(venv_python),
                    "-c",
                    "import sys; print('.'.join(map(str, sys.version_info[:3])))",
                ],
                text=True,
            ).strip()
            print(f"Packaged runtime Python: {version}")

            # Run the existing application checks against the extracted release package.
            tests_dir = app_root / "tests"
            tests_dir.mkdir()
            for name in ("faults_test.py", "smoke_test.py"):
                copied_test = tests_dir / name
                shutil.copy2(ROOT / "tests" / name, copied_test)
                subprocess.run(
                    [str(venv_python), str(copied_test)],
                    cwd=app_root,
                    env=env,
                    check=True,
                )
        finally:
            stop_process(process)


def main() -> int:
    work_dir = Path(tempfile.mkdtemp(prefix="data-foundry-platform-check-"))
    print(f"Temporary check files: {work_dir}")
    env = prepare_isolated_environment(work_dir)
    app_root = build_and_extract(work_dir)
    run_packaged_launcher(app_root, work_dir, env)
    print(f"Packaged launcher and application checks passed on {sys.platform}.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
