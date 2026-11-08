"""Verified source updates with backups and optional isolated dependency upgrades."""
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.request
import uuid
import zipfile

APP = Path(__file__).resolve().parent
MANIFEST_URL = "https://getgoingfast.pro/tools/ggf-refmod/update.json"
TORCH = ["torch==2.10.0", "torchvision==0.25.0", "torchaudio==2.10.0"]
TORCH_INDEX = "https://download.pytorch.org/whl/cu130"
UV_URL = "https://github.com/astral-sh/uv/releases/download/0.9.7/uv-x86_64-pc-windows-msvc.zip"
UV_SHA256 = "5d250c32d3604e28dbe18dc65c668ff628c53e00dde2c642576e831e4a60da64"


def version(value):
    if not re.fullmatch(r"\d+\.\d+\.\d+", str(value)):
        raise ValueError("The update contains an invalid version.")
    return tuple(map(int, value.split(".")))


def installed_version(app=APP):
    text = (app / "config.py").read_text(encoding="utf-8")
    return re.search(r"^VERSION = ['\"]([^'\"]+)['\"]", text, re.M).group(1)


def fingerprint(requirements, torch=TORCH, index=TORCH_INDEX):
    requirements=requirements.replace(b"\r\n",b"\n")
    return hashlib.sha256(requirements + json.dumps([torch, index], sort_keys=True).encode()).hexdigest()


def fetch(url):
    if not url.startswith("https://"):
        raise ValueError("Updates require an HTTPS download.")
    request = urllib.request.Request(url, headers={"User-Agent": "GGF-RefMod-Updater", "Cache-Control": "no-cache"})
    return urllib.request.urlopen(request, timeout=60)


def check(app=APP):
    with fetch(MANIFEST_URL+"?check="+str(time.time_ns())) as response:
        manifest = json.load(response)
    version(manifest["version"])
    if not re.fullmatch(r"[a-f0-9]{40}", manifest["commit"]):
        raise ValueError("The update commit is invalid.")
    if not re.fullmatch(r"[a-f0-9]{64}", manifest["sha256"]):
        raise ValueError("The update checksum is invalid.")
    return manifest, version(manifest["version"]) > version(installed_version(app))


def source_path(name):
    path = PurePosixPath(name)
    if "\\" in name or path.is_absolute() or ".." in path.parts or not path.parts:
        raise ValueError("The update contains an unsafe file path.")
    if any(":" in part or part.endswith((".", " ")) for part in path.parts):
        raise ValueError("The update contains an invalid Windows file path.")
    if path.parts[0] in ("assets", "vendor", "workflows"):
        allowed = True
    else:
        allowed = len(path.parts) == 1 and (
            path.suffix in (".py", ".md", ".txt") or name in
            ("LICENSE", ".gitignore", "models.lock.json", "sources.lock.json")
        )
    if not allowed or path.suffix.lower() in (".bat", ".exe", ".safetensors", ".pyc", ".zip"):
        raise ValueError("The update tried to replace a private or unsupported file: " + name)
    return path


def prepare(app=APP, manifest=None):
    if manifest is None:
        manifest, available = check(app)
        if not available:
            return None
    folder = app / "runtime/updates" / uuid.uuid4().hex
    folder.mkdir(parents=True)
    archive = folder / "source.zip"
    errors = []
    for url in manifest["source_urls"]:
        try:
            digest = hashlib.sha256()
            size = 0
            with fetch(url) as response, archive.open("wb") as output:
                while block := response.read(1024 * 1024):
                    size += len(block)
                    if size > 128 * 1024**2:
                        raise ValueError("The update download is too large.")
                    digest.update(block)
                    output.write(block)
            if digest.hexdigest() != manifest["sha256"]:
                raise ValueError("Update checksum verification failed.")
            break
        except (OSError, ValueError) as error:
            errors.append(str(error))
    else:
        raise RuntimeError("Could not download a verified update. " + " ".join(errors))
    source = folder / "source"
    source.mkdir()
    files = []
    with zipfile.ZipFile(archive) as package:
        if len(package.infolist()) > 5000 or sum(item.file_size for item in package.infolist()) > 512 * 1024**2:
            raise ValueError("The update archive exceeds its size limit.")
        for item in package.infolist():
            if item.is_dir():
                continue
            if not item.filename.startswith("Minimax-Refmod/") or (item.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError("The update has an invalid root or symbolic link.")
            name = item.filename[len("Minimax-Refmod/"):]
            path = source_path(name)
            if name.casefold() in {value.casefold() for value in files}:
                raise ValueError("The update contains a duplicate file.")
            target = source.joinpath(*path.parts)
            if not target.resolve().is_relative_to(source.resolve()):
                raise ValueError("The update path escaped its source folder.")
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(package.read(item))
            files.append(name)
    if installed_version(source) != manifest["version"]:
        raise ValueError("The downloaded app version does not match the update.")
    required = {"app.py", "app_updates.py", "config.py", "runtime.py", "ui.py", "requirements.txt"}
    if not required.issubset(files):
        raise ValueError("The update is missing required app files.")
    manifest = dict(manifest, files=files)
    (folder / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    shutil.copy2(Path(__file__), folder / "apply.py")
    return folder


def active_python(app=APP):
    pointer = app / "runtime/active-python.txt"
    if pointer.is_file():
        path = (app / pointer.read_text(encoding="utf-8").strip()).resolve()
        if not path.is_relative_to((app / "runtime/environments").resolve()) or not path.is_file():
            raise RuntimeError("The updated Python environment cannot be found.")
        return path
    return app / ".venv/Scripts/python.exe"


@contextmanager
def app_lock(app):
    import msvcrt
    folder = app / "runtime"
    folder.mkdir(exist_ok=True)
    handles = []
    try:
        for name in ("app.lock", "inference.lock", "update.lock"):
            handle = (folder / name).open("a+b")
            if handle.tell() == 0:
                handle.write(b"0")
                handle.flush()
            handle.seek(0)
            try:
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
            except OSError:
                handle.close()
                raise RuntimeError("Close RefMod and its generation worker, then run UPDATE.bat again.") from None
            handles.append(handle)
        yield
    finally:
        for handle in handles:
            handle.close()


def run(command, folder):
    result = subprocess.run([str(value) for value in command], cwd=folder,
                            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace",
                            env={**os.environ, "UV_CACHE_DIR": str(folder.parent.parent / "uv-cache")},
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    with (folder / "update.log").open("a", encoding="utf-8") as output:
        output.write(result.stdout + "\n")
    if result.returncode:
        raise RuntimeError("Update validation failed. See " + str(folder / "update.log"))


def uv(app):
    cached = app.parent / "installer-cache/uv.exe"
    if cached.is_file():
        return cached
    tools = app / "runtime/update-tools"
    tools.mkdir(parents=True, exist_ok=True)
    archive = tools / "uv.zip"
    with fetch(UV_URL) as response:
        archive.write_bytes(response.read())
    if hashlib.sha256(archive.read_bytes()).hexdigest() != UV_SHA256:
        raise RuntimeError("The Python installer checksum did not match.")
    with zipfile.ZipFile(archive) as package:
        (tools / "uv.exe").write_bytes(package.read("uv.exe"))
    return tools / "uv.exe"


def atomic_write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    temp.write_bytes(content)
    os.replace(temp, path)


def apply(folder, app=APP):
    app = app.resolve()
    folder = folder.resolve()
    if not folder.is_relative_to((app / "runtime/updates").resolve()):
        raise ValueError("The staged update is outside this app.")
    manifest = json.loads((folder / "manifest.json").read_text(encoding="utf-8"))
    source = folder / "source"
    if version(manifest["version"]) <= version(installed_version(app)):
        return "The app is already up to date."
    with app_lock(app):
        python = active_python(app)
        torch = manifest.get("torch", TORCH)
        index = manifest.get("torch_index", TORCH_INDEX)
        new_fingerprint = fingerprint((source / "requirements.txt").read_bytes(), torch, index)
        receipt = app / "runtime/update-state.json"
        current_fingerprint = (json.loads(receipt.read_text()).get("dependencies") if receipt.exists()
                               else fingerprint((app / "requirements.txt").read_bytes()))
        new_python = None
        if new_fingerprint != current_fingerprint:
            new_env = app / "runtime/environments" / (manifest["version"] + "-" + uuid.uuid4().hex[:8])
            tool = uv(app)
            run([tool, "venv", "--python", sys._base_executable, new_env], folder)
            new_python = new_env / "Scripts/python.exe"
            run([tool, "pip", "install", "--python", new_python, *torch, "--index-url", index], folder)
            run([tool, "pip", "install", "--python", new_python, "-r", source / "requirements.txt"], folder)
            run([tool, "pip", "check", "--python", new_python], folder)
            python = new_python
        run([python, source / "app.py", "--self-check"], folder)
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        backup = app / "runtime/update-backups" / stamp
        backup.mkdir(parents=True)
        paths = list(manifest["files"]) + ["runtime/active-python.txt", "runtime/update-state.json"]
        existed = []
        for name in paths:
            if name in manifest["files"]:
                source_path(name)
            target = app / name
            if not target.resolve().is_relative_to(app):
                raise ValueError("An app file points outside the installation: " + name)
            if target.exists():
                dest = backup / name
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(target, dest)
                existed.append(name)
        (backup / "receipt.json").write_text(json.dumps(dict(version=manifest["version"], files=paths, existed=existed), indent=2))
        changed = []
        try:
            for name in manifest["files"]:
                atomic_write(app / name, (source / name).read_bytes())
                changed.append(name)
            if new_python:
                atomic_write(app / "runtime/active-python.txt", new_python.relative_to(app).as_posix().encode())
                changed.append("runtime/active-python.txt")
            atomic_write(receipt, json.dumps(dict(version=manifest["version"], commit=manifest["commit"], dependencies=new_fingerprint)).encode())
            changed.append("runtime/update-state.json")
        except Exception:
            for name in reversed(changed):
                if name in existed:
                    atomic_write(app / name, (backup / name).read_bytes())
                elif (app / name).exists():
                    preserved = backup / "new-files" / name
                    preserved.parent.mkdir(parents=True, exist_ok=True)
                    (app / name).rename(preserved)
            raise
        return "Updated to " + manifest["version"] + ". Backup: " + str(backup)


def launch_prepared(folder, app=APP, wait_pid=None, relaunch=False):
    command = [sys.executable, str(folder / "apply.py"), "--apply", str(folder), "--app-dir", str(app)]
    if wait_pid:
        command += ["--wait-pid", str(wait_pid)]
    if relaunch:
        command += ["--relaunch"]
    log = (folder / "launcher.log").open("a", encoding="utf-8")
    try:
        subprocess.Popen(command, cwd=app, stdout=log, stderr=subprocess.STDOUT,
                         creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    finally:
        log.close()


def wait_for_exit(pid):
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.OpenProcess.restype = wintypes.HANDLE
    handle = kernel.OpenProcess(0x100000, False, pid)
    if not handle:
        return
    try:
        if kernel.WaitForSingleObject(wintypes.HANDLE(handle), 300000) != 0:
            raise RuntimeError("RefMod did not close. Run UPDATE.bat after closing it.")
    finally:
        kernel.CloseHandle(wintypes.HANDLE(handle))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true")
    parser.add_argument("--apply", type=Path)
    parser.add_argument("--app-dir", type=Path, default=APP)
    parser.add_argument("--wait-pid", type=int)
    parser.add_argument("--relaunch", action="store_true")
    args = parser.parse_args()
    app = args.app_dir
    try:
        if args.wait_pid:
            wait_for_exit(args.wait_pid)
        folder = args.apply
        if not folder:
            with app_lock(app):
                folder = prepare(app)
            if folder is None:
                print("You are up to date.")
                return
        message=apply(folder, app)
        atomic_write(app/"runtime/update-result.json",json.dumps(dict(status="complete",message=message)).encode())
        print(message, flush=True)
        if args.relaunch:
            subprocess.Popen([str(active_python(app)), str(app / "app.py"), "--no-browser"], cwd=app,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except Exception as error:
        atomic_write(app/"runtime/update-result.json",json.dumps(dict(status="failed",message=str(error))).encode())
        if args.relaunch:
            subprocess.Popen([str(active_python(app)),str(app/"app.py"),"--no-browser"],cwd=app,
                             creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        print("Update failed:", error, flush=True)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
