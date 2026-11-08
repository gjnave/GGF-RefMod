"""One resumable model transfer. Only the worker process owns the output file."""
from __future__ import annotations

import json
import os
import secrets
import socket
import subprocess
import time
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

from config import APP

ARIA = APP / 'runtime/tools/aria2-1.37.0-win-64bit-build1/aria2c.exe'
HIDDEN = getattr(subprocess, 'CREATE_NO_WINDOW', 0)


def emit(event, **values):
    print(json.dumps(dict(event=event, **values)), flush=True)


def live_size(path):
    # Directory entries on Windows can report zero until the writer closes.
    try:
        with path.open('rb') as file:
            return file.seek(0, os.SEEK_END)
    except FileNotFoundError:
        return 0


def stop_requested(job):
    return job is not None and (job / 'stop-download').exists()


class Attempt:
    def __init__(self, job, item, total):
        self.job = job
        self.item = item
        self.total = total
        self.id = secrets.token_hex(8)

    def reconnect_requested(self):
        if stop_requested(self.job):
            raise InterruptedError('Model download stopped.')
        return self.job is not None and (self.job / ('reconnect-' + self.id)).exists()

    def update(self, received, speed):
        emit('transfer', attempt=self.id, name=self.item, total=self.total,
             received=min(received, self.total), speed=max(0, int(speed)))


def aria_transfer(url, path, attempt):
    # A private RPC session provides actual completed pieces, even in a sparse file.
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    secret = secrets.token_hex(24)
    opener = build_opener(ProxyHandler({}))

    def rpc(method, *args):
        body = json.dumps({'jsonrpc':'2.0', 'id':'worker', 'method':'aria2.' + method,
                           'params':['token:' + secret, *args]}).encode()
        req = Request(f'http://127.0.0.1:{port}/jsonrpc', data=body,
                      headers={'Content-Type':'application/json'})
        with opener.open(req, timeout=3) as response:
            return json.load(response)['result']

    command = [str(ARIA), '--no-conf=true', '--enable-rpc=true', '--rpc-listen-all=false',
               f'--rpc-listen-port={port}', f'--rpc-secret={secret}',
               f'--stop-with-process={os.getpid()}', '--console-log-level=warn']
    # Ask the local routing table; connecting a UDP socket sends no packets.
    try:
        with socket.socket(socket.AF_INET6, socket.SOCK_DGRAM) as route:
            route.connect(('2001:4860:4860::8888',443))
    except OSError:
        command.append('--disable-ipv6=true')
    proc = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                            creationflags=HIDDEN)
    ready = False
    try:
        for _ in range(60):
            if proc.poll() is not None:
                raise RuntimeError(f'aria2c could not start (exit {proc.returncode}).')
            try:
                rpc('getVersion')
                ready = True
                break
            except OSError:
                time.sleep(.1)
        if not ready:
            raise RuntimeError('The local aria2c process did not respond.')
        opts = {'dir':str(path.parent), 'out':path.name, 'continue':'true',
                'split':'4', 'max-connection-per-server':'4', 'min-split-size':'4M',
                'file-allocation':'none', 'auto-save-interval':'1', 'always-resume':'true',
                'auto-file-renaming':'false', 'allow-overwrite':'false', 'check-certificate':'true',
                'max-tries':'5', 'retry-wait':'2', 'connect-timeout':'30', 'timeout':'45'}
        gid = rpc('addUri', [url], opts)
        while True:
            state = rpc('tellStatus', gid, ['status','completedLength','downloadSpeed','errorMessage'])
            if state['status'] == 'complete':
                return True
            if state['status'] in ('error','removed'):
                raise RuntimeError('aria2c: ' + (state.get('errorMessage') or state['status']))
            attempt.update(int(state['completedLength']), int(state['downloadSpeed']))
            if attempt.reconnect_requested():
                return False
            time.sleep(.5)
    finally:
        if ready and proc.poll() is None:
            try:
                rpc('shutdown')  # Closes connections and saves the .aria2 piece map.
            except OSError:
                pass
        try:
            proc.wait(timeout=12 if ready else 1)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def curl_transfer(url, path, attempt):
    if Path(str(path) + '.aria2').exists() or Path(str(path) + '.segmented').exists():
        raise RuntimeError('Run INSTALL.bat to restore aria2c and resume this segmented download.')
    command = ['curl.exe','-q','--fail','--silent','--show-error','--location','--retry','5',
               '--connect-timeout','30','--continue-at','-','--output',str(path),url]
    last_size, last_time = live_size(path), time.monotonic()
    with subprocess.Popen(command, creationflags=HIDDEN) as proc:
        try:
            while proc.poll() is None:
                received, now = live_size(path), time.monotonic()
                attempt.update(received, (received-last_size) / max(.001, now-last_time))
                if attempt.reconnect_requested():
                    return False
                last_size, last_time = received, now
                time.sleep(.5)
            if proc.returncode:
                raise RuntimeError(f'curl exited with code {proc.returncode}. The partial file was kept.')
            return True
        finally:
            if proc.poll() is None:
                proc.terminate()
                proc.wait()


def download(url, path, expected_size, job=None):
    name = path.name.removesuffix('.partial')
    while True:
        if stop_requested(job):
            raise InterruptedError('Model download stopped.')
        attempt = Attempt(job, name, expected_size)
        if ARIA.is_file():
            marker = Path(str(path) + '.segmented')
            if not marker.exists():
                marker.write_text(str(live_size(path)), encoding='ascii')
            complete = aria_transfer(url, path, attempt)
        else:
            complete = curl_transfer(url, path, attempt)
        if complete:
            emit('stage', message='Verifying SHA-256: ' + name)
            return
        emit('stage', message='Reconnecting: ' + name)
