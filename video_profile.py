"""Verified in-place link for the supported GrizzlyMax video checkpoint."""
import hashlib
import json
import os
import uuid
from pathlib import Path

from config import APP, model_file

PROFILE = APP / 'video-model.json'
SIZE = 12540867891
SHA256 = '8bf716aaf1c8a5f8da4c28bbdbaeaf184e83b320d036b326507e4cd055a8853f'
INT8 = 'MiniMax H3 INT8'
HYBRID = 'GrizzlyMax INT4 Hybrid'


def settings():
    return json.loads(PROFILE.read_text('utf-8')) if PROFILE.exists() else {'active': INT8}


def save(data):
    tmp = PROFILE.with_suffix('.' + uuid.uuid4().hex + '.tmp')
    tmp.write_text(json.dumps(data, indent=2), encoding='utf-8')
    os.replace(tmp, PROFILE)


def linked_path(profile):
    path = Path(profile.get('path', ''))
    if not path.is_file():
        raise ValueError('Select the GrizzlyMax models folder or hybrid checkpoint and link it.')
    stat = path.stat()
    if stat.st_size != SIZE or profile.get('sha256') != SHA256 or stat.st_mtime_ns != profile.get('mtime'):
        raise ValueError('The linked checkpoint changed. Link it again to verify it.')
    return path


def link(location):
    root = Path(location.strip().strip('"')).expanduser()
    if not location.strip() or not root.exists():
        raise ValueError('Select an existing GrizzlyMax models folder or checkpoint.')
    candidates = root.rglob('*.safetensors') if root.is_dir() else [root]
    for path in candidates:
        before = path.stat()
        if before.st_size != SIZE:
            continue
        with path.open('rb') as source:
            digest = hashlib.file_digest(source, 'sha256').hexdigest()
        if digest == SHA256 and path.stat().st_mtime_ns == before.st_mtime_ns:
            save(dict(active=HYBRID, path=str(path.resolve()), sha256=digest, mtime=before.st_mtime_ns))
            return
    raise ValueError('No matching GrizzlyMax b20-49 W4A8 hybrid checkpoint found. The previous selection is still active.')


def activate(model):
    profile = settings()
    if model == HYBRID:
        linked_path(profile)
    elif model != INT8:
        raise ValueError('Unknown video model.')
    save(dict(profile, active=model))


def active_label():
    profile = settings()
    if profile['active'] == HYBRID:
        return f"Active video model: {HYBRID}\n{profile['path']}"
    return f'Active video model: {INT8}\n{model_file("base")}'
