"""Discover MiniMax video adapters and snapshot each selection for its worker."""
import json
import os
import uuid
from collections import Counter
from pathlib import Path

import config as cfg

LINKS = cfg.APP / 'lora-links.json'


def linked_files():
    return json.loads(LINKS.read_text('utf-8')) if LINKS.is_file() else []


def catalog():
    root = cfg.MODELS / 'loras'
    managed = managed_files()
    files = {str(p.resolve()): p.relative_to(root).as_posix() for p in root.rglob('*.safetensors')
             if p.resolve() not in managed}
    for name in linked_files():
        p = Path(name)
        if p.is_file() and str(p) not in files and p.resolve() not in managed:
            files[str(p)] = str(p)
    return files


def choices():
    files = catalog()
    names = Counter(Path(label).name for label in files.values())
    choices = [(Path(label).name if names[Path(label).name]==1 else label, path)
               for path,label in files.items()]
    return [('No LoRA', '')] + sorted(choices)


def check_adapter(path):
    from safetensors import safe_open
    with safe_open(str(path), framework='numpy') as data:
        if not any(any(tag in key.lower() for tag in ('lora_', '.lora.', 'hada_', 'lokr_', '.diff', '.oft_', '.dora_scale')) for key in data.keys()):
            raise ValueError('This file does not contain recognizable LoRA adapter weights.')


def managed_files():
    return {path.resolve() for key in ('turbo','fast_preview','dmd','refiner','pika_fl2va','pika_ref2va')
            for path in (cfg.model_file(key), cfg.MODELS/cfg.FILES[key]['path'])}

def link(location):
    path = Path(location.strip().strip('"')).resolve()
    if not path.is_file() or path.suffix.lower() != '.safetensors':
        raise ValueError('Choose a MiniMax H3 LoRA .safetensors file.')
    if path in managed_files():
        raise ValueError('This adapter is already managed by a generation preset or the upscale control.')
    check_adapter(path)
    files = linked_files()
    if str(path) not in files:
        files.append(str(path))
        tmp = LINKS.with_suffix('.' + uuid.uuid4().hex + '.tmp')
        tmp.write_text(json.dumps(files, indent=2), encoding='utf-8')
        os.replace(tmp, LINKS)
    return str(path)


def selection(slots):
    available = catalog()
    entries = []
    seen = set()
    for name, strength in slots:
        if not name:
            continue
        if name not in available:
            raise ValueError('A selected LoRA is unavailable. Refresh the list and select it again.')
        if name in seen:
            raise ValueError('Select each LoRA only once.')
        seen.add(name)
        check_adapter(Path(name))
        stat = Path(name).stat()
        entries.append(dict(path=name, strength=cfg.numeric(strength,-2,2,'LoRA strength'),
                            size=stat.st_size,mtime=stat.st_mtime_ns))
    return entries


def apply(model, entries, report):
    import comfy.sd
    import comfy.utils
    for entry in entries:
        if not entry['strength']:
            continue
        path = Path(entry['path'])
        stat = path.stat()
        if (stat.st_size,stat.st_mtime_ns) != (entry['size'],entry['mtime']):
            raise ValueError(f'{path.name} changed after selection. Select it again.')
        weights = comfy.utils.load_torch_file(str(path),safe_load=True)
        patched, _ = comfy.sd.load_lora_for_models(model,None,weights,entry['strength'],0)
        matched = sum(len(v) for v in patched.patches.values()) - sum(len(v) for v in model.patches.values())
        if not matched:
            raise ValueError(f'{path.name} did not match any video model weights. Choose a MiniMax H3 LoRA.')
        report(f'{path.name} at {entry["strength"]:g} applied to {matched} video weights')
        model = patched
    return model
