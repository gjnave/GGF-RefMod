"""App-relative paths and validated product settings. No inference imports."""
from pathlib import Path
import json, math, os, re, secrets, threading

APP = Path(__file__).resolve().parent
MODELS = APP / 'models'
JOBS = APP / 'jobs'
OUTPUTS = APP / 'outputs'
LIBRARY = MODELS / 'refmods'
MANIFEST = json.loads((APP / 'models.lock.json').read_text())
FILES = {x['key']: x for x in MANIFEST}
VERSION = '0.11.2'
DEFAULT_PRESET = 'pika_auto'

# (UI label, managed adapter, fixed steps, first-pass sampler, video shift)
PRESETS = {
    'pika_auto': ('Pika Flex · automatic (default, 6 steps)', None, 6, 'lcm', 12.0),
    'turbo': ('Turbo (8 steps)', 'turbo', 8, 'res_multistep', 12.0),
    'fast': ('Fast Preview (4 steps)', 'fast_preview', 4, 'euler', 6.0),
    'pika': ('Pika Flex · text / first-last (6 steps)', 'pika_fl2va', 6, 'lcm', 12.0),
    'pika_refs': ('Pika Flex · RefMods (6 steps)', 'pika_ref2va', 6, 'lcm', 12.0),
    'custom': ('Custom steps', None, None, 'res_multistep', 12.0),
}
FAST_DOWNLOAD = 'Fast Preview adapter only (1.96 GB)'

def saved_preset(data):
    return data.get('preset', 'turbo' if data.get('turbo', True) else 'custom')

def sampling_settings(preset, steps):
    if preset not in PRESETS:
        raise ValueError('Choose a generation preset.')
    _,adapter,fixed,sampler,shift=PRESETS[preset]
    return adapter,fixed or numeric(steps,1,100,'Steps',True),sampler,shift

def model_links():
    p=APP/'model-links.json'
    return json.loads(p.read_text()) if p.exists() else {}

def model_file(key):
    return Path(model_links().get(key, str(MODELS/FILES[key]['path'])))

def model_name(key):
    return model_file(key).name

def existing_model_location(value):
    value = str(value or '').strip().strip('"').strip()
    if not value:
        raise ValueError('Choose a models folder or paste a .safetensors file path before linking.')
    path = Path(value).expanduser().resolve()
    if path.is_dir() or (path.is_file() and path.suffix.lower() == '.safetensors'):
        return path
    raise ValueError(f'Cannot access this models folder or .safetensors file: {path}')

def matching_model_specs(path, item):
    if not path.is_file():
        return []
    size = path.stat().st_size
    return [spec for spec in (item, *item.get('link_variants', [])) if spec['size'] == size]

def initialize():
    for p in (MODELS, JOBS, OUTPUTS, LIBRARY): p.mkdir(parents=True, exist_ok=True)

def frames_for(seconds):
    seconds=float(seconds)
    if not math.isfinite(seconds) or seconds <= 0: raise ValueError('Duration must be positive.')
    n=max(5,round(seconds*24))
    return n + (5-n%17)%17

def canvas(ratio, megapixels):
    ratios={'Landscape 16:9':16/9,'Portrait 9:16':9/16,'Square 1:1':1.0}
    if ratio not in ratios: raise ValueError('Choose an available aspect ratio.')
    mp=float(megapixels)
    if not math.isfinite(mp) or not .1 <= mp <= 4: raise ValueError('Canvas must be between 0.1 and 4 megapixels.')
    r=ratios[ratio]
    return max(32,round(math.sqrt(mp*1e6*r)/32)*32),max(32,round(math.sqrt(mp*1e6/r)/32)*32)

def numeric(value, low, high, label, integer=False):
    v=float(value)
    if not math.isfinite(v) or not low <= v <= high or (integer and v != int(v)):
        raise ValueError(f'{label} must be {low}–{high}.')
    return int(v) if integer else v

def seed_input(value):
    text='' if value is None else str(value).strip()
    if not text or text.lower()=='random': return ''
    try: seed=int(text)
    except ValueError: raise ValueError('Enter a whole number or leave the seed blank for Random.')
    if seed == -1: return ''
    if not 0 <= seed < 2**64: raise ValueError('Seed must be between 0 and 2^64-1, or -1.')
    return str(seed)

def seed_value(value):
    text=seed_input(value)
    return int(text) if text else secrets.randbelow(2**53)

def safe_name(value):
    name=re.sub(r'[^A-Za-z0-9_-]+','_',str(value).strip()).strip('_')[:64]
    if not name: raise ValueError('Enter a name for this RefMod.')
    return name

def library_path(name):
    p=(LIBRARY / str(name)).resolve()
    if not p.is_relative_to(LIBRARY.resolve()) or not p.is_file() or p.suffix!='.safetensors':
        raise ValueError('Choose an existing RefMod from the library.')
    return p

def model_status():
    links=model_links()
    rows = []
    for item in MANIFEST:
        key = item['key']
        path = model_file(key)
        matches = matching_model_specs(path, item)
        status = ('Linked' if key in links else 'Installed') + ' (size checked)'
        rows.append(dict(model=key, file=str(path), status=status if matches else 'Missing or incomplete',
                         GB=round((matches or [item])[0]['size']/1e9, 2)))
    return rows

def require_models(keys):
    if 'fast_preview' in keys:
        path=model_file('fast_preview')
        if not path.is_file() or path.stat().st_size != FILES['fast_preview']['size']:
            raise ValueError(f'In Settings, select "{FAST_DOWNLOAD}" and click Download missing models, or link an existing adapter folder.')
    missing=[FILES[k]['path'] for k in keys if not matching_model_specs(model_file(k), FILES[k])]
    if missing: raise ValueError('Open Settings to download or link these models:\n'+'\n'.join(missing))

SETTINGS_LOCK=threading.RLock()
SETTINGS_FILE=APP/'settings.json'

def read_settings():
    p=SETTINGS_FILE
    defaults={'phone_enabled':False,'username':'','password':'','recycle_videos':True}
    return {**defaults,**json.loads(p.read_text(encoding='utf-8'))} if p.exists() else defaults

def update_settings(**changes):
    with SETTINGS_LOCK:
        p=SETTINGS_FILE
        tmp=p.with_name('settings-'+secrets.token_hex(8)+'.tmp')
        tmp.write_text(json.dumps({**read_settings(),**changes},indent=2),encoding='utf-8')
        os.replace(tmp,p)
