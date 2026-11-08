"""Pinned, resumable downloads. Preserve incomplete or invalid files."""
import argparse, hashlib, json, shutil, time
from pathlib import Path
from config import APP, MODELS, MANIFEST, initialize, model_file, model_links, matching_model_specs, existing_model_location, update_settings
import video_profile
import model_transfer

def digest(path):
    h=hashlib.sha256()
    with open(path,'rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''): h.update(b)
    return h.hexdigest()

def verified(path,item):
    return path.is_file() and path.stat().st_size==item['size'] and digest(path)==item['sha256']

def preserve_parts(partial, reason):
    suffix = f'.{reason}-{time.time_ns()}'
    for file in (partial, Path(str(partial)+'.aria2'), Path(str(partial)+'.segmented')):
        if file.exists():
            file.rename(file.with_name(file.name + suffix))


def run(verify_only=False, keys=None, job=None):
    initialize()
    entries=[m for m in MANIFEST if not keys or m['key'] in keys]
    for i,item in enumerate(entries,1):
        if model_transfer.stop_requested(job):
            raise InterruptedError('Model download stopped.')
        linked=model_file(item['key'])
        path=MODELS/item['path']
        print(f"[{i}/{len(entries)}] Verifying {item['path']}",flush=True)
        if linked != path and any(verified(linked,spec) for spec in matching_model_specs(linked,item)):
            print('Linked model SHA256 verified.',flush=True)
            continue
        if verified(path,item):
            if linked != path:
                links=model_links();links.pop(item['key'],None)
                (APP/'model-links.json').write_text(json.dumps(links,indent=2))
            print('SHA256 verified.',flush=True)
            continue
        if verify_only: raise RuntimeError('Missing or invalid: '+str(path))
        path.parent.mkdir(parents=True,exist_ok=True)
        if path.exists(): path.rename(path.with_name(path.name+f'.unverified-{time.time_ns()}'))
        partial=path.with_suffix(path.suffix+'.partial')
        control=Path(str(partial)+'.aria2')
        segmented=Path(str(partial)+'.segmented')
        if not control.exists() and not verified(partial,item):
            # A segmented file without a piece map is not a known contiguous prefix.
            actual_size=model_transfer.live_size(partial)
            prefix=segmented.read_text().strip() if segmented.exists() else None
            if (prefix is not None and prefix!=str(actual_size)) or actual_size >= item['size']:
                preserve_parts(partial,'unverified')
        if control.exists() or model_transfer.live_size(partial) != item['size']:
            remaining=max(0,item['size']-model_transfer.live_size(partial))
            if shutil.disk_usage(path.parent).free < remaining + 1024**3:
                raise RuntimeError(f'Insufficient disk space for {path.name}. Need {remaining/1e9:.1f} GB plus 1 GB reserve.')
            url=f"https://huggingface.co/{item['repo']}/resolve/{item['revision']}/{item['remote']}"
            model_transfer.download(url,partial,item['size'],job)
        if not verified(partial,item):
            preserve_parts(partial,'unverified')
            raise RuntimeError(f'SHA256 verification failed for {path.name}. File preserved. Retry in Settings.')
        partial.rename(path)
        if segmented.exists():
            receipt=(job or segmented.parent)/(segmented.name+f'.finished-{time.time_ns()}')
            shutil.move(str(segmented),str(receipt))
        links=model_links()
        if item['key'] in links:
            links.pop(item['key'])
            (APP/'model-links.json').write_text(json.dumps(links,indent=2))
        print('Downloaded and SHA256 verified.',flush=True)
    print('All selected model files verified.',flush=True)

def link_existing(folder):
    root=existing_model_location(folder)
    links=model_links()
    matched=set()
    sizes={spec['size'] for item in MANIFEST for spec in (item,*item.get('link_variants',[]))}
    candidates=[]
    import os
    if root.is_file():
        candidates.append(root)
    for base,dirs,names in (os.walk(root) if root.is_dir() else ()):
        dirs[:]=[d for d in dirs if d not in {'.git','.venv','venv','node_modules','__pycache__'}]
        for name in names:
            p=Path(base)/name
            if p.suffix.lower()=='.safetensors' and p.is_file() and p.stat().st_size in sizes: candidates.append(p)
    for item in MANIFEST:
        for p in candidates:
            specs=matching_model_specs(p,item)
            if not specs: continue
            print('Verifying existing '+str(p),flush=True)
            if any(verified(p,spec) for spec in specs):
                links[item['key']]=str(p);matched.add(item['key']);break
    if not matched:
        raise ValueError(f'No supported models passed size and SHA-256 verification in {root}.')
    dest=APP/'model-links.json'
    temp=dest.with_suffix('.tmp')
    temp.write_text(json.dumps(links,indent=2))
    temp.replace(dest)
    update_settings(existing_model_location=str(root))
    message=f'Linked {len(matched)} verified model file(s) from {root}.'
    if 'base' in matched:
        video_profile.activate(video_profile.INT8)
        message+=' Now using MiniMax H3 INT8.'
    else:
        message+=f" Linked supporting models. Active video model: {video_profile.settings()['active']}."
    print(message,flush=True)
    return message

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--verify',action='store_true');p.add_argument('--keys',nargs='+');p.add_argument('--link-folder');p.add_argument('--job-dir',type=Path)
    p.add_argument('--verify-hybrid',type=Path)
    p.add_argument('--link-grizzly')
    a=p.parse_args()
    try:
        if a.verify_hybrid:
            from video_profile import SIZE,SHA256
            print('Verifying linked GrizzlyMax INT4 hybrid...',flush=True)
            if not verified(a.verify_hybrid,dict(size=SIZE,sha256=SHA256)):
                raise ValueError('Linked hybrid verification failed. Link the GrizzlyMax checkpoint again.')
        if a.link_grizzly is not None:
            video_profile.link(a.link_grizzly)
            model_transfer.emit('result',ok=True,details=video_profile.active_label())
        elif a.link_folder is not None:
            model_transfer.emit('result',ok=True,details=link_existing(a.link_folder))
        else: run(a.verify,a.keys,a.job_dir)
    except InterruptedError:
        print('Download stopped. Progress saved.',flush=True)
    except Exception as error:
        model_transfer.emit('result',ok=False,error=str(error))
        raise SystemExit(1)
