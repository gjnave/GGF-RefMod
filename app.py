from __future__ import annotations
import argparse, json, os, subprocess, sys
from config import APP,initialize

def self_check():
    initialize()
    import torch,gradio,av
    print('Python',sys.version,'Gradio',gradio.__version__,'Torch',torch.__version__,'CUDA',torch.version.cuda,flush=True)
    if not torch.cuda.is_available():raise RuntimeError('An NVIDIA CUDA GPU is required for generation.')
    print('GPU',torch.cuda.get_device_name(0),'Capability',torch.cuda.get_device_capability(0),'Architectures',torch.cuda.get_arch_list(),flush=True)
    x=torch.ones((32,32),device='cuda');assert (x@x).mean().item()==32
    del x;torch.cuda.empty_cache()
    subprocess.run([sys.executable,str(APP/'core_worker.py'),'--probe'],check=True)
    from ui import build_ui
    build_ui(remote=False)
    print('CUDA kernel, bundled imports, and interface checks passed. Models are managed in Settings.',flush=True)

def single_instance():
    import msvcrt,webbrowser
    folder=APP/'runtime';folder.mkdir(exist_ok=True)
    handle=(folder/'app.lock').open('a+b')
    if handle.tell()==0:handle.write(b'0');handle.flush()
    handle.seek(0)
    try:msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
    except OSError:
        handle.close()
        saved=folder/'server-address.json'
        if saved.exists():
            port=json.loads(saved.read_text())['port']
            if 7900<=port<=7909:webbrowser.open(f'http://127.0.0.1:{port}')
        print('Minimax Refmod is already running.');sys.exit(0)
    return handle

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--self-check',action='store_true');p.add_argument('--no-browser',action='store_true')
    a=p.parse_args()
    if a.self_check:self_check()
    else:
        instance=single_instance()
        from ui import launch
        launch(not a.no_browser)
