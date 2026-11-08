"""Shared job lock, process ownership, saved workspaces, and real progress."""
from __future__ import annotations
import atexit, json, os, re, shutil, subprocess, sys, threading, time, uuid
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
from config import APP,JOBS,OUTPUTS,LIBRARY,initialize,read_settings

initialize()
LOCK=threading.RLock()
ACTIVE=None
PROCESSES={}
PENDING=deque()
GENERATIONS=defaultdict(list)
SHUTTING_DOWN=False

def write_json(path,data):
    temp=path.with_suffix('.tmp')
    temp.write_text(json.dumps(data,indent=2),encoding='utf-8')
    os.replace(temp,path)

def workspace(token):
    if not re.fullmatch(r'[a-f0-9]{32}',str(token)): raise ValueError('Reload the page to initialize your workspace.')
    p=JOBS/'workspaces'/token;p.mkdir(parents=True,exist_ok=True);return p

def load_workspace(token):
    with LOCK:
        p=workspace(token)/'state.json'
        return json.loads(p.read_text(encoding='utf-8')) if p.exists() else {}

def save_workspace(token,data):
    with LOCK:
        write_json(workspace(token)/'state.json',data)

def is_busy():
    with LOCK: return ACTIVE is not None

def ingest(path,jobdir,index=0):
    if not path: return None
    source=Path(path)
    if not source.is_file() or source.stat().st_size==0: raise ValueError('Upload incomplete. Select the file again.')
    dest=jobdir/'inputs'/f'{index:02d}-{source.name}'
    dest.parent.mkdir(exist_ok=True)
    shutil.copy2(source,dest)
    return str(dest)

def start(token,request,controls=None):
    with LOCK:
        if SHUTTING_DOWN:raise ValueError('The app is shutting down.')
        saved=load_workspace(token)
        generation=request['kind']=='generate'
        if ACTIVE and not generation: raise ValueError('An app job is already running. Wait for it to finish or stop it.')
        jobid=datetime.now().strftime('%Y%m%d-%H%M%S')+'-'+uuid.uuid4().hex[:8]
        folder=JOBS/jobid;folder.mkdir()
        request=dict(request,job_dir=str(folder),output_name='Minimax-Refmod-'+jobid)
        for k in ('audio','first_frame','last_frame'):
            if request.get(k): request[k]=ingest(request[k],folder,{'audio':100,'first_frame':101,'last_frame':102}[k])
        if request.get('media'): request['media']=[ingest(x,folder,i) for i,x in enumerate(request['media'])]
        if request['kind']=='create':request['name']+='_'+jobid
        now=time.time()
        state={'id':jobid,'token':token,'kind':request['kind'],'status':'queued','stage':'Waiting in queue','submitted':now,'started':now,'result':{},'job_dir':str(folder)}
        if generation:
            state['label']=' '.join(request['prompt'].split())[:100]
        write_json(folder/'request.json',request)
        saved.update(job=jobid,controls=controls or saved.get('controls',{}))
        save_workspace(token,saved)
        _write(state)
        if generation:GENERATIONS[token].append(state)
        PENDING.append(state)
        _start_next()
        return jobid

def _write(state):
    write_json(Path(state['job_dir'])/'status.json',state)

def _start_next():
    """Called under LOCK, including the handoff when a worker exits."""
    global ACTIVE
    while PENDING and ACTIVE is None and not SHUTTING_DOWN:
        state=PENDING.popleft()
        try:
            request=json.loads((Path(state['job_dir'])/'request.json').read_text(encoding='utf-8'))
            state.update(status='running',stage='Starting',started=time.time())
            _write(state)
            ACTIVE=state
            threading.Thread(target=_run,args=(state,request),daemon=True).start()
        except Exception as exc:
            ACTIVE=None
            state.update(status='error',stage=str(exc),finished=time.time())
            _write(state)

def restore_queue():
    """Rebuild waiting submissions once, at server startup."""
    with LOCK:
        recovered=[]
        for path in JOBS.glob('20*/status.json'):
            try:
                state=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError) as exc:
                print(f'Cannot read job status {path}: {exc}',file=sys.stderr)
                continue
            if state.get('kind')!='generate':continue
            if Path(state.get('job_dir','')).resolve()!=path.parent.resolve():
                # Copied queue entries must never run or write status in another installation.
                print(f'Job path needs relocation: {path}',file=sys.stderr)
                continue
            if 'submitted' not in state:
                if not state.get('result',{}).get('files'):continue
                state['submitted']=state.get('started',0)
                state['label']=state['id']
            recovered.append(state)
        for state in sorted(recovered,key=lambda item:item['submitted']):
            GENERATIONS[state['token']].append(state)
            if state['status']=='queued':
                PENDING.append(state)
            elif state['status']=='running':
                state.update(status='stopped',stage='Interrupted by app restart.',finished=time.time())
                _write(state)
        _start_next()

def generation_list(token):
    with LOCK:
        positions={s['id']:i for i,s in enumerate(PENDING,1)}
        rows=GENERATIONS.get(token,[])
        live=[s for s in rows if s['status'] in ('running','queued')]
        recent=[s for s in reversed(rows) if s['status'] not in ('running','queued')][:20]
        return [dict(s,position=positions.get(s['id'])) for s in live+recent]

def workspace_clips(token):
    """All playable clips in this workspace, newest first, including batch outputs."""
    with LOCK:
        clips=[]
        for row in reversed(GENERATIONS.get(token,[])):
            files=row.get('result',{}).get('files',[])
            for index in range(len(files)-1,-1,-1):
                clips.append((row['id'],index+1,row['label'],files[index]))
        return clips

def workspace_clip_path(token,key):
    with LOCK:
        for job,index,_,file in workspace_clips(token):
            if key!=f'{job}-{index}':continue
            root=OUTPUTS.resolve()
            expected=root/f'Minimax-Refmod-{job}-{index:02d}.mp4'
            source=Path(file).resolve()
            if source!=expected or source.parent!=root:
                raise ValueError('This clip is outside the app output folder.')
            return source
    raise ValueError('Select a clip from this workspace.')

def clip_project(token,key):
    with LOCK:
        source=workspace_clip_path(token,key)
        if not source.is_file():raise ValueError('This clip is no longer in the gallery.')
        job,number=key.rsplit('-',1)
        state=next(row for row in GENERATIONS[token] if row['id']==job)
        request=json.loads((JOBS/job/'request.json').read_text(encoding='utf-8'))
        record=dict(state.get('clip_records',{}).get(source.name,{}))
        elapsed=(state['finished']-state['started']) if state.get('finished') and state.get('started') else None
        return source,int(number),request,record,elapsed

def delete_clip(token,key,destination):
    """Move only an owned, generated MP4. Its original batch index stays stable."""
    with LOCK:
        source=workspace_clip_path(token,key)
        if not source.is_file():
            return 'This clip is already removed.'
        recycle=read_settings()['recycle_videos'] is True
        if destination!=('bin' if recycle else 'folder'):
            raise ValueError('The delete setting changed. Please confirm the new destination.')
        if recycle:
            from recycle_video import recycle_video
            recycle_video(source)
            return 'Video moved to the Windows Recycle Bin.'
        root=source.parent
        trash=root/'Deleted videos'
        if trash.resolve().parent!=root or trash.is_symlink() or trash.resolve()!=trash:
            raise ValueError('The Deleted videos folder must be inside outputs.')
        trash.mkdir(exist_ok=True)
        destination=trash/source.name
        if destination.exists():
            destination=trash/f'{source.stem}-{uuid.uuid4().hex}{source.suffix}'
        source.rename(destination)
        return 'Video moved to outputs/Deleted videos.'

def cancel_waiting(token,jobid):
    with LOCK:
        for state in PENDING:
            if state['id']==jobid and state['token']==token:
                state.update(status='cancelled',stage='Cancelled before starting.',finished=time.time())
                _write(state)
                PENDING.remove(state)
                return 'Cancelled the waiting submission.'
        raise ValueError('Select one of your waiting submissions. It may have already started.')

def _run(state,request):
    global ACTIVE
    folder=Path(state['job_dir']);kind=request['kind'];answer=None
    if kind in ('download','verify','link'):
        args=[sys.executable,'-u',str(APP/'download_models.py'),'--job-dir',str(folder)]
        if kind=='link':
            flag='--link-grizzly' if request.get('family')=='grizzly' else '--link-folder'
            args += [flag,request['folder']]
        else:
            if kind=='verify':args+=['--verify']
            args+=['--keys',*request['keys']]
            if request.get('verify_hybrid'):
                args+=['--verify-hybrid',request['verify_hybrid']]
    else:args=[sys.executable,'-u',str(APP/'core_worker.py'),str(folder/'request.json')]
    flags=getattr(subprocess,'CREATE_NO_WINDOW',0)
    try:
        with (folder/'worker.log').open('w',encoding='utf-8') as log:
            with LOCK:
                if state.get('cancelled'):raise RuntimeError('Stopped by user.')
                process=subprocess.Popen(args,cwd=APP,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True,encoding='utf-8',errors='replace',bufsize=1,creationflags=flags)
                PROCESSES[state['id']]=process
            for line in process.stdout:
                log.write(line);log.flush()
                try:event=json.loads(line)
                except (ValueError,TypeError):
                    if kind in ('download','verify','link') and line.strip():
                        with LOCK:
                            state['stage']=line.strip()[:240]
                            state.pop('transfer',None)
                    continue
                with LOCK:
                    if event.get('event')=='transfer':
                        state['transfer']=event
                        state['stage']='Downloading '+event['name']
                    elif event.get('event')=='stage':
                        state['stage']=event['message']
                        state.pop('progress',None)
                        state.pop('transfer',None)
                    elif event.get('event')=='progress':state['progress']=[event['value'],event['total']]
                    elif event.get('event')=='clip' and kind=='generate':
                        files=state['result'].setdefault('files',[])
                        if event['file'] not in files:
                            files.append(event['file'])
                            if 'record' in event:
                                state.setdefault('clip_records',{})[Path(event['file']).name]=event['record']
                            _write(state)
                    elif event.get('event')=='result':answer=event
            code=process.wait()
        with LOCK:
            if state.get('cancelled'):state.update(status='stopped',stage='Stopped. Saved files and partial downloads remain.')
            elif code or (answer and not answer.get('ok')):
                state.update(status='error',stage=(answer or {}).get('error',f'Worker exited with code {code}. See the job log.'))
            elif kind in ('download','verify','link'):
                message=(answer or {}).get('details') or 'Model operation complete. Refresh model status.'
                state.update(status='complete',stage=message,result={'details':message})
            elif answer:
                state.update(status='complete',stage='Complete',result=answer)
            else:state.update(status='error',stage='Worker exited without a result. See the job log.')
    except Exception as e:
        with LOCK:state.update(status='stopped' if state.get('cancelled') else 'error',stage=str(e))
    finally:
        with LOCK:
            state.pop('transfer',None)
            state['finished']=time.time();_write(state)
            PROCESSES.pop(state['id'],None)
            if ACTIVE is state:
                ACTIVE=None
                _start_next()

def stop(token):
    with LOCK:
        if not ACTIVE:return 'No active job.'
        if ACTIVE['token']!=token:raise ValueError('This job belongs to another browser workspace.')
        ACTIVE['cancelled']=True
        if ACTIVE['kind']=='download':
            (Path(ACTIVE['job_dir'])/'stop-download').touch()
            return 'Stopping the download and saving its progress…'
        process=PROCESSES.get(ACTIVE['id'])
        if process and process.poll() is None:
            # /T targets only the worker and descendants owned by this job.
            subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
        return 'Stopping the app-owned job…'

def reconnect(token):
    with LOCK:
        if not ACTIVE or ACTIVE['kind']!='download' or ACTIVE.get('cancelled'):
            raise ValueError('There is no active model download to reconnect.')
        if ACTIVE['token']!=token:
            raise ValueError('This download belongs to another browser workspace.')
        transfer=ACTIVE.get('transfer')
        if not transfer:
            raise ValueError('Wait for a model transfer to start.')
        (Path(ACTIVE['job_dir'])/('reconnect-'+transfer['attempt'])).touch()
        return 'Reconnecting and resuming the current model…'

def current(token):
    with LOCK:
        if ACTIVE and ACTIVE['token']==token:return dict(ACTIVE)
        for state in PENDING:
            if state['token']==token:return dict(state)
        saved=load_workspace(token);job=saved.get('job','')
        if not re.fullmatch(r'[0-9]{8}-[0-9]{6}-[a-f0-9]{8}',job):return None
        p=JOBS/job/'status.json'
        state=json.loads(p.read_text()) if p.exists() else None
        if state and state['status']=='running':
            state.update(status='stopped',stage='The previous app session ended. Inputs and logs were preserved.')
        return state

def release():
    if is_busy():raise ValueError('Stop or finish the active job first.')
    return 'GPU memory is released after every job. The interface remains open.'

def shutdown_workers():
    global SHUTTING_DOWN
    with LOCK:
        SHUTTING_DOWN=True
        if ACTIVE:
            try:stop(ACTIVE['token'])
            except Exception:pass
atexit.register(shutdown_workers)
