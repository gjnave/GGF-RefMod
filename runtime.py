"""Shared job lock, process ownership, saved workspaces, and real progress."""
from __future__ import annotations
import atexit, json, os, re, shutil, subprocess, sys, threading, time, uuid
from datetime import datetime
from collections import deque, defaultdict
from pathlib import Path
from config import APP,JOBS,OUTPUTS,LIBRARY,initialize,read_settings
import generation_eta

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
            state['batch_total']=request.get('batch',1)
            if request.get('rerun_of'):
                state['rerun_of']=request['rerun_of']
        write_json(folder/'request.json',request)
        if generation:
            saved.pop('reused_generation',None)
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
            if state['kind']=='generate':
                state['batch_total']=request.get('batch',1)
                completed=len(state.get('result',{}).get('files',[]))
                for index,file in enumerate(state.get('result',{}).get('files',[]),1):
                    expected=OUTPUTS/f"Minimax-Refmod-{state['id']}-{index:02d}.mp4"
                    if Path(file).resolve()!=expected.resolve():
                        raise ValueError('Saved batch output does not belong to this job.')
                request=dict(request,resume_from=completed,completed_files=list(state.get('result',{}).get('files',[])))
                state['clip_index']=completed+1
                if state.get('restart_count'):
                    # Retain the original request and a separate request for each restart.
                    attempt=Path(state['job_dir'])/f"request-restart-{state['restart_count']}.json"
                    write_json(attempt,request)
                    request['worker_request_path']=str(attempt)
                history=[row for rows in GENERATIONS.values() for row in rows]
                history.sort(key=lambda row: row.get('submitted',row.get('started',0)))
                baseline=generation_eta.history(request,history)
                state['timing']=generation_eta.initialize(request,baseline,time.time())
            state.update(status='running',stage='Starting',started=time.time())
            for key in ('finished','progress','cancelled','cancel_reason'):
                state.pop(key,None)
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
        known={row['id'] for rows in GENERATIONS.values() for row in rows}
        for path in JOBS.glob('20*/status.json'):
            try:
                state=json.loads(path.read_text(encoding='utf-8'))
            except (ValueError,OSError) as exc:
                print(f'Cannot read job status {path}: {exc}',file=sys.stderr)
                continue
            if state.get('kind')!='generate':continue
            if state['id'] in known:continue
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
                if state.get('cancelled') and state.get('cancel_reason')!='shutdown':
                    state.update(status='stopped',stage='Stopped by user.',finished=time.time())
                else:
                    state.update(status='queued',stage='Restarting interrupted generation.',
                                 restart_count=state.get('restart_count',0)+1)
                    PENDING.append(state)
                _write(state)
            elif state['status']=='interrupted':
                state.update(status='queued',stage='Restarting interrupted generation.',
                             restart_count=state.get('restart_count',0)+1)
                PENDING.append(state)
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

def delete_refmods(names):
    """Recycle only selected library files, preserving references needed by live jobs."""
    from config import library_path
    from recycle_video import recycle_video as recycle_file
    with LOCK:
        names=list(dict.fromkeys(names or []))
        if not names:raise ValueError('Choose at least one RefMod to delete.')
        paths={name:library_path(name) for name in names}
        needed=set()
        for state in ([ACTIVE] if ACTIVE else [])+list(PENDING):
            request=json.loads((Path(state['job_dir'])/'request.json').read_text(encoding='utf-8'))
            needed.update(Path(path).resolve() for path in request.get('mods',[]))
        blocked=[name for name,path in paths.items() if path in needed]
        if blocked:raise ValueError('These RefMods are used by a running or queued generation. Finish it or remove it from the queue first: '+', '.join(blocked))
        removed=[];errors=[]
        for name,path in paths.items():
            try:recycle_file(path);removed.append(name)
            except OSError as exc:errors.append(name+': '+str(exc))
        return removed,errors


def edit_lineup(token,action,key=None,direction=None):
    with LOCK:
        saved=load_workspace(token)
        lineup=list(saved.get('shot_lineup',[]))
        if action=='add':
            if not workspace_clip_path(token,key).is_file():raise ValueError('This shot is no longer available.')
            if key not in lineup:lineup.append(key)
        elif action=='remove':
            if key in lineup:lineup.remove(key)
        elif action=='move':
            if key not in lineup or direction not in (-1,1):raise ValueError('Choose a shot and direction.')
            index=lineup.index(key);target=index+direction
            if 0<=target<len(lineup):lineup[index],lineup[target]=lineup[target],lineup[index]
        elif action=='clear':
            lineup=[]
            saved.pop('latest_stitch',None)
        elif action=='dismiss-stitch':saved.pop('latest_stitch',None)
        else:raise ValueError('Choose a lineup action.')
        saved['shot_lineup']=lineup
        save_workspace(token,saved)


def lineup_sources(token):
    with LOCK:
        keys=list(load_workspace(token).get('shot_lineup',[]))
        if len(keys)<2:raise ValueError('Add at least two shots to the lineup before stitching.')
        sources=[workspace_clip_path(token,key) for key in keys]
        if any(not p.is_file() for p in sources):raise ValueError('Remove unavailable shots from the lineup first.')
        return keys,sources


def favorite_clip(token,key,favorite):
    """Persist a favorite only for an existing output owned by this workspace."""
    if not isinstance(favorite,bool):raise ValueError('Choose whether to favorite this clip.')
    with LOCK:
        source=workspace_clip_path(token,key)
        if not source.is_file():raise ValueError('This clip is no longer in the gallery.')
        saved=load_workspace(token)
        favorites=set(saved.get('favorite_clips',[]))
        if favorite:favorites.add(key)
        else:favorites.discard(key)
        saved['favorite_clips']=sorted(favorites)
        save_workspace(token,saved)


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

def run_again(token,jobid):
    with LOCK:
        source=next((row for row in GENERATIONS.get(token,[]) if row['id']==jobid),None)
        if source is None:
            raise ValueError('Choose a recent submission from this workspace.')
        if source['status'] in ('running','queued'):
            raise ValueError('This submission is already running or waiting.')
        existing=next((row for row in GENERATIONS[token]
                       if row.get('rerun_of')==jobid and row['status'] in ('running','queued')),None)
        if existing:
            return existing['id'],False
        folder=JOBS/jobid
        if folder.resolve().parent!=JOBS.resolve():
            raise ValueError('The saved submission is outside the app jobs folder.')
        request=json.loads((folder/'request.json').read_text(encoding='utf-8'))
        if request.get('kind')!='generate':
            raise ValueError('This action requires a saved video generation.')
        paths=list(request.get('mods',[]))+[a['path'] for a in request.get('loras',[])]
        paths += [request[k] for k in ('audio','first_frame','last_frame') if request.get(k)]
        missing=[str(path) for path in paths if not Path(path).is_file()]
        if missing:
            raise ValueError('Restore the saved input before running again: '+missing[0])
        for key in ('job_dir','output_name','resume_from','completed_files','worker_request_path'):
            request.pop(key,None)
        request['rerun_of']=jobid
        return start(token,request),True

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
    else:args=[sys.executable,'-u',str(APP/'core_worker.py'),request.get('worker_request_path',str(folder/'request.json'))]
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
                        if 'clip_index' in event:
                            state['clip_index']=event['clip_index']
                        if kind=='generate' and state.get('timing'):
                            generation_eta.stage(state['timing'],event.get('phase'),event.get('clip_index',state.get('clip_index',1)),time.time())
                        state.pop('progress',None)
                        state.pop('transfer',None)
                    elif event.get('event')=='progress':
                        state['progress']=[event['value'],event['total']]
                        if kind=='generate' and state.get('timing'):
                            generation_eta.progress(state['timing'],event['value'],event['total'],time.time())
                    elif event.get('event')=='clip' and kind=='generate':
                        files=state['result'].setdefault('files',[])
                        if event['file'] not in files:
                            files.append(event['file'])
                            if 'record' in event:
                                if state.get('timing'):
                                    generation_eta.clip_finished(state['timing'],event['record'],time.time())
                                state.setdefault('clip_records',{})[Path(event['file']).name]=event['record']
                            _write(state)
                    elif event.get('event')=='result':answer=event
            code=process.wait()
        with LOCK:
            if state.get('cancelled'):
                interrupted=kind=='generate' and state.get('cancel_reason')=='shutdown'
                state.update(status='interrupted' if interrupted else 'stopped',
                             stage='Generation will restart when the app opens.' if interrupted else 'Stopped. Saved files and partial downloads remain.')
            elif code or (answer and not answer.get('ok')):
                state.update(status='error',stage=(answer or {}).get('error',f'Worker exited with code {code}. See the job log.'))
            elif kind in ('download','verify','link'):
                message=(answer or {}).get('details') or 'Model operation complete. Refresh model status.'
                state.update(status='complete',stage=message,result={'details':message})
            elif answer:
                state.update(status='complete',stage='Complete',result=answer)
            else:state.update(status='error',stage='Worker exited without a result. See the job log.')
    except Exception as e:
        with LOCK:
            interrupted=kind=='generate' and state.get('cancel_reason')=='shutdown'
            status='interrupted' if interrupted else ('stopped' if state.get('cancelled') else 'error')
            state.update(status=status,stage=str(e))
    finally:
        with LOCK:
            state.pop('transfer',None)
            state['finished']=time.time();_write(state)
            PROCESSES.pop(state['id'],None)
            if ACTIVE is state:
                ACTIVE=None
                _start_next()

def stop(token,reason='user'):
    with LOCK:
        if not ACTIVE:return 'No active job.'
        if ACTIVE['token']!=token:raise ValueError('This job belongs to another browser workspace.')
        ACTIVE['cancelled']=True
        ACTIVE['cancel_reason']=reason
        _write(ACTIVE)
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

def generation_estimate(token):
    with LOCK:
        state=current(token) if token else None
        estimate=None
        if state and state.get('timing'):
            estimate=generation_eta.estimate(state['timing'],state['status'],state['started'],state.get('finished') or time.time())
        return state,estimate

def release():
    if is_busy():raise ValueError('Stop or finish the active job first.')
    return 'GPU memory is released after every job. The interface remains open.'

def shutdown_workers():
    global SHUTTING_DOWN
    with LOCK:
        SHUTTING_DOWN=True
        if ACTIVE:
            try:stop(ACTIVE['token'],reason='shutdown')
            except Exception:pass
atexit.register(shutdown_workers)
