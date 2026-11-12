from __future__ import annotations
import base64, html, json, os, shutil, socket, threading, time, uuid
from pathlib import Path
import gradio as gr
import config as cfg
import runtime
import video_profile
import video_loras
import pika_profile
import project_history
import generation_eta
import app_updates
import media_previews as previews
from urllib.parse import quote
from functools import lru_cache

CSS=(cfg.APP/'assets/theme.css').read_text(encoding='utf-8')+'''
.gradio-container {font-family: 'Segoe UI',Arial,sans-serif!important; max-width:1400px!important; padding:clamp(10px,2vw,24px)!important}
.gradio-container :is(textarea,input:not([type]),input[type="text"],input[type="number"],input[type="password"],select) {font-size:16px}
.ggf-hero-logo {width:96px;height:96px}.ggf-hero {grid-template-columns:1fr 280px 96px;padding:20px 24px}
#activity {border:1px solid #344b6e;background:#111c2f;border-radius:10px;padding:12px 16px;min-height:52px}
.main-action {min-height:54px!important;font-size:18px!important;font-weight:750!important}
.status-note {color:#9db0cc;font-size:14px}.ggf-footer {color:#9db0cc;text-align:center;padding:16px}
.model-grid {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,280px),1fr));gap:10px}
.model-card {background:#111c2f;border:1px solid #2a3b56;border-radius:10px;padding:14px;min-width:0}
.model-card strong {color:#e8eef9}.model-card p {color:#9db0cc;margin:5px 0;overflow-wrap:anywhere;font-size:12px}
.model-card .missing {color:#f5b942}.model-card .installed {color:#80dca5}
.upload-previews {display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,210px),1fr));gap:10px}
.upload-previews figure {margin:0;padding:8px;border:1px solid #344b6e;border-radius:8px;background:#091425}
.upload-previews img,.upload-previews video {width:100%;height:clamp(180px,26vw,300px);object-fit:contain;background:#07101f;display:block}
.upload-previews audio {width:100%;display:block}.upload-previews figcaption {color:#e8eef9;overflow-wrap:anywhere;margin-top:6px}
button:disabled {background:#25344a!important;color:#9db0cc!important}
.gradio-container button:not([role="tab"]):hover {background:#274266!important;color:#e8eef9!important}
.gradio-container button.primary:hover {background:#ffd16e!important;color:#091425!important}
.live-dot {display:inline-block;width:10px;height:10px;border-radius:50%;background:#f5b942;margin-right:10px;animation:ggf-pulse 1s infinite alternate}
@keyframes ggf-pulse {to{opacity:.25}}
@media(max-width:680px){.ggf-hero{grid-template-columns:1fr 70px}.ggf-hero-logo{width:70px;height:70px}.ggf-hero-slogan{display:none}}
'''
RESTORE_JS='''() => { let key=localStorage.getItem('minimax-refmod-workspace'); if(!/^[a-f0-9]{32}$/.test(key||'')){key=crypto.randomUUID().replaceAll('-','');localStorage.setItem('minimax-refmod-workspace',key);} if(!window.__minimaxKeyHandler){window.__minimaxKeyHandler=true; document.addEventListener('keydown', e=>{if(e.ctrlKey && e.key==='Enter' && !e.repeat){const b=[...document.querySelectorAll('.main-action button,button.main-action')].find(x=>x.offsetParent!==null&&!x.disabled);if(b){e.preventDefault();b.click();}}});} return [key];}'''
SERVERS=[]

def theme():
    values={}
    colors={'body_background_fill':'#07101f','body_text_color':'#e8eef9','block_background_fill':'#111c2f','block_border_color':'#2a3b56','block_label_background_fill':'#111c2f','block_label_text_color':'#e8eef9','block_title_text_color':'#e8eef9','input_background_fill':'#091425','input_border_color':'#3a4f70','input_placeholder_color':'#9db0cc','button_primary_background_fill':'#f5b942','button_primary_text_color':'#091425','button_secondary_background_fill':'#192d47','button_secondary_text_color':'#e8eef9'}
    for k,v in colors.items():values[k]=v;values[k+'_dark']=v
    return gr.themes.Base().set(**values)

def model_table():
    names={'base':'Video model','encoder':'Text encoder','video_vae':'Video reference encoder','audio_vae':'Audio reference encoder','turbo':'8-step Turbo','fast_preview':'Fast Preview (optional)','upscaler':'Latent upscaler','refiner':'Video refiner','dmd':'Refinement acceleration','pika_fl2va':'Pika text / first-last adapter (optional)','pika_ref2va':'Pika references adapter (optional)','pika_base':'Pika references checkpoint (optional)'}
    rows=[]
    for m in displayed_models():
        cls='missing' if m['status']=='Missing or incomplete' else 'installed'
        rows.append(f'<div class="model-card"><strong>{names[m["model"]]}</strong><p class="{cls}">{m["status"]} · {m["GB"]} GB</p><p>{html.escape(m["file"])}</p></div>')
    return '<div class="model-grid">'+''.join(rows)+'</div>'

def displayed_models():
    rows=cfg.model_status()
    profile=video_profile.settings()
    if profile['active']==video_profile.HYBRID:
        try:
            video_profile.linked_path(profile)
            status='Linked · GrizzlyMax INT4 Hybrid'
        except ValueError:
            status='Missing or incomplete'
        base=next(row for row in rows if row['model']=='base')
        base.update(file=profile['path'],status=status,GB=round(video_profile.SIZE/1e9,2))
    return rows

def library():
    return sorted((str(p.relative_to(cfg.LIBRARY)).replace('\\','/') for p in cfg.LIBRARY.rglob('*.safetensors')),key=lambda name:(name.casefold(),name))

def metadata(name):
    from safetensors import safe_open
    path=cfg.library_path(name)
    with safe_open(str(path),framework='numpy') as f:
        raw=f.metadata() or {};m=raw.get('refmod_meta',raw.get('audio_refmod_meta'))
        if not m:raise ValueError('This file has no RefMod metadata.')
        return json.loads(m)

def reference_labels(names,strength=1):
    counts={'image':0,'video':0,'audio':0};labels={'image':'Picture','video':'Video','audio':'Audio'};rows=[]
    buttons=[]
    if names and strength==0:
        return 'No active references at strength 0.',gr.update(value='',visible=False)
    for name in names or []:
        try:
            m=metadata(name)
            members=m.get('refs',m.get('members',[])) if m.get('kind')=='bundle' else [m]
            for ref in members:
                kind=ref.get('kind','image')
                if kind in counts:
                    counts[kind]+=1
                    label=f'{labels[kind]} {counts[kind]}'
                    ref_name=ref.get('name',name)
                    rows.append(f"<{label}> = {ref_name}")
                    prompt_name=Path(str(ref_name)).name.removesuffix('.safetensors') if m.get('kind')=='bundle' else Path(name).stem
                    escaped_name=html.escape(prompt_name,quote=True)
                    buttons.append(f'<button type="button" data-ref="{escaped_name}" data-kind="{kind}" title="{html.escape(label+": "+prompt_name,quote=True)}">{escaped_name}</button>')
        except Exception as e:rows.append(str(e))
    shortcuts='<span>Insert reference</span> '+ ' '.join(buttons) if buttons else ''
    return '\n'.join(rows) or 'Select RefMods to see their prompt labels.',gr.update(value=shortcuts,visible=bool(buttons))

def reference_selection(names):
    available=library()
    selected=[name for name in names or [] if name in available]
    return gr.update(choices=available,value=selected,interactive=True)


def guarded(fn):
    def run(*args,**kw):
        try:return fn(*args,**kw)
        except Exception as e:raise gr.Error(str(e)) from e
    return run

def reorder_refmods(token,names,event:gr.EventData):
    names=list(names or [])
    name=getattr(event,'name',None)
    direction=getattr(event,'direction',None)
    target_name=getattr(event,'target_name',None)
    if name not in names:
        raise gr.Error('Select a RefMod before reordering it.')
    if target_name in names and target_name!=name:
        names.remove(name)
        target=names.index(target_name)+(1 if getattr(event,'after',False) else 0)
        names.insert(target,name)
    elif direction in (-1,1):
        index=names.index(name)
        target=index+direction
        if 0<=target<len(names):
            names[index],names[target]=names[target],names[index]
    else:
        return gr.skip()
    with runtime.LOCK:
        saved=runtime.load_workspace(token)
        saved.setdefault('controls',{})['selected']=names
        runtime.save_workspace(token,saved)
    return reference_selection(names)

def reference_action(token,names,event:gr.EventData):
    if getattr(event,'action',None)=='replace':
        name=getattr(event,'name',None)
        if name not in (names or []):
            raise gr.Error('Select a RefMod first.')
        return gr.update(interactive=False),gr.update(visible=True),name,gr.update(choices=library(),value=name)
    return reorder_refmods(token,names,event),gr.skip(),gr.skip(),gr.skip()

@guarded
def replace_refmod(token,names,target,replacement):
    names=list(names or [])
    if target not in names or replacement not in library():
        raise ValueError('Select an available RefMod to replace this reference.')
    index=names.index(target)
    if replacement in names:
        other=names.index(replacement)
        names[index],names[other]=names[other],names[index]
    else:
        names[index]=replacement
    with runtime.LOCK:
        saved=runtime.load_workspace(token)
        saved.setdefault('controls',{})['selected']=names
        runtime.save_workspace(token,saved)
    return reference_selection(names),gr.update(visible=False),''

@guarded
def create(token,media,audio,name,mode,concept,description,resolution,pool,latent_frames,refinement,max_tokens,max_frames,audio_seconds,audio_concept,layout):
    if not media and not audio:raise ValueError('Add reference images/videos or an audio reference.')
    if mode not in ('Full Reference','Compressed Reference'):raise ValueError('Choose a reference mode.')
    if concept not in ('identity','clothing','style','pose_motion','background','generic'):raise ValueError('Choose a concept type.')
    cfg.require_models((['video_vae'] if media else [])+(['audio_vae'] if audio else []))
    req=dict(kind='create',media=media or [],audio=audio,name=cfg.safe_name(name),mode=mode,concept=concept,description=description,
        resolution=cfg.numeric(resolution,256,2048,'Reference resolution',True),pool=cfg.numeric(pool,4,64,'Pooling grid',True),
        latent_frames=cfg.numeric(latent_frames,1,256,'Latent frames',True),refinement=cfg.numeric(refinement,0,2000,'Refinement steps',True),
        max_tokens=cfg.numeric(max_tokens,0,1048576,'Token budget',True),max_frames=cfg.numeric(max_frames,1,10000,'Input frames',True),
        audio_seconds=cfg.numeric(audio_seconds,.1,600,'Audio seconds'),audio_concept=audio_concept,layout=layout)
    runtime.start(token,req)
    return 'Creating RefMod…',[],''

@guarded
def generate(token,prompt,selected,strength,retention,aspect,mp,duration,seed,preset,steps,upscale,upmp,refinesteps,first,last,batch,lora1='',weight1=1,lora2='',weight2=1,model='Use Settings'):
    if not prompt.strip():raise ValueError('Describe the scene, motion, and sound.')
    chosen_preset=preset
    preset=pika_profile.resolve(preset,selected if float(strength)>0 else [])
    w,h=cfg.canvas(aspect,mp);uw,uh=cfg.canvas(aspect,upmp)
    frames=cfg.frames_for(duration)
    if upscale and uw*uh<=w*h:raise ValueError('Upscale megapixels must be larger than the working canvas.')
    if len(selected or [])>8:raise ValueError('Select up to eight RefMods.')
    paths=[str(cfg.library_path(n)) for n in selected or []]
    strength=cfg.numeric(strength,0,1,'Reference strength');retention=cfg.numeric(retention,0,1,'Retention')
    profile=video_profile.settings()
    if pika_profile.enabled(preset) and model=='Use Settings':
        profile=dict(profile,active=video_profile.INT8)
    if model!='Use Settings':
        if model not in (video_profile.INT8,video_profile.HYBRID):raise ValueError('Choose a video model.')
        profile=dict(profile,active=model)
    hybrid=profile['active']==video_profile.HYBRID
    if hybrid:
        video_profile.linked_path(profile)
    adapter,nsteps,_,_=cfg.sampling_settings(preset,steps)
    req=dict(kind='generate',video_profile=profile,prompt=prompt,mods=paths,strengths=[strength]*len(paths),retention=retention,width=w,height=h,frames=frames,
        seed=cfg.seed_value(seed),preset=preset,steps=nsteps,upscale=bool(upscale),output_frames=max(1,round(float(duration)*24)),
        up_width=uw,up_height=uh,refine_steps=cfg.numeric(refinesteps,1,8,'Refinement steps',True),first_frame=first,last_frame=last,
        batch=cfg.numeric(batch,1,8,'Batch count',True))
    req['loras']=video_loras.selection([(lora1,weight1),(lora2,weight2)])
    pika_profile.validate(req)
    base_key=pika_profile.model_key(preset)
    cfg.require_models(([] if hybrid else [base_key])+['encoder','video_vae','audio_vae']+([adapter] if adapter else [])+(['upscaler','dmd','refiner'] if upscale else []))
    controls=dict(prompt=prompt,selected=selected or [],aspect=aspect,mp=mp,duration=duration,seed_input=cfg.seed_input(seed),preset=chosen_preset,custom_steps=steps,upscale=upscale,
                  strength=strength,upmp=upmp,model=model,loras=[(lora1,weight1),(lora2,weight2)])
    req['form']=controls
    runtime.start(token,req,controls)
    return 'Video submitted. See Generation queue.',gr.skip(),gr.skip()

def refresh_library(selected):
    return reference_selection(selected)


@guarded
def manage_refmods(names,confirmed,selected,remote):
    if remote:raise ValueError('Delete RefMods from the PC interface.')
    if not confirmed:raise ValueError('Confirm that the selected RefMods should move to the Windows Recycle Bin.')
    removed,errors=runtime.delete_refmods(names)
    available=library()
    message=f'Moved {len(removed)} RefMod(s) to the Windows Recycle Bin. Restore them there if needed.' if removed else 'No RefMods were moved.'
    if errors:message+='\n\nCould not move:\n\n'+'\n\n'.join(errors)
    return (gr.update(choices=available,value=[name for name in names or [] if name in available]),
            reference_selection([name for name in selected or [] if name not in removed]),False,message)


def lora_menus(first,second):
    choices=video_loras.choices()
    available={value for _,value in choices}
    return [gr.update(choices=choices,value=value if value in available else '') for value in (first,second)]

@guarded
def link_lora(location,first,second,remote):
    if remote:
        raise ValueError('Link local files from the PC interface.')
    path=video_loras.link(location or '')
    if path not in (first,second):
        if not first: first=path
        elif not second: second=path
    return *lora_menus(first,second),f'Linked {Path(path).name}.'

def describe_imports(files):
    from safetensors import safe_open
    rows=[]
    for file in files or []:
        try:
            with safe_open(str(file),framework='numpy') as f:
                data=f.metadata() or {}
                meta=json.loads(data.get('refmod_meta') or data['audio_refmod_meta'])
                kind=meta.get('kind') or ('audio' if 'audio_refmod_meta' in data else 'image')
                rows.append(f"{Path(file).name}: {kind} · {meta.get('name') or 'Unnamed reference'}")
        except Exception:
            rows.append(f'{Path(file).name}: could not read RefMod metadata.')
    return gr.update(value='\n'.join(rows),visible=bool(rows))

@guarded
def import_mod(files):
    if runtime.is_busy():raise ValueError('Wait for the active job before importing RefMods.')
    from safetensors import safe_open
    count=0
    for file in files or []:
        with safe_open(str(file),framework='numpy') as f:
            data=f.metadata() or {}
            if 'refmod_meta' not in data and 'audio_refmod_meta' not in data:raise ValueError(Path(file).name+' is not a RefMod file.')
        target=cfg.LIBRARY/Path(file).name
        if target.exists():target=cfg.LIBRARY/(target.stem+'_'+uuid.uuid4().hex[:8]+target.suffix)
        shutil.copy2(file,target);count+=1
    return f'Imported {count} RefMods. Refresh the library to select them.'

@guarded
def maintenance(token,action,profile,remote):
    if remote:raise ValueError('Manage model files from the local PC interface.')
    if profile==cfg.FAST_DOWNLOAD:
        keys=['fast_preview']
    elif profile==pika_profile.DEFAULT_DOWNLOAD:
        keys=['base','pika_base','encoder','video_vae','audio_vae','pika_fl2va','pika_ref2va']
    elif profile==pika_profile.DOWNLOAD:
        keys=['base','encoder','video_vae','audio_vae','pika_fl2va']
    elif profile==pika_profile.REFERENCE_DOWNLOAD:
        keys=['pika_base','encoder','video_vae','audio_vae','pika_ref2va']
    elif profile=='RefMod creator — 5.8 GB':
        keys=['video_vae','audio_vae']
    else:
        keys=[key for key in cfg.FILES if key not in ('fast_preview','pika_fl2va','pika_ref2va','pika_base')]
    video=video_profile.settings()
    use_hybrid=video['active']==video_profile.HYBRID and 'base' in keys and profile not in (pika_profile.DOWNLOAD,pika_profile.DEFAULT_DOWNLOAD) and action in ('download','verify')
    if use_hybrid:
        video_profile.linked_path(video)
        keys.remove('base')
    runtime.start(token,dict(kind=action,keys=keys,verify_hybrid=video['path'] if use_hybrid else None))
    return 'Model operation started. Progress appears above.'

@guarded
def start_model_link(token,location,family,remote,pick):
    if remote:
        raise ValueError('Change model settings on the local PC.')
    if pick:
        location=browse_models(remote,family)
        if not isinstance(location,str):return gr.skip(),gr.skip(),gr.skip()
    location=str(cfg.existing_model_location(location))
    job=runtime.start(token,dict(kind='link',family=family,folder=location))
    return location, f'Checking {location}. Large checkpoints can take several minutes to verify.',job

def link_progress(token,job):
    if not job:return [gr.skip()]*4
    path=cfg.JOBS/job/'status.json'
    state=json.loads(path.read_text(encoding='utf-8'))
    if state['token']!=token or state['status'] in ('queued','running'):return [gr.skip()]*4
    return state['stage'],'',video_profile.active_label(),model_table()

@guarded
def browse_models(remote,kind):
    if remote:
        raise ValueError('Browse for model folders on the local PC.')
    import subprocess,sys
    dialogs={
        'lora':'filedialog.askopenfilename(parent=root,title="Choose a MiniMax H3 LoRA",filetypes=[("LoRA weights","*.safetensors")])',
        'grizzly':'filedialog.askdirectory(parent=root,title="Select GrizzlyMax models folder")',
        'standard':'filedialog.askdirectory(parent=root,title="Select existing MiniMax models folder")',
    }
    dialog=dialogs[kind]
    picker=('import tkinter as tk; from tkinter import filedialog; '
            'root=tk.Tk(); root.withdraw(); root.attributes("-topmost",True); '
            f'print({dialog}); root.destroy()')
    picked=subprocess.run([sys.executable,'-X','utf8','-c',picker],capture_output=True,text=True,
                          encoding='utf-8',check=True,creationflags=subprocess.CREATE_NO_WINDOW)
    return picked.stdout.strip() or gr.skip()

def download_panel(state):
    if not state or state['kind']!='download':
        return ''
    transfer=state.get('transfer')
    if not transfer:
        return html.escape(state['stage'])
    done,total,speed=transfer['received'],transfer['total'],transfer['speed']
    if speed:
        minutes,seconds=divmod(round(max(0,total-done)/speed),60)
        eta=f'{minutes}m {seconds:02d}s'
    else:
        eta='estimating'
    name=html.escape(transfer['name'])
    return (f'<div><strong>{name}</strong><br><progress style="width:100%" value="{done}" max="{total}" aria-label="Model download"></progress>'
            f'<div>{done/1e9:.2f} / {total/1e9:.2f} GB ({100*done/total:.1f}%) · '
            f'{speed/1e6:.1f} MB/s · ETA {eta}</div></div>')

@guarded
def resume_connection(token,remote):
    if remote:
        raise ValueError('Manage model downloads from the local PC interface.')
    gr.Info(runtime.reconnect(token))

def queue_view(token,previous):
    rows=runtime.generation_list(token)
    live=[];recent=[]
    for row in rows:
        title=html.escape(row['label'])
        label=row['status'].capitalize()
        if row['position']:
            label=f"Waiting #{row['position']}"
        links=[]
        for i,file in enumerate(row.get('result',{}).get('files',[]),1):
            if not Path(file).is_file():continue
            href='./gradio_api/file='+quote(Path(file).as_posix(),safe='/:')
            links.append(f'<a href="{href}" target="_blank" rel="noopener">Open video {i}</a>')
        info=html.escape(row['stage']) if row['status']=='error' else ''
        button=(f'<button type="button" data-remove="{row["id"]}" aria-label="Remove {title} from queue">Remove</button>' if row['position'] else '')
        section=live if row['status'] in ('queued','running') else recent
        if section is recent:
            button=f'<button type="button" class="queue-rerun" data-rerun="{row["id"]}" aria-label="Run again: {title}">Run again</button>'
        section.append(f'<li><strong>{label}</strong><span>{title}</span>{button} {" ".join(links)} {info}</li>')
    panel=('<ul>'+''.join(live)+'</ul>' if live else '<p>Queue is empty.</p>')
    if recent:panel+='<details><summary>Recent submissions</summary><ul>'+''.join(recent)+'</ul></details>'
    return (panel,panel) if panel!=previous else (gr.skip(),gr.skip())

def queue_action(token,event:gr.EventData):
    try:
        if event.action=='rerun':
            _,created=runtime.run_again(token,event.job)
            gr.Info('Submitted again with the saved prompt, seed and settings.' if created else 'This retry is already running or waiting.')
        elif event.action=='remove':
            gr.Info(runtime.cancel_waiting(token,event.job))
        else:
            raise ValueError('Choose Run again or Remove.')
    except (OSError,ValueError) as e:gr.Warning(str(e))
    return queue_view(token,None)

@guarded
def inspect_project(token,key):
    return gr.update(visible=True),project_history.details(project_history.project(token,key)),key

@guarded
def reuse_project(token,key):
    p=project_history.project(token,key)
    choices=library(); adapters=video_loras.choices()
    v=project_history.reuse_values(p,choices,{path for _,path in adapters})
    slots=[value for path,weight in v['slots'] for value in (gr.update(choices=adapters,value=path),weight)]
    with runtime.LOCK:
        saved=runtime.load_workspace(token)
        saved['controls']=dict(prompt=v['prompt'],selected=v['selected'],strength=v['strength'],retention=v['retention'],
                               aspect=v['aspect'],mp=v['mp'],duration=v['duration'],seed_input=v['seed'],preset=v['preset'],
                               custom_steps=v['steps'],upscale=v['upscale'],upmp=v['upmp'],model=v['model'],loras=v['slots'])
        saved['reused_generation']=dict(first_frame=v['frames'][0],last_frame=v['frames'][1],retention=v['retention'],
                                       refine_steps=v['refinesteps'],batch=v['batch'])
        runtime.save_workspace(token,saved)
    gr.Info('Prompt, references and generation settings restored. Review them, then click Generate Video.')
    return (v['prompt'],reference_selection(v['selected']),v['strength'],v['retention'],v['aspect'],v['mp'],v['duration'],
            v['seed'],v['preset'],gr.update(value=v['steps'],visible=v['preset']=='custom'),v['upscale'],v['upmp'],v['refinesteps'],
            *v['frames'],v['batch'],*slots,v['model'])

def media_url(path):
    return './gradio_api/file='+quote(Path(path).as_posix(),safe='/:')

@lru_cache(maxsize=4096)
def clip_poster(video,job_dir,index):
    import av
    poster=job_dir/f'gallery-{index}.jpg'
    if poster.is_file():return media_url(poster)
    try:
        with av.open(video) as media:
            frame=next(media.decode(video=0)).to_image()
        frame.thumbnail((192,192))
        temporary=poster.with_name(f'gallery-{uuid.uuid4().hex}.tmp')
        frame.convert('RGB').save(temporary,format='JPEG',quality=83)
        os.replace(temporary,poster)
        return media_url(poster)
    except (OSError,ValueError,StopIteration):
        return ''

def clip_gallery(token,previous,pinned=''):
    clips=[]
    for clip in runtime.workspace_clips(token):
        job,index,_,file=clip
        if Path(file).is_file():clips.append(clip)
    destination='bin' if cfg.read_settings()['recycle_videos'] is True else 'folder'
    favorites=set(runtime.load_workspace(token).get('favorite_clips',[])) if token else set()
    lineup=runtime.load_workspace(token).get('shot_lineup',[]) if token else []
    keys=[destination,[(job,index,f'{job}-{index}' in favorites) for job,index,_,_ in clips],lineup]
    if keys==previous:return gr.skip(),gr.skip()
    buttons=[]
    for job,index,label,file in clips:
        title=html.escape(f'{label} · Clip {index}',quote=True)
        src=html.escape(media_url(file),quote=True)
        poster=html.escape(clip_poster(file,cfg.JOBS/job,index),quote=True)
        picture=f'<img loading="lazy" src="{poster}" alt="">' if poster else '▶'
        favorite=f'{job}-{index}' in favorites
        favorite_label=('Remove favorite: ' if favorite else 'Favorite: ')+title
        key=f'{job}-{index}';position=lineup.index(key)+1 if key in lineup else 0
        buttons.append(f'<span class="clip-item" hidden><button type="button" class="clip-tile" data-destination="{destination}" data-key="{key}" data-lineup="{position}" data-favorite="{str(favorite).lower()}" data-src="{src}" title="{title}" aria-label="Play {title}">{picture}</button><button type="button" class="clip-favorite" aria-label="{favorite_label}" aria-pressed="{str(favorite).lower()}" title="{favorite_label}">{"★" if favorite else "☆"}</button><button type="button" class="clip-remove" aria-label="Delete {title}" title="Delete video">×</button><label class="clip-select" title="Add or remove shot from lineup"><input type="checkbox" aria-label="Include {title} in lineup" {"checked" if position else ""}><span>{position or "Select"}</span></label></span>')
    return '<div class="clip-grid">'+''.join(buttons)+'</div><p class="clip-empty" hidden>No favorite videos yet. Use ☆ on an output to add it here.</p>',keys

def gallery_action(token,event:gr.EventData):
    try:
        if getattr(event,'action',None)=='favorite':
            runtime.favorite_clip(token,event.key,getattr(event,'favorite',None))
        elif getattr(event,'action',None) in ('lineup-add','lineup-remove'):
            runtime.edit_lineup(token,event.action.removeprefix('lineup-'),event.key)
        else:
            gr.Info(runtime.delete_clip(token,event.key,getattr(event,'destination',None)))
    except (OSError,ValueError) as exc:
        gr.Warning(f'Could not update the clip: {exc}')
    return clip_gallery(token,None,getattr(event,'pinned',''))

def lineup_view(token,previous):
    if not token:return gr.skip(),gr.skip()
    saved=runtime.load_workspace(token)
    keys=saved.get('shot_lineup',[])
    clips={f'{job}-{index}':(job,index,label,Path(file)) for job,index,label,file in runtime.workspace_clips(token)}
    available={key for key,(_,_,_,file) in clips.items() if file.is_file()}
    marker=[keys,sorted(available),bool(saved.get('latest_stitch'))]
    if marker==previous:return gr.skip(),gr.skip()
    rows=[];missing=False
    for position,key in enumerate(keys):
        clip=clips.get(key) if key in available else None
        title=html.escape(f'{clip[2]} · Clip {clip[1]}' if clip else 'Unavailable shot')
        poster=clip_poster(clip[3],cfg.JOBS/clip[0],clip[1]) if clip else ''
        picture=f'<img src="{html.escape(poster,quote=True)}" alt="">' if poster else ''
        missing=missing or not clip
        rows.append(f'<li data-key="{html.escape(key,quote=True)}">{picture}<span>{position+1}. {title}</span><div class="lineup-row-actions"><button type="button" data-action="move" data-direction="-1" aria-label="Move shot {position+1} up" {"disabled" if position==0 else ""}>↑</button><button type="button" data-action="move" data-direction="1" aria-label="Move shot {position+1} down" {"disabled" if position==len(keys)-1 else ""}>↓</button><button type="button" data-action="remove" aria-label="Remove shot {position+1} from lineup">Remove</button></div></li>')
    content='<ol>'+''.join(rows)+'</ol>' if rows else '<p>Select shots using the checkboxes on their thumbnails, or add the active video.</p>'
    if missing:content+='<p>Remove unavailable shots before playing or stitching.</p>'
    content+=f'<div class="lineup-actions"><button type="button" data-play-lineup {"disabled" if not keys or missing else ""}>Play lineup</button><button type="button" data-stop-lineup disabled>Stop lineup</button><button type="button" data-action="clear" {"disabled" if not keys and not saved.get("latest_stitch") else ""}>Clear lineup &amp; preview</button><span class="lineup-play-status" aria-live="polite"></span></div>'
    return content,marker


def lineup_action(token,event:gr.EventData):
    preview=gr.skip()
    try:
        runtime.edit_lineup(token,event.action,getattr(event,'key',None),getattr(event,'direction',None))
        if event.action=='clear':preview=''
    except (ValueError,OSError) as exc:gr.Warning(str(exc))
    return *lineup_view(token,None),preview


def clear_stitched_preview(token):
    runtime.edit_lineup(token,'dismiss-stitch')
    return '',*lineup_view(token,None)


def stitched_result(token):
    if not token:return ''
    saved=runtime.load_workspace(token).get('latest_stitch')
    if not saved:return ''
    path=Path(saved['path']);folder=(cfg.OUTPUTS/'Stitched videos').resolve()
    if path.resolve().parent!=folder or not path.is_file():return ''
    address=html.escape(media_url(path),quote=True)
    return f'<p>{saved["shots"]} shots · {saved["seconds"]:.1f} seconds</p><video controls playsinline preload="metadata" src="{address}" aria-label="Stitched lineup preview"></video><p><a href="{address}" download="{html.escape(path.name,quote=True)}">Download stitched MP4</a></p><button type="button" class="clear-stitched">Clear stitched preview</button>'


def stitch_lineup(token,progress=gr.Progress()):
    import av,shot_lineup
    try:
        keys,sources=runtime.lineup_sources(token)
        path,details=shot_lineup.stitch(sources,cfg.OUTPUTS/'Stitched videos',keys,lambda f,msg:progress(f,desc=msg))
        with runtime.LOCK:
            saved=runtime.load_workspace(token)
            saved['latest_stitch']=dict(path=str(path),shots=len(keys),seconds=details['duration_seconds'])
            runtime.save_workspace(token,saved)
        return stitched_result(token)
    except (OSError,ValueError,av.error.FFmpegError) as exc:raise gr.Error(f'Could not stitch the lineup: {exc}') from exc


def convert_to_gif(token,key,caption='',font='Arial'):
    import gif_export
    try:
        path=gif_export.export(runtime.workspace_clip_path(token,key),loop=True,caption=caption,font=font)
    except (OSError,ValueError,IndexError) as exc:
        raise gr.Error(f'GIF export failed: {exc}') from exc
    url=html.escape(media_url(path),quote=True)
    name=html.escape(path.name,quote=True)
    return f'<a href="{url}" download="{name}">Download GIF</a> · {name}<details><summary>Preview GIF</summary><img src="{url}" alt="Exported animated GIF" style="max-width:100%"></details>'

def generation_progress(token,previous):
    state,estimate=runtime.generation_estimate(token)
    content=''
    if state and state['kind']=='generate':
        status=state['status']
        stage=html.escape(state['stage'])
        batch=state.get('batch_total',1)
        clip=state.get('clip_index',len(state.get('result',{}).get('files',[]))+1)
        title=f'Video {min(clip,batch)} of {batch}' if batch>1 else 'Video generation'
        value=''
        detail=''
        progress=state.get('progress')
        if status=='complete':
            value=' value="100"'
            detail='100% complete'
        elif status=='running':
            percent=estimate['percent'] if estimate else 0
            value=f' value="{percent:g}"'
            remaining=estimate['remaining'] if estimate else None
            left='Estimating time left…' if remaining is None else f'About {generation_eta.duration(remaining)} left'
            detail=f'{percent:.0f}% overall · {left}'
        if status in ('running','complete'):
            elapsed=estimate['elapsed'] if estimate else (state.get('finished') or time.time())-state['started']
            detail+=f' · Elapsed {project_history.clock(elapsed)}'
        if status=='running' and progress and progress[1]>0:
            detail+=f' · Stage {progress[0]:g} / {progress[1]:g}'
        if status in ('running','complete'):
            bar=f'<progress max="100"{value} aria-label="Video generation progress"></progress>'
        else:
            bar=''
        content=f'<div class="generation-progress"><strong>{title}</strong><div>{stage}</div>{bar}<span>{detail}</span></div>'
    if content==previous:return gr.skip(),gr.skip()
    return gr.update(value=content,visible=bool(content)),content

def poll(token,last_seen):
    if not token:return [gr.skip()]*10
    state=runtime.current(token);busy=runtime.is_busy()
    create_btn=gr.update(interactive=not busy,value='Working…' if busy else 'Create RefMod')
    generate_btn=gr.update(interactive=True,value='Add to queue' if busy else 'Generate Video')
    transfer_ui=[download_panel(state),gr.update(interactive=bool(state and state.get('transfer') and not state.get('cancelled')))]
    if not state:return ['Ready. Add references or open Settings to set up models.',create_btn,generate_btn,*([gr.skip()]*5),*transfer_ui]
    elapsed=(state.get('finished') or time.time())-state['started']
    progress=state.get('progress');step=f' · {progress[0]}/{progress[1]}' if progress else ''
    dot='<span class="live-dot"></span>' if state['status']=='running' else ''
    status=f"{dot}<strong>{html.escape(state['stage'])}</strong>{step} · {elapsed:.0f}s"
    r=state['result']
    files=[file for file in r.get('files',[]) if Path(file).is_file()]
    marker=state['id']+state['status']+str(len(files))
    if marker==last_seen or state['status'] in ('running','queued'):return [status,create_btn,generate_btn,*([gr.skip()]*5),*transfer_ui]
    log=Path(state['job_dir'])/'worker.log'
    return [status,create_btn,generate_btn,files,r.get('details',''),str(log) if log.is_file() else None,marker,model_table(),*transfer_ui]

def restore(token):
    saved=runtime.load_workspace(token);c=saved.get('controls',{})
    if not saved.get('pika_default_0111'):
        if not saved.get('reused_generation'):
            saved['controls_before_pika_default']=dict(c)
            c=dict(c,model='Use Settings',preset=c.get('preset') if pika_profile.enabled(c.get('preset')) else cfg.DEFAULT_PRESET)
            saved['controls']=c
        saved['pika_default_0111']=True
        runtime.save_workspace(token,saved)
    request={}
    job=saved.get('job','')
    if job and Path(job).name==job:
        request_path=cfg.JOBS/job/'request.json'
        if request_path.is_file():request=json.loads(request_path.read_text(encoding='utf-8'))
    creator=request if request.get('kind')=='create' else {}
    frames=saved.get('reused_generation') or (request if request.get('kind')=='generate' else {})
    media=[p for p in creator.get('media',[]) if Path(p).is_file()]
    slots=c.get('loras',[('',1),('',1)])
    menus=lora_menus(slots[0][0],slots[1][0])
    preset=cfg.saved_preset(c)
    # Older workspaces stored the generated number, so only restore an explicit input.
    return (token,c.get('prompt',''),reference_selection(c.get('selected',[])),c.get('aspect','Portrait 9:16'),
            c.get('mp',.6),c.get('duration',5),c.get('seed_input',''),preset,c.get('upscale',False),media or None,creator.get('audio'),creator.get('name',''),
            creator.get('mode','Full Reference'),creator.get('concept','identity'),creator.get('description',''),frames.get('first_frame'),frames.get('last_frame'),
            menus[0],slots[0][1],menus[1],slots[1][1],gr.update(value=c.get('custom_steps',20),visible=preset=='custom'),
            c.get('model','Use Settings'),c.get('strength',1),frames.get('retention',1),c.get('upmp',1.2),frames.get('refine_steps',2),frames.get('batch',1))

def save_network(enabled,username,password):
    if runtime.is_busy():raise gr.Error('Finish or stop the active job first.')
    if password and not username.strip():raise gr.Error('Enter a username when using a password.')
    cfg.update_settings(phone_enabled=enabled,username=username.strip(),password=password)
    return 'Saved. Restart with RUN.bat to apply phone access settings.'

def check_app_update(remote=False):
    if remote:raise gr.Error('Update the app from this PC.')
    try:
        manifest,available=app_updates.check()
        return f"Installed {cfg.VERSION}. Latest release {manifest['version']}." + (' Click Update app to install it.' if available else ' You are up to date.')
    except Exception as error:
        raise gr.Error(str(error)) from error

def update_app(remote=False,progress=gr.Progress()):
    if remote:raise gr.Error('Update the app from this PC.')
    try:
        with runtime.LOCK:
            if runtime.is_busy() or runtime.PENDING:
                raise ValueError('Finish or stop the active job and clear the waiting queue before updating.')
        progress(0,desc='Downloading and verifying the update')
        folder=app_updates.prepare()
        if folder is None:return 'You are up to date.'
        with runtime.LOCK:
            if runtime.is_busy() or runtime.PENDING:
                raise ValueError('A job started while the update downloaded. Finish it, then click Update app again.')
            runtime.SHUTTING_DOWN=True
            try:
                app_updates.launch_prepared(folder,wait_pid=os.getpid(),relaunch=True)
            except Exception:
                runtime.SHUTTING_DOWN=False
                raise
        def close_for_update():
            time.sleep(2)
            for server in SERVERS:server.close()
            os._exit(0)
        threading.Thread(target=close_for_update,daemon=True).start()
        return 'The verified update is ready. RefMod will close, install it and restart. Refresh this page when it returns.'
    except Exception as error:
        raise gr.Error(str(error)) from error

def stop_server(confirmed):
    if not confirmed:raise gr.Error('Confirm that restarting requires RUN.bat on the PC.')
    if runtime.is_busy():raise gr.Error('Finish or stop the active job first.')
    def end():
        time.sleep(1)
        for server in SERVERS:server.close()
        os._exit(0)
    threading.Thread(target=end,daemon=True).start()
    return 'Stopping both app interfaces. Restart with RUN.bat on the PC.'

def build_ui(remote=False):
    settings=cfg.read_settings();logo=base64.b64encode((cfg.APP/'assets/ggf-brain-logo.png').read_bytes()).decode()
    missing=any(m['status']=='Missing or incomplete' for m in displayed_models())
    with gr.Blocks(title='Minimax Refmod · Get Going Fast') as ui:
        token=gr.State('');last_seen=gr.State('');browser_token=gr.Textbox(visible=False)
        gr.HTML('<header class="ggf-hero"><div><div class="ggf-kicker">GET GOING FAST · LOCAL AI</div><h1>Minimax Refmod</h1><p>Create reusable references. Bring them into your next video.</p><nav class="ggf-links"><a href="https://getgoingfast.pro" target="_blank" rel="noopener">GetGoingFast.pro ↗</a><a href="https://youtube.com/@TheAIHobbyGuy" target="_blank" rel="noopener">TheAIHobbyGuy ↗</a></nav></div><div class="ggf-hero-slogan">Your Time Is Limited.<br><strong>Get Going Fast.</strong></div><img class="ggf-hero-logo" alt="Get Going Fast" src="data:image/png;base64,'+logo+'"></header>')
        activity=gr.HTML('Ready. Set up models in Settings to begin.',elem_id='activity')
        with gr.Tabs(selected='settings' if missing and not remote else 'create'):
            with gr.Tab('Create RefMod',id='create'):
                gr.Markdown('**1. Add references → 2. Name your RefMod → 3. Create.** Full Reference retains more detail. Compressed Reference reduces token cost and can lose identity detail.')
                with gr.Row():
                    with gr.Column():
                        media=gr.File(file_count='multiple',file_types=list(previews.PICTURES+previews.VIDEOS),label='Reference images and videos',type='filepath',elem_id='ref-media')
                        media_view=gr.HTML(elem_id='ref-media-preview')
                        audio=gr.File(type='filepath',label='Optional audio reference',file_types=list(previews.AUDIO),elem_id='ref-audio')
                        audio_view=gr.HTML(elem_id='ref-audio-preview')
                    with gr.Column():
                        name=gr.Textbox(label='RefMod name',placeholder='e.g. red_jacket or character_alex')
                        mode=gr.Radio(['Full Reference','Compressed Reference'],value='Full Reference',label='Reference mode')
                        concept=gr.Dropdown([('Identity','identity'),('Clothing','clothing'),('Style','style'),('Pose or motion','pose_motion'),('Background','background'),('General reference','generic')],value='identity',filterable=False,label='Reference purpose')
                        description=gr.Textbox(label='Description',placeholder='What should this reference preserve?')
                        make=gr.Button('Create RefMod',variant='primary',elem_classes='main-action')
                        stop1=gr.Button('Stop')
                        gr.Markdown('Saved RefMods appear in the Generate Video library. A reference guides the result; it does not guarantee exact identity or motion.')
                with gr.Accordion('Creation settings',open=False):
                    with gr.Row():
                        resolution=gr.Slider(256,2048,value=1024,step=256,label='Reference resolution')
                        max_frames=gr.Number(value=450,precision=0,label='Maximum input video frames')
                        latent_frames=gr.Number(value=56,precision=0,label='Video temporal limit',info='Full Reference: sampled source frames; use 22 or more to retain motion. Compressed Reference: maximum latent frames. Sampling covers the clip.')
                    with gr.Row():
                        pool=gr.Slider(4,64,value=16,step=2,label='Compressed pooling grid',visible=False)
                        refinement=gr.Slider(0,2000,value=500,step=50,label='Compressed refinement steps',visible=False)
                        max_tokens=gr.Number(value=5120,precision=0,label='Token budget (0 = unlimited)')
                    with gr.Row():
                        audio_seconds=gr.Number(value=30,label='Audio reference seconds')
                        audio_concept=gr.Dropdown([('Voice','voice'),('Singing','singing'),('Music style','music_style'),('Sound effects','sound_fx'),('Ambience','ambience')],value='voice',filterable=False,label='Audio purpose')
                        layout=gr.Radio([('Separate files','separate_files'),('One bundle','bundle')],value='separate_files',label='Save format')
            with gr.Tab('Generate Video',id='generate'):
                gr.Markdown('**Select saved RefMods and describe the scene, movement, dialogue, and sound.** Use the reference labels below in your prompt. Ctrl+Enter starts the visible Create or Generate action.')
                with gr.Row(elem_id='generation-layout'):
                    with gr.Column(elem_id='generation-inputs'):
                        selected=gr.Dropdown(library(),multiselect=True,max_choices=8,label='RefMod library (choose up to 8)',info='Use ↑ / ↓ to change order; double-click a name to replace it. Every active member of a selected bundle is used, even when its tag is not in the prompt.',elem_id='refmod-library',value=[])
                        refmod_sort=gr.HTML('',elem_id='refmod-order',
                            js_on_load=(cfg.APP/'assets/refmod-order.js').read_text(encoding='utf-8'))
                        replace_target=gr.State('')
                        with gr.Group(visible=False,elem_id='refmod-replace') as replace_panel:
                            replacement=gr.Dropdown(library(),label='Choose one replacement',value=None,filterable=True)
                            cancel_replace=gr.Button('Cancel replacement',size='sm')
                        refresh=gr.Button('Refresh library',size='sm')
                        labels=gr.Textbox(label='Reference labels for your prompt',interactive=False,lines=3)
                        shortcuts=gr.HTML('',visible=False,elem_id='prompt-shortcuts',js_on_load=(cfg.APP/'assets/prompt-shortcuts.js').read_text(encoding='utf-8'))
                        prompt=gr.Textbox(label='Scene and sound',elem_id='scene-and-sound',lines=8,placeholder='The character from <Picture 1> walks through a rainy street. Gentle camera tracking. Footsteps and distant traffic.')
                        with gr.Row():
                            aspect=gr.Dropdown(['Portrait 9:16','Landscape 16:9','Square 1:1'],value='Portrait 9:16',filterable=False,label='Aspect ratio')
                        duration=gr.Number(value=5,label='Duration in seconds',info='Export length at 24 fps. Enter 1 for a one-second clip.')
                        preset=gr.Radio([(v[0],k) for k,v in cfg.PRESETS.items()],value=cfg.DEFAULT_PRESET,label='Generation preset',info='Pika automatic is the default and supports image, video, audio and mixed RefMods.')
                        pika_notice=gr.Markdown('',visible=False,elem_id='pika-note')
                        with gr.Accordion('LoRAs (optional)',open=False):
                            gr.Markdown('Choose MiniMax H3 video LoRAs. Set strength to 0 to skip an adapter. Select Custom steps if you supply your own speed LoRA.')
                            slots=[]
                            for index in (1,2):
                                with gr.Row():
                                    slots.append(gr.Dropdown(video_loras.choices(),value='',label=f'LoRA {index}'))
                                    slots.append(gr.Slider(-2,2,value=1,step=.05,label=f'LoRA {index} strength'))
                            lora_refresh=gr.Button('Refresh LoRA list')
                            with gr.Group(visible=not remote):
                                lora_path=gr.Textbox(label='Local LoRA file')
                                with gr.Row():
                                    lora_browse=gr.Button('Browse for LoRA')
                                    lora_link=gr.Button('Link LoRA')
                                lora_message=gr.Textbox(label='LoRA link status',interactive=False)
                        go=gr.Button('Generate Video',variant='primary',elem_classes='main-action')
                        stop2=gr.Button('Stop current job')
                    with gr.Column(elem_id='generation-output'):
                        count_options=''.join(f'<option value="{i}"{" selected" if i==20 else ""}>{i}</option>' for i in range(1,21))+'<option value="all">All</option>'
                        generation_bar=gr.HTML('',visible=False,elem_id='generation-progress')
                        progress_marker=gr.State('')
                        gr.HTML('<video controls playsinline preload="metadata" aria-label="Generated video"></video><div class="clip-caption" aria-live="polite">Completed videos appear here.</div>'
                                f'<div class="clip-tools"><label><input type="checkbox" class="loop-playback"> Loop playback</label><label><input type="checkbox" class="continuous-playback"> Continuous play</label><label>Thumbnails shown <select aria-label="Thumbnails shown">{count_options}</select></label><label>Show <select aria-label="Filter outputs"><option value="all">All outputs</option><option value="favorites">Favorites</option></select></label><button type="button" class="favorite-current" aria-pressed="false" hidden>☆ Favorite</button><button type="button" class="lineup-current" hidden>Add to lineup</button><button type="button" class="delete-current" hidden>× Delete current video</button></div>',elem_id='clip-player')
                        with gr.Row():
                            reuse_prompt=gr.Button('Reuse prompt & settings',size='sm',elem_id='reuse-active-prompt')
                            gif_button=gr.Button('Convert to GIF',size='sm',elem_id='convert-to-gif')
                        with gr.Accordion('GIF caption (optional)',open=False):
                            gif_caption=gr.Textbox(label='Caption',placeholder='Add a caption to the bottom of the GIF',lines=2)
                            gif_font=gr.Dropdown(['Arial','Times New Roman','Courier New'],value='Arial',filterable=False,label='Caption font')
                        gif_result=gr.HTML('')
                        gallery=gr.HTML('',elem_id='clip-gallery',js_on_load=(cfg.APP/'assets/clip-gallery.js').read_text(encoding='utf-8'))
                        gr.HTML('<div class="clip-navigation-bar"><button type="button" class="clips-prev" aria-label="Previous thumbnail page">Previous</button><span class="clip-page-status" aria-live="polite"></span><button type="button" class="clips-next" aria-label="Next thumbnail page">Next</button></div>',elem_id='clip-navigation')
                        gallery_keys=gr.State(None)
                        pinned_key=gr.Textbox(visible=False)
                        with gr.Group(elem_id='shot-lineup-box'):
                            gr.Markdown('### Shot lineup\nSelect shots, arrange them with the arrows, then play or stitch them in that order.')
                            lineup=gr.HTML('',elem_id='shot-lineup',js_on_load=(cfg.APP/'assets/shot-lineup.js').read_text(encoding='utf-8'))
                            lineup_marker=gr.State(None)
                            stitch_button=gr.Button('Stitch lineup into MP4',size='sm')
                            gr.Markdown('Stitching keeps the audio, uses the first shot’s canvas at 24 fps, and fits other shots without cropping.')
                            stitched=gr.HTML('',elem_id='stitched-output',js_on_load="""element.addEventListener('click',event=>{if(event.target.closest('.clear-stitched')){element.querySelector('video')?.pause();trigger('click',{});}});""")
                        with gr.Group(elem_id='queue-box'):
                            gr.Markdown('### Generation queue')
                            queue_panel=gr.HTML('<p>Queue is empty.</p>',elem_id='generation-queue',
                                js_on_load="""element.addEventListener('click',event=>{const button=event.target.closest('button[data-remove],button[data-rerun]');if(button)trigger('click',{job:button.dataset.rerun||button.dataset.remove,action:button.dataset.rerun?'rerun':'remove'});});""")
                            queue_marker=gr.State(None)
                        inspect=gr.Button('View project details',elem_id='inspect-project',size='sm')
                        gr.Markdown('Double-click a video or thumbnail to see how it was made.')
                        inspected_key=gr.State('')
                        with gr.Group(visible=False,elem_id='project-details') as project_panel:
                            facts=gr.HTML()
                            with gr.Row():
                                reuse=gr.Button('Reuse these settings',variant='primary')
                                dismiss=gr.Button('Close details')
                with gr.Accordion('Video settings',open=False):
                    run_model=gr.Dropdown(['Use Settings',video_profile.INT8,video_profile.HYBRID],value='Use Settings',label='Model for this generation',
                                          info='A project can reuse its original model without changing your default.')
                    with gr.Row():
                        mp=gr.Slider(.1,1.0,value=.5,step=.05,label='Working megapixels')
                        seed=gr.Textbox(value='',label='Seed',placeholder='Random',info='Leave blank for a new seed each run. Enter a number to reuse it.')
                        steps=gr.Slider(1,100,value=20,step=1,label='Sampling steps',visible=False)
                        batch=gr.Slider(1,8,value=1,step=1,label='Sequential videos')
                    with gr.Row():
                        strength=gr.Slider(0,1,value=1,step=.05,label='RefMod strength')
                        retention=gr.Slider(0,1,value=1,step=.05,label='Reference retention')
                    with gr.Row():
                        first=gr.Image(type='filepath',label='Optional first frame',height=160,elem_id='first-frame')
                        last=gr.Image(type='filepath',label='Optional last frame',height=160,elem_id='last-frame')
                    upscale=gr.Checkbox(False,label='Upscale and refine video',elem_classes='ggf-toggle')
                    with gr.Row(visible=False) as upscale_options:
                        upmp=gr.Slider(.2,4,value=1.2,step=.1,label='Upscaled megapixels')
                        refinesteps=gr.Slider(1,8,value=2,step=1,label='Refinement steps')
                    gr.Markdown('Exports match the entered duration at 24 fps. Long clips and large canvases use more memory. Upscale uses a learned latent upscaler and preserves the generated audio.')
                with gr.Accordion('Import existing RefMods',open=False):
                    imports=gr.File(file_count='multiple',file_types=['.safetensors'],label='RefMod files',elem_id='ref-import')
                    import_view=gr.Textbox(label='Selected RefMods',interactive=False,visible=False)
                    import_button=gr.Button('Add to library');import_status=gr.Markdown()
            with gr.Tab('RefMod library',id='library') as library_tab:
                gr.Markdown('### Manage saved RefMods\nChoose one or more RefMods to remove. Files go to the Windows Recycle Bin, where they can be restored. RefMods used by running or queued generations are protected.',elem_id='refmod-manager-intro')
                manager_selected=gr.Dropdown(library(),multiselect=True,value=[],label='Saved RefMods',elem_id='refmod-manager',interactive=not remote)
                manager_refresh=gr.Button('Refresh RefMods')
                manager_confirm=gr.Checkbox(False,label='Move the selected RefMods to the Windows Recycle Bin',interactive=not remote)
                manager_delete=gr.Button('Delete selected RefMods',variant='stop',interactive=not remote)
                manager_status=gr.Markdown('Choose RefMods, confirm, then click Delete selected RefMods.',elem_id='refmod-manager-status')
                if remote:gr.Markdown('RefMod deletion is available on the PC interface.')
            with gr.Tab('Settings',id='settings'):
                with gr.Group(visible=not remote):
                    gr.Markdown('### Output gallery')
                    recycle=gr.Checkbox(value=settings['recycle_videos'],label='Send deleted videos to the Windows Recycle Bin',
                                        info='Applies only to clips deleted in Refmod. Uncheck to use the app’s Deleted videos folder.',elem_classes='ggf-toggle')
                with gr.Group(visible=not remote):
                    active_model=gr.Textbox(value=video_profile.active_label(),label='Video generation model',interactive=False)
                    gr.Markdown('### GrizzlyMax INT4')
                    grizzly_path=gr.Textbox(label='GrizzlyMax folder or checkpoint',info='Browse below, or paste a path here and press Enter.')
                    link_grizzly=gr.Button('Link and Use GrizzlyMax Models')
                    gr.Markdown('### Standard MiniMax models')
                    folder=gr.Textbox(label='MiniMax folder or checkpoint',
                        value=settings.get('existing_model_location','') if not remote else '',
                        info='Choose a models folder to search its subfolders, or paste a .safetensors path and press Enter.')
                    browse_standard=gr.Button('Browse and Use MiniMax Models',variant='primary')
                    link_status=gr.Textbox(label='Model linking status',interactive=False)
                    link_job=gr.State('')
                gr.Markdown('### Models\nDownload the creator models first, or install the complete video set. Existing model folders can be linked after SHA-256 verification; no copying is required.')
                models=gr.HTML(model_table())
                model_refresh=gr.Button('Refresh model status')
                with gr.Group(visible=not remote):
                    profile=gr.Radio([pika_profile.DEFAULT_DOWNLOAD,'RefMod creator — 5.8 GB','Creator + video + upscale — 47.6 GB',cfg.FAST_DOWNLOAD,pika_profile.DOWNLOAD,pika_profile.REFERENCE_DOWNLOAD],value=pika_profile.DEFAULT_DOWNLOAD,label='Models to manage')
                    gr.Markdown('Pika sets reuse installed encoder and VAE files. The text / first-last option adds a 1.96 GB adapter to native INT8 H3. The references set adds its own 20.97 GB checkpoint and 1.96 GB adapter. Missing shared files are downloaded too.')
                    with gr.Row():
                        download=gr.Button('Download missing models',variant='primary')
                        reconnect_btn=gr.Button('Reconnect and resume',interactive=False)
                        verify=gr.Button('Verify model hashes')
                        stop_models=gr.Button('Stop model operation')
                    transfer_info=gr.HTML(elem_id='model-download-progress')
                maintenance_status=gr.Markdown()
                with gr.Group(visible=not remote):
                    gr.Markdown('### App updates')
                    with gr.Row():
                        check_update=gr.Button('Check for updates')
                        install_update=gr.Button('Update app')
                    result_file=cfg.APP/'runtime/update-result.json'
                    last_update=json.loads(result_file.read_text()).get('message','') if result_file.is_file() else ''
                    update_status=gr.Markdown(last_update)
                with gr.Accordion('Phone access',open=False):
                    gr.Markdown('Use your phone on the same network. The PC runs generation and must remain on. The local PC interface remains login-free. A blank password disables phone login. Changes apply after restart.')
                    network=gr.Checkbox(value=settings['phone_enabled'],label='Enable phone access',interactive=not remote,elem_classes='ggf-toggle')
                    username=gr.Textbox(value=settings['username'] if not remote else '',label='Phone username',interactive=not remote)
                    password=gr.Textbox(value=settings['password'] if not remote else '',type='password',label='Phone password',interactive=not remote)
                    save_net=gr.Button('Save phone settings',visible=not remote);network_status=gr.Markdown()
                with gr.Accordion('App controls and build information',open=False):
                    gr.Markdown(f'Version {cfg.VERSION}. Inference uses the bundled Comfy core without a ComfyUI server.')
                    free=gr.Button('Release models / free GPU memory');free_status=gr.Markdown()
                    confirm=gr.Checkbox(False,label='Stopping the server closes local and phone access; restarting requires RUN.bat on this PC.')
                    end=gr.Button('Stop server');end_status=gr.Markdown()
        downloads=gr.File(file_count='multiple',label='Saved outputs',interactive=False)
        details=gr.Textbox(label='Result details',lines=5,interactive=False)
        log=gr.File(label='Job log',interactive=False)
        gr.HTML('<footer class="ggf-footer">Powered by MiniMax H3 · RefMod by Luisacaotica · Comfy inference core</footer>')
        media.change(lambda items:previews.render(items,media_view),media,media_view,trigger_mode='always_last',show_progress='minimal')
        def change_preset(name,seconds,references):
            pika=pika_profile.enabled(name)
            adapter_updates=[gr.update(interactive=True) for _ in (0,1)]
            weight_updates=[gr.update(interactive=True) for _ in (0,1)]
            frame_updates=[gr.update(interactive=True) for _ in (0,1)]
            selected_update=gr.update(interactive=True)
            upscale_update=gr.update(interactive=True)
            return [gr.update(visible=name=='custom'),gr.update(value=pika_profile.note(name),visible=pika),upscale_update,
                    adapter_updates[0],weight_updates[0],adapter_updates[1],weight_updates[1],*frame_updates,selected_update,
                    gr.update(minimum=None,maximum=None)]
        gr.on([preset.change,selected.input],change_preset,[preset,duration,selected],[steps,pika_notice,upscale,*slots,first,last,selected,duration],queue=False,show_progress='hidden')
        upscale.change(lambda enabled:gr.update(visible=enabled),upscale,upscale_options,queue=False,show_progress='hidden')
        mode.change(lambda choice:[gr.update(visible=choice=='Compressed Reference') for _ in range(2)],mode,[pool,refinement],queue=False,show_progress='hidden')
        audio.change(lambda item:previews.render([item] if item else [],audio_view,True),audio,audio_view,trigger_mode='always_last',show_progress='minimal')
        imports.change(describe_imports,imports,import_view,queue=False)
        make.click(create,[token,media,audio,name,mode,concept,description,resolution,pool,latent_frames,refinement,max_tokens,max_frames,audio_seconds,audio_concept,layout],[activity,downloads,details],queue=False)
        lora_browse.click(lambda:browse_models(remote,'lora'),None,lora_path)
        lora_link.click(lambda loc,a,b:link_lora(loc,a,b,remote),[lora_path,*slots[::2]],[*slots[::2],lora_message],concurrency_id='lora-links')
        lora_refresh.click(lora_menus,slots[::2],slots[::2],queue=False)
        generation_fields=[prompt,selected,strength,retention,aspect,mp,duration,seed,preset,steps,upscale,upmp,refinesteps,first,last,batch,*slots,run_model]
        go.click(generate,[token,*generation_fields],[activity,downloads,details],queue=False)
        queue_panel.click(queue_action,token,[queue_panel,queue_marker],queue=False,show_progress='hidden')
        inspect.click(inspect_project,[token,pinned_key],[project_panel,facts,inspected_key],
                      js="(t,k)=>[t,document.querySelector('#clip-player video')?.dataset.key||'']",queue=False,show_progress='hidden').success(
                          None,None,None,js="()=>requestAnimationFrame(()=>document.querySelector('#project-details')?.scrollIntoView({block:'nearest'}))",queue=False)
        dismiss.click(lambda:(gr.update(visible=False),'',''),None,[project_panel,facts,inspected_key],queue=False)
        reuse.click(reuse_project,[token,inspected_key],generation_fields,queue=False)
        reuse_prompt.click(reuse_project,[token,pinned_key],generation_fields,
                          js="(t,k)=>[t,document.querySelector('#clip-player video')?.dataset.key||'']",
                          queue=False,show_progress='hidden').success(None,None,None,
                              js="()=>{const p=document.querySelector('#scene-and-sound textarea');if(p){p.focus({preventScroll:true});p.setSelectionRange(p.value.length,p.value.length);p.scrollIntoView({block:'center',behavior:'smooth'});}}",
                              queue=False)
        for stop in (stop1,stop2,stop_models):stop.click(guarded(runtime.stop),token,activity,queue=False)
        refresh.click(refresh_library,selected,selected,queue=False)
        manager_refresh.click(lambda names:gr.update(choices=library(),value=[n for n in names or [] if n in library()]),manager_selected,manager_selected,queue=False)
        library_tab.select(lambda names:gr.update(choices=library(),value=[n for n in names or [] if n in library()]),manager_selected,manager_selected,queue=False)
        manager_selected.input(lambda:False,None,manager_confirm,queue=False)
        manager_delete.click(lambda names,confirmed,chosen:manage_refmods(names,confirmed,chosen,remote),[manager_selected,manager_confirm,selected],[manager_selected,selected,manager_confirm,manager_status],concurrency_id='refmod-delete',concurrency_limit=1)
        refmod_sort.click(reference_action,[token,selected],[selected,replace_panel,replace_target,replacement],queue=False,show_progress='hidden')
        replacement.input(replace_refmod,[token,selected,replace_target,replacement],[selected,replace_panel,replace_target],queue=False,show_progress='hidden')
        cancel_replace.click(lambda:(gr.update(interactive=True),gr.update(visible=False),''),None,[selected,replace_panel,replace_target],queue=False)
        gr.on([selected.change,strength.change],reference_labels,[selected,strength],[labels,shortcuts],queue=False,trigger_mode='always_last')
        import_button.click(import_mod,imports,import_status,queue=False)
        model_refresh.click(model_table,None,models,queue=False)
        if not remote:
            ui.load(lambda:cfg.read_settings().get('existing_model_location',''),None,folder,queue=False)
            for button,field,family in ((link_grizzly,grizzly_path,'grizzly'),(browse_standard,folder,'standard')):
                button.click(lambda t,p,f=family:start_model_link(t,p,f,remote,True),[token,field],[field,link_status,link_job],concurrency_id='model-picker')
                field.submit(lambda t,p,f=family:start_model_link(t,p,f,remote,False),[token,field],[field,link_status,link_job],queue=False)
        ui.load(video_profile.active_label,None,active_model,queue=False)
        ui.load(model_table,None,models,queue=False)
        reconnect_btn.click(lambda t:resume_connection(t,remote),token,None,queue=False)
        for button,action in ((download,'download'),(verify,'verify')):
            button.click(lambda t,p,a=action:maintenance(t,a,p,remote),[token,profile],maintenance_status,queue=False)
        save_net.click(save_network,[network,username,password],network_status,queue=False)
        check_update.click(lambda:check_app_update(remote),None,update_status,queue=False)
        install_update.click(lambda progress=gr.Progress():update_app(remote,progress),None,update_status,concurrency_id='app-update',concurrency_limit=1)
        if not remote:
            ui.load(lambda:cfg.read_settings()['recycle_videos'],None,recycle,queue=False)
            recycle.input(lambda enabled:cfg.update_settings(recycle_videos=enabled),recycle,None,queue=False)
        free.click(guarded(runtime.release),None,free_status,queue=False)
        end.click(stop_server,confirm,end_status,queue=False)
        restored=ui.load(restore,browser_token,[token,prompt,selected,aspect,mp,duration,seed,preset,upscale,media,audio,name,mode,concept,description,first,last,*slots,steps,run_model,strength,retention,upmp,refinesteps,batch],js=RESTORE_JS,queue=False)
        restored.success(stitched_result,token,stitched,queue=False)
        restored.success(change_preset,[preset,duration,selected],[steps,pika_notice,upscale,*slots,first,last,selected,duration],queue=False,show_progress='hidden')
        timer=gr.Timer(1)
        timer.tick(generation_progress,[token,progress_marker],[generation_bar,progress_marker],queue=False,show_progress='hidden')
        if not remote:
            timer.tick(link_progress,[token,link_job],[link_status,link_job,active_model,models],queue=False,show_progress='hidden')
        timer.tick(poll,[token,last_seen],[activity,make,go,downloads,details,log,last_seen,models,transfer_info,reconnect_btn],queue=False)
        gallery.click(gallery_action,token,[gallery,gallery_keys],queue=False,show_progress='hidden')
        lineup.click(lineup_action,token,[lineup,lineup_marker,stitched],queue=False,show_progress='hidden')
        stitched.click(clear_stitched_preview,token,[stitched,lineup,lineup_marker],queue=False,show_progress='hidden')
        timer.tick(lineup_view,[token,lineup_marker],[lineup,lineup_marker],queue=False,show_progress='hidden')
        stitch_button.click(stitch_lineup,token,stitched,concurrency_id='shot-stitch',concurrency_limit=1)
        gif_button.click(convert_to_gif,[token,pinned_key,gif_caption,gif_font],gif_result,js="(t,k,c,f)=>[t,document.querySelector('#clip-player video')?.dataset.key||'',c,f]",concurrency_limit=1,show_progress='full')
        timer.tick(clip_gallery,[token,gallery_keys,pinned_key],[gallery,gallery_keys],js="(t,k,p)=>[t,k,sessionStorage.getItem('minimax-refmod-gallery:'+localStorage.getItem('minimax-refmod-workspace'))||'']",queue=False,show_progress='hidden')
        timer.tick(queue_view,[token,queue_marker],[queue_panel,queue_marker],queue=False)
    return ui

def launch(open_browser=True):
    settings=cfg.read_settings()
    runtime.restore_queue()
    for remote in ([False,True] if settings['phone_enabled'] else [False]):
        host='0.0.0.0' if remote else '127.0.0.1'
        start=7910 if remote else 7900
        for port in range(start,start+10):
            sock=socket.socket()
            try:sock.bind((host,port));sock.close();break
            except OSError:sock.close()
        else:raise RuntimeError(f'No free port in {start}–{start+9}.')
        ui=build_ui(remote)
        auth=(settings['username'],settings['password']) if remote and settings['password'] else None
        ui.launch(server_name=host,server_port=port,inbrowser=open_browser and not remote,prevent_thread_lock=True,share=False,auth=auth,
            css=CSS,theme=theme(),allowed_paths=[str(cfg.JOBS.resolve()),str(cfg.OUTPUTS.resolve()),str(cfg.LIBRARY.resolve())],show_error=True)
        SERVERS.append(ui)
        if not remote:(cfg.APP/'runtime/server-address.json').write_text(json.dumps({'port':port}))
        if remote:print(f'Phone access: http://{socket.gethostbyname(socket.gethostname())}:{port}',flush=True)
    try:
        while True:time.sleep(1)
    except KeyboardInterrupt:
        runtime.shutdown_workers()
        for ui in SERVERS:ui.close()
