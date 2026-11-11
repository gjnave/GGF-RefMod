"""Read generation receipts without changing the workspace or any saved clip."""
from pathlib import Path
import html

import config as cfg
import runtime
import video_profile
import pika_profile


def project(token,key):
    source,index,req,rec,job_time=runtime.clip_project(token,key)
    profile=req.get('video_profile',{})
    model=rec.get('model') or profile.get('active',video_profile.INT8)
    checkpoint=rec.get('checkpoint') or (Path(profile['path']).name if model==video_profile.HYBRID and profile.get('path') else '')
    w,h=(req['up_width'],req['up_height']) if req.get('upscale') else (req['width'],req['height'])
    frames,fps=rec.get('frames',req['frames']),rec.get('fps',24)
    w,h=rec.get('width',w),rec.get('height',h)
    if not rec:  # Older jobs have no clip receipt. Prefer the MP4's own stream information.
        import av
        try:
            with av.open(str(source)) as media:
                stream=media.streams.video[0]
                w,h=stream.width,stream.height
                fps=float(stream.average_rate or fps)
                frames=stream.frames or frames
        except (OSError,ValueError,IndexError):pass
    return dict(name=source.name,index=index,req=req,model=model,checkpoint=checkpoint,checkpoint_bytes=rec.get('checkpoint_bytes'),
                seed=str(rec.get('seed',(int(req['seed'])+index-1)%2**64)),width=w,height=h,frames=frames,fps=fps,
                elapsed=rec.get('elapsed'),job_time=job_time)


def clock(seconds):
    if seconds is None:return 'Not recorded'
    m,s=divmod(round(max(0,seconds)),60)
    return f'{m}m {s:02d}s' if m else f'{s}s'


def details(p):
    r=p['req']; preset=cfg.saved_preset(r)
    rows=[('Video',p['name']),('Model',p['model']+(' · '+p['checkpoint'] if p['checkpoint'] else ' · checkpoint filename not recorded')),
          ('Seed',p['seed']),('Video length',f"{p['frames']/p['fps']:.2f} s · {p['frames']} frames at {p['fps']:g} fps"),
          ('Output size',f"{p['width']} × {p['height']}"),('Clip render time',clock(p['elapsed'])),('Job run time',clock(p['job_time'])),
          ('Sampling',f"{cfg.PRESETS[preset][0]} · {r['steps']} steps"),('Batch',f"Clip {p['index']} of {r.get('batch',1)}")]
    if 'duration' in r.get('form',{}):rows.append(('Requested length',f"{r['form']['duration']:g} s"))
    if r.get('mods'):rows.append(('RefMods',', '.join(Path(path).name for path in r['mods'])))
    if r.get('loras'):rows.append(('LoRAs',', '.join(f"{Path(a['path']).name} ({a['strength']:g})" for a in r['loras'])))
    return ('<section class="project-sheet"><h3>Project details</h3><dl class="project-facts">'
            + ''.join(f'<dt>{k}</dt><dd>{html.escape(str(v))}</dd>' for k,v in rows)
            + '</dl><strong>Scene and sound</strong><p class="project-prompt">'
            + html.escape(r['prompt'])+'</p></section>')


def legacy_canvas(r,aspect=None,up=False):
    from itertools import product
    sizes=[n/10 for n in range(2,41)] if up else [n/100 for n in range(10,101,5)]
    target=(r['up_width'],r['up_height']) if up else (r['width'],r['height'])
    for ratio,mp in product([aspect] if aspect else ['Portrait 9:16','Landscape 16:9','Square 1:1'],sizes):
        if cfg.canvas(ratio,mp)==target:return ratio,mp
    raise ValueError('This older clip has canvas dimensions the current form cannot reproduce.')


def reuse_values(p,library,adapters):
    r=p['req']; f=r.get('form',{}); model=p['model']
    profile=video_profile.settings()
    checkpoint=video_profile.linked_path(profile) if model==video_profile.HYBRID else cfg.model_file(pika_profile.model_key(cfg.saved_preset(r)))
    if p['checkpoint_bytes'] and (not checkpoint.is_file() or checkpoint.stat().st_size!=p['checkpoint_bytes']):
        raise ValueError('Link the checkpoint shown in Project details in Settings before reusing this project.')
    if 'aspect' in f:aspect,mp,upmp=f['aspect'],f['mp'],f['upmp']
    else:
        aspect,mp=legacy_canvas(r)
        upmp=legacy_canvas(r,aspect,True)[1] if r.get('upscale') else 1.2
    by_path={(cfg.LIBRARY/n).resolve():n for n in library}
    names=f.get('selected')
    if names is None:names=[by_path.get(Path(n).resolve(),Path(n).name) for n in r.get('mods',[])]
    missing=[n for n in names if n not in library]
    slots=f.get('loras') or [(a['path'],a['strength']) for a in r.get('loras',[])]
    slots=(list(slots)+[('',1),('',1)])[:2]
    missing.extend(Path(path).name for path,_ in slots if path and path not in adapters)
    frames=[r.get('first_frame'),r.get('last_frame')]
    missing.extend(Path(path).name for path in frames if path and not Path(path).is_file())
    if missing:raise ValueError('Restore these project inputs before reusing: '+', '.join(missing))
    seed=p['seed'] if cfg.seed_input(f.get('seed_input','')) else ''
    return dict(prompt=r['prompt'],selected=names,strength=f.get('strength',(r.get('strengths') or [1])[0]),retention=r.get('retention',1),
                aspect=aspect,mp=mp,duration=f.get('duration',r['frames']/24),seed=seed,preset=cfg.saved_preset(r),steps=f.get('custom_steps',r['steps']),
                upscale=r['upscale'],upmp=upmp,refinesteps=r.get('refine_steps',2),frames=frames,slots=slots,model=model,batch=r.get('batch',1))
