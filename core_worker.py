"""One app-owned job per process; JSONL events on stdout, diagnostics on stderr.

Imports selected Comfy inference modules directly. No ComfyUI server/frontend.
"""
from __future__ import annotations
import contextlib, json, sys, time, traceback
import video_profile
import video_loras
from config import saved_preset, sampling_settings, PRESETS
from fractions import Fraction
from pathlib import Path
from config import APP, MODELS, LIBRARY, FILES, model_file, model_name, require_models, initialize

PROTOCOL=sys.stdout
def emit(kind, **data):
    PROTOCOL.write(json.dumps({'event':kind,**data})+'\n');PROTOCOL.flush()

@contextlib.contextmanager
def inference_slot():
    # The worker owns the lock, so a server restart cannot overlap an orphan worker.
    import errno, msvcrt, time
    with (APP/'runtime'/'inference.lock').open('a+b') as handle:
        handle.seek(0)
        while True:
            try:
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
                break
            except OSError as exc:
                if exc.errno not in (errno.EACCES,errno.EAGAIN):raise
                emit('stage',message='Waiting for the previous inference worker')
                time.sleep(2)
        yield

def boot(cpu=False):
    initialize()
    sys.path.insert(0,str(APP/'vendor'))
    sys.path.insert(0,str(APP/'vendor/comfy_core'))
    import comfy.cli_args
    comfy.cli_args.args.cpu=cpu
    # Preview models aren't needed by this product; keep sampling progress only.
    import torch, nodes, folder_paths, comfy.utils
    folder_paths.models_dir=str(MODELS)
    folder_paths.set_output_directory(str(APP/'outputs'))
    folder_paths.set_input_directory(str(APP/'jobs'))
    for key,item in FILES.items():
        folder_paths.add_model_folder_path(item['path'].split('/')[0],str(model_file(key).parent),is_default=True)
    folder_paths.add_model_folder_path('refmods',str(LIBRARY),is_default=True)
    import refmod.nodes
    from comfy_extras import nodes_minimax_h3, nodes_custom_sampler, nodes_hunyuan, nodes_video
    comfy.utils.set_progress_bar_global_hook(lambda value,total,preview=None,**kw:emit('progress',value=value,total=total))
    return torch,nodes,refmod.nodes,nodes_minimax_h3,nodes_custom_sampler,nodes_hunyuan,nodes_video

def load_mods(paths,strengths):
    from refmod.core import H3RefMod,read_refmod_meta
    from refmod.bundle import load_bundle
    mods=[]
    for path,strength in zip(paths,strengths):
        stem=str(Path(path).with_suffix(''))
        meta=read_refmod_meta(stem)
        group=load_bundle(stem) if meta.get('kind')=='bundle' else [(H3RefMod.load(stem),1.0)]
        mods.extend((m,float(s)*float(strength)) for m,s in group)
    return mods

def create(request,engine):
    torch,nodes,ref,_,_,_,_=engine
    from PIL import Image,ImageOps
    import numpy as np
    from refmod.common import load_video_file
    from refmod.prompt import reference_map
    emit('stage',message='Loading reference media')
    images={};videos={}
    for path in request['media']:
        if Path(path).suffix.lower() in {'.png','.jpg','.jpeg','.webp','.bmp'}:
            with Image.open(path) as im:
                im=ImageOps.exif_transpose(im).convert('RGB')
                im.thumbnail((request['resolution'],request['resolution']))
                images[f'ref_image_{len(images)+1}']=torch.from_numpy(np.asarray(im).copy()).float().unsqueeze(0)/255
        else:
            videos[f'ref_video_{len(videos)+1}']=load_video_file(path,request['max_frames'],request['resolution'])
    keys=(['video_vae'] if images or videos else [])+(['audio_vae'] if request.get('audio') else [])
    require_models(keys)
    emit('stage',message='Loading reference encoders')
    vae=nodes.VAELoader().load_vae(model_name('video_vae'))[0] if images or videos else None
    audio_vae=nodes.VAELoader().load_vae(model_name('audio_vae'))[0] if request.get('audio') else None
    audio=None
    if request.get('audio'):
        import av
        with av.open(request['audio']) as container:
            resampler=av.AudioResampler(format='fltp',layout='stereo',rate=32000)
            chunks=[];count=0;limit=int(request['audio_seconds']*32000)
            for frame in container.decode(audio=0):
                for out in resampler.resample(frame):
                    chunks.append(out.to_ndarray());count+=out.samples
                if count>=limit: break
            if not chunks: raise ValueError('Audio reference contains no decodable audio.')
            waveform=np.concatenate(chunks,axis=1)[:,:limit]
            audio={'waveform':torch.from_numpy(waveform).unsqueeze(0),'sample_rate':32000}
    emit('stage',message='Encoding and saving RefMod')
    result=ref.MiniMaxH3RefModMasterExtract.execute(
        name=request['name'],mode=request['mode'],refs_image=images,refs_video=videos,
        vae=vae,audio=audio,audio_vae=audio_vae,concept_type=request['concept'],
        ref_resolution=request['resolution'],pool_h=request['pool'],pool_w=request['pool'],
        latent_frames=request['latent_frames'],identity=request['refinement'],multiplier=1,
        max_tokens=request['max_tokens'],budget_policy='truncate',save=True,
        save_layout=request['layout'],audio_max_seconds=request['audio_seconds'],
        audio_max_tokens=request['max_tokens'],audio_budget_policy='error',
        audio_concept_type=request['audio_concept'],description=request['description'])
    details=json.loads(result[1]);preview=None
    if images:
        t=next(iter(images.values()))[0]
        preview=str(Path(request['job_dir'])/'reference-preview.png')
        Image.fromarray((t.numpy()*255).astype('uint8')).save(preview)
    return {'files':details['saved_paths'],'details':result[1], 'reference_map':reference_map(result[0]),'preview':preview}

def generate(request,engine):
    torch,nodes,ref,h3,sampling,upscale,video=engine
    from refmod.prompt import reference_map
    profile=request.get('video_profile',{'active':video_profile.INT8})
    hybrid=profile['active']==video_profile.HYBRID
    preset=saved_preset(request)
    adapter,steps,sampler_name,video_shift=sampling_settings(preset,request['steps'])
    require_models(([] if hybrid else ['base'])+['encoder','video_vae','audio_vae']+([adapter] if adapter else [])+(['upscaler','dmd','refiner'] if request['upscale'] else []))
    emit('stage',message='Loading MiniMax H3 and text encoder')
    if hybrid:
        import comfy.sd
        checkpoint=video_profile.linked_path(profile)
        base=comfy.sd.load_diffusion_model(str(checkpoint))
    else:
        import folder_paths
        base_name=model_name('base')
        checkpoint=Path(folder_paths.get_full_path_or_raise('diffusion_models',base_name))
        base=nodes.UNETLoader().load_unet(base_name,'default')[0]
    clip=nodes.CLIPLoader().load_clip(model_name('encoder'),'minimax','default')[0]
    vae=nodes.VAELoader().load_vae(model_name('video_vae'))[0]
    audio_vae=nodes.VAELoader().load_vae(model_name('audio_vae'))[0]
    adapters=request.get('loras',[])
    base=video_loras.apply(base,adapters,lambda message:emit('stage',message=message)) if adapters else base
    model=base
    if adapter=='fast_preview':
        path=model_file(adapter)
        stat=path.stat()
        model=video_loras.apply(base,[dict(path=str(path),size=stat.st_size,mtime=stat.st_mtime_ns,strength=1)],lambda msg:emit('stage',message=msg))
    elif adapter=='turbo':
        model=nodes.LoraLoaderModelOnly().load_lora_model_only(base,model_name('turbo'),1.0)[0]
    model=h3.MiniMaxH3SigmaShift.execute(model,video_shift,3.0)[0]
    mods=load_mods(request['mods'],request['strengths'])
    first=last=None
    if request.get('first_frame'):
        first=load_image(request['first_frame'])
    if request.get('last_frame'):
        last=load_image(request['last_frame'])
    emit('stage',message='Encoding prompt and reference conditioning')
    with torch.no_grad():
        positive,latent=h3.MiniMaxH3ImageToVideo.execute(clip,vae,request['prompt'],request['width'],request['height'],request['frames'],first,last)
        if mods:
            positive=ref.MiniMaxH3RefModApply.execute(positive,mods,retention=request['retention'],curve_direction='constant',curve_shape='linear',curve_value=1.0,override=False,scramble_mode='shuffle',scramble_keep=1)[0]
        noise=sampling.RandomNoise.execute(request['seed'])[0]
        sampler=sampling.KSamplerSelect.execute(sampler_name)[0]
        sigmas=sampling.BasicScheduler.execute(model,'simple',steps,1.0)[0]
        guider=sampling.BasicGuider.execute(model,positive)[0]
        emit('stage',message=f'Generating video and audio: {PRESETS[preset][0]} ({steps} steps, {sampler_name})')
        latent=sampling.SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent)[1]
        if request['upscale']:
            import comfy.model_management as mm
            import comfy.nested_tensor
            emit('stage',message='Upscaling video latents; preserving audio')
            mm.unload_all_models()
            upmodel=upscale.LatentUpscaleModelLoader.execute(model_name('upscaler'))[0]
            latent=h3.MiniMaxH3LatentUpscaleWithModel.execute(upmodel,latent,request['up_width'],request['up_height'])[0]
            v,a=latent['samples'].unbind()
            latent['noise_mask']=comfy.nested_tensor.NestedTensor((torch.ones_like(v),torch.zeros_like(a)))
            # Fresh base avoids stacking the first-pass FL2V turbo adapter with DMD.
            refined=nodes.LoraLoaderModelOnly().load_lora_model_only(base,model_name('dmd'),1.0)[0]
            refined=nodes.LoraLoaderModelOnly().load_lora_model_only(refined,model_name('refiner'),1.0)[0]
            refined=h3.MiniMaxH3SigmaShift.execute(refined,12.0,3.0)[0]
            guider=sampling.BasicGuider.execute(refined,positive)[0]
            schedule=sampling.BasicScheduler.execute(refined,'simple',8,1.0)[0]
            # A low-noise tail actually provides two refinement steps.
            schedule=schedule[-(request['refine_steps']+1):]
            emit('stage',message='Refining enlarged video')
            refine_sampler=sampling.KSamplerSelect.execute('res_multistep')[0]
            latent=sampling.SamplerCustomAdvanced.execute(noise,guider,refine_sampler,schedule,latent)[1]
            # Keep the original audio bit-for-bit in latent space.
            latent['samples']=comfy.nested_tensor.NestedTensor((latent['samples'].unbind()[0],a))
        import comfy.model_management as mm
        mm.unload_all_models()
        emit('stage',message='Decoding video')
        images=nodes.VAEDecode().decode(vae,latent)[0]
        emit('stage',message='Decoding audio')
        from comfy_extras.nodes_audio import vae_decode_audio
        audio=vae_decode_audio(audio_vae,latent)
        emit('stage',message='Saving MP4')
        target=str(APP/'outputs'/(request['output_name']+'.mp4'))
        output=video.CreateVideo.execute(images,24.0,audio,bit_depth=8,color_space='sRGB')[0]
        metadata={'prompt':request['prompt'],'seed':str(request['seed']),'preset':PRESETS[preset][0]}
        if adapters:
            metadata['loras']=json.dumps([(Path(a['path']).name,a['strength']) for a in adapters])
        output.save_to(target,metadata=metadata)
    record=dict(model=profile['active'],checkpoint=checkpoint.name,checkpoint_bytes=checkpoint.stat().st_size,seed=str(request['seed']),
                frames=int(images.shape[0]),width=int(images.shape[2]),height=int(images.shape[1]),fps=24)
    return {'video':target,'files':[target],'seed':request['seed'],'frames':request['frames'],'record':record,'reference_map':reference_map(mods),'details':f"Saved {record['frames']} frames at 24 fps. Seed {request['seed']}."}

def load_image(path):
    from PIL import Image,ImageOps
    import numpy as np,torch
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()).float().unsqueeze(0)/255

def main():
    probe='--probe' in sys.argv
    request=json.loads(Path(sys.argv[1]).read_text()) if not probe else None
    try:
        with contextlib.redirect_stdout(sys.stderr), (contextlib.nullcontext() if probe else inference_slot()):
            emit('stage',message='Starting private inference worker')
            clip_start=time.monotonic()
            engine=boot(cpu=probe)
            if probe:
                result={'imports':'ok','server_loaded':'server' in sys.modules}
            else:
                if request['kind']=='create':result=create(request,engine)
                else:
                    results=[]
                    for index in range(request.get('batch',1)):
                        job=dict(request,seed=(request['seed']+index)%(2**64),output_name=request['output_name']+f'-{index+1:02d}')
                        emit('stage',message=f"Video {index+1} of {request.get('batch',1)}",clip_index=index+1)
                        results.append(generate(job,engine))
                        finished=time.monotonic()
                        record=results[-1]['record']
                        record['elapsed']=round(finished-clip_start,3)
                        emit('clip',file=results[-1]['video'],record=record)
                        clip_start=finished
                    result=dict(results[-1],files=[f for r in results for f in r['files']],details='\n'.join(r['details'] for r in results))
        emit('result',ok=True,**result)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit('result',ok=False,error=str(error));sys.exit(1)

if __name__=='__main__': main()
