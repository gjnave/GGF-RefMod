"""One app-owned job per process; JSONL events on stdout, diagnostics on stderr.

Imports selected Comfy inference modules directly. No ComfyUI server/frontend.
"""
from __future__ import annotations
import contextlib, json, sys, time, traceback, uuid
import video_profile
import video_loras
import pika_profile
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
        if strength<=0:continue
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

def prepare_video_output(name):
    root=(APP/'outputs').resolve()
    path=(root/(name+'.mp4')).resolve()
    if path.parent!=root:
        raise ValueError('Video output must be inside the app output folder.')
    if path.is_file():
        path.rename(path.with_name(path.stem+'.interrupted-'+uuid.uuid4().hex[:8]+path.suffix))
    return path

def trim_output(request,images,audio):
    """Keep sampling on H3's frame grid; export the requested video/audio length."""
    wanted=request.get('output_frames')
    if wanted is None:return images,audio
    if type(wanted) is not int or not 1<=wanted<=len(images):
        raise ValueError('Requested export length is outside the generated video.')
    images=images[:wanted]
    samples=round(wanted*int(audio['sample_rate'])/24)
    waveform=audio['waveform'][...,:samples]
    if waveform.shape[-1]<samples:
        import torch.nn.functional as functional
        waveform=functional.pad(waveform,(0,samples-waveform.shape[-1]))
    return images,dict(audio,waveform=waveform)


def generate(request,engine):
    torch,nodes,ref,h3,sampling,upscale,video=engine
    active=[path for path,strength in zip(request.get('mods',[]),request.get('strengths',[])) if strength>0]
    request=dict(request,preset=pika_profile.resolve(saved_preset(request),active))
    pika_profile.validate(request)
    from refmod.prompt import reference_map
    profile=request.get('video_profile',{'active':video_profile.INT8})
    hybrid=profile['active']==video_profile.HYBRID
    preset=saved_preset(request)
    adapter,steps,sampler_name,video_shift=sampling_settings(preset,request['steps'])
    base_key=pika_profile.model_key(preset)
    require_models(([] if hybrid else [base_key])+['encoder','video_vae','audio_vae']+([adapter] if adapter else [])+(['upscaler','dmd','refiner'] if request['upscale'] else []))
    mods=load_mods(request['mods'],request['strengths'])
    if preset=='pika_refs':
        pika_profile.validate_references(mods)
    # The inference lock is held here, including after waiting for an orphaned worker.
    target=str(prepare_video_output(request['output_name']))
    emit('stage',message='Loading MiniMax H3 and text encoder',phase='load')
    if hybrid:
        import comfy.sd
        checkpoint=video_profile.linked_path(profile)
        base=comfy.sd.load_diffusion_model(str(checkpoint))
    else:
        import folder_paths
        base_name=model_name(base_key)
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
    elif adapter in ('turbo','pika_fl2va','pika_ref2va'):
        model=nodes.LoraLoaderModelOnly().load_lora_model_only(base,model_name(adapter),1.0)[0]
        if pika_profile.enabled(preset):
            matched=sum(len(value) for value in model.patches.values())-sum(len(value) for value in base.patches.values())
            if matched<=0:raise ValueError('The Pika adapter did not match its checkpoint. Verify the Pika model set in Settings.')
            emit('stage',message=f'Applied Pika adapter to {matched} model weights')
    model=h3.MiniMaxH3SigmaShift.execute(model,video_shift,3.0)[0]
    first=last=None
    if request.get('first_frame'):
        first=load_image(request['first_frame'])
    if request.get('last_frame'):
        last=load_image(request['last_frame'])
    emit('stage',message='Encoding prompt and reference conditioning',phase='encode')
    with torch.no_grad():
        if preset=='pika_refs':
            positive,latent=h3.MiniMaxH3ReferenceToVideo.execute(clip,request['prompt'],request['width'],request['height'],request['frames'],vae=vae,audio_vae=audio_vae)
            for image,index in ((first,0),(last,request.get('output_frames',request['frames'])-1)):
                if image is not None:
                    positive=h3.MiniMaxH3AddGuide.execute(positive,latent,index,vae=vae,image=image)[0]
        else:
            positive,latent=h3.MiniMaxH3ImageToVideo.execute(clip,vae,request['prompt'],request['width'],request['height'],request['frames'],first,last)
            if last is not None and request.get('output_frames',request['frames'])!=request['frames']:
                # Export may be shorter than H3's padded sampling grid.
                import node_helpers
                keyframes=[dict(item) for item in positive[0][1].get('minimax_keyframes',[])]
                for item in keyframes:
                    if item['resolved_frame_index']==request['frames']-1:
                        item['resolved_frame_index']=request['output_frames']-1
                positive=node_helpers.conditioning_set_values(positive,{'minimax_keyframes':keyframes})
        if mods:
            positive=ref.MiniMaxH3RefModApply.execute(positive,mods,retention=request['retention'],curve_direction='constant',curve_shape='linear',curve_value=1.0,override=False,scramble_mode='shuffle',scramble_keep=1)[0]
        from collections import Counter
        reference_kinds=dict(Counter(mod.kind for mod,strength in mods if strength>0))
        emit('conditioning',reference_kinds=reference_kinds,keyframe_indices=[item['resolved_frame_index'] for item in positive[0][1].get('minimax_keyframes',[])])
        noise=sampling.RandomNoise.execute(request['seed'])[0]
        sampler=sampling.KSamplerSelect.execute(sampler_name)[0]
        if pika_profile.enabled(preset):
            sigmas=sampling.ManualSigmas.execute(','.join(str(value) for value in pika_profile.SIGMAS))[0]
        else:
            sigmas=sampling.BasicScheduler.execute(model,'simple',steps,1.0)[0]
        guider=sampling.BasicGuider.execute(model,positive)[0]
        emit('stage',message=f'Generating video and audio: {PRESETS[preset][0]} ({steps} steps, {sampler_name})',phase='sample')
        latent=sampling.SamplerCustomAdvanced.execute(noise,guider,sampler,sigmas,latent)[1]
        if request['upscale']:
            import comfy.model_management as mm
            import comfy.nested_tensor
            emit('stage',message='Upscaling video latents; preserving audio',phase='upscale')
            mm.unload_all_models()
            upmodel=upscale.LatentUpscaleModelLoader.execute(model_name('upscaler'))[0]
            latent=h3.MiniMaxH3LatentUpscaleWithModel.execute(upmodel,latent,request['up_width'],request['up_height'])[0]
            # Endpoint latents must share the enlarged target grid during refinement.
            # Reusing the first pass's smaller guides produces a packed-layout mismatch.
            if first is not None or last is not None:
                import node_helpers
                positive=node_helpers.conditioning_set_values(positive,{'minimax_keyframes':[]})
                for image,index in ((first,0),(last,request.get('output_frames',request['frames'])-1)):
                    if image is not None:
                        if index==0 and preset!='pika_refs':
                            image=h3._resize(image[:1],request['up_width'],request['up_height'],'disabled')
                        positive=h3.MiniMaxH3AddGuide.execute(positive,latent,index,vae=vae,image=image)[0]
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
            emit('stage',message='Refining enlarged video',phase='refine')
            refine_sampler=sampling.KSamplerSelect.execute('res_multistep')[0]
            latent=sampling.SamplerCustomAdvanced.execute(noise,guider,refine_sampler,schedule,latent)[1]
            # Keep the original audio bit-for-bit in latent space.
            latent['samples']=comfy.nested_tensor.NestedTensor((latent['samples'].unbind()[0],a))
        import comfy.model_management as mm
        mm.unload_all_models()
        emit('stage',message='Decoding video',phase='decode_video')
        images=nodes.VAEDecode().decode(vae,latent)[0]
        emit('stage',message='Decoding audio',phase='decode_audio')
        from comfy_extras.nodes_audio import vae_decode_audio
        audio=vae_decode_audio(audio_vae,latent)
        images,audio=trim_output(request,images,audio)
        emit('stage',message='Saving MP4',phase='save')
        output=video.CreateVideo.execute(images,24.0,audio,bit_depth=8,color_space='sRGB')[0]
        metadata={'prompt':request['prompt'],'seed':str(request['seed']),'preset':PRESETS[preset][0]}
        if pika_profile.enabled(preset):
            metadata.update(pika_adapter=model_name(adapter),sampler='lcm',sigmas=json.dumps(pika_profile.SIGMAS))
        if adapters:
            metadata['loras']=json.dumps([(Path(a['path']).name,a['strength']) for a in adapters])
        output.save_to(target,metadata=metadata)
    record=dict(model=profile['active'],checkpoint=checkpoint.name,checkpoint_bytes=checkpoint.stat().st_size,seed=str(request['seed']),
                frames=int(images.shape[0]),width=int(images.shape[2]),height=int(images.shape[1]),fps=24,sampling_frames=request['frames'],reference_kinds=reference_kinds)
    if pika_profile.enabled(preset):
        record.update(pika_adapter=model_name(adapter),sampler='lcm',sigmas=list(pika_profile.SIGMAS),
                      torch_peak_reserved_mib=round(torch.cuda.max_memory_reserved()/1024**2))
    return {'video':target,'files':[target],'seed':request['seed'],'frames':record['frames'],'record':record,'reference_map':reference_map(mods),'details':f"Saved {record['frames']} frames at 24 fps. Seed {request['seed']}."}

def load_image(path):
    from PIL import Image,ImageOps
    import numpy as np,torch
    with Image.open(path) as im:
        return torch.from_numpy(np.asarray(ImageOps.exif_transpose(im).convert('RGB')).copy()).float().unsqueeze(0)/255

def main():
    probe='--probe' in sys.argv
    request=json.loads(Path(sys.argv[1]).read_text()) if not probe else None
    try:
        if request and request['kind']=='generate' and request.get('resume_from',0)>=request.get('batch',1):
            files=request.get('completed_files',[])
            emit('result',ok=True,files=files,video=files[-1] if files else None,details='All batch videos were already saved.')
            return
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
                    for index in range(request.get('resume_from',0),request.get('batch',1)):
                        job=dict(request,seed=(request['seed']+index)%(2**64),output_name=request['output_name']+f'-{index+1:02d}')
                        emit('stage',message=f"Video {index+1} of {request.get('batch',1)}",clip_index=index+1,phase='load')
                        results.append(generate(job,engine))
                        finished=time.monotonic()
                        record=results[-1]['record']
                        record['elapsed']=round(finished-clip_start,3)
                        emit('clip',file=results[-1]['video'],record=record)
                        clip_start=finished
                    result=dict(results[-1],files=request.get('completed_files',[])+[f for r in results for f in r['files']],details='\n'.join(r['details'] for r in results))
        emit('result',ok=True,**result)
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        emit('result',ok=False,error=str(error));sys.exit(1)

if __name__=='__main__': main()
