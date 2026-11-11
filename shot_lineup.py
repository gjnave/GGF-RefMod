"""Stream an ordered set of shots into one H.264/AAC MP4 using bundled PyAV."""
from fractions import Fraction
from pathlib import Path
from uuid import uuid4
import json
import av
import numpy as np
from PIL import Image


def fitted(frame,width,height):
    if (frame.width,frame.height)==(width,height):return frame.reformat(width,height,'yuv420p')
    scale=min(width/frame.width,height/frame.height)
    w,h=max(1,round(frame.width*scale)),max(1,round(frame.height*scale))
    image=frame.reformat(w,h,'rgb24').to_image()
    canvas=Image.new('RGB',(width,height))
    canvas.paste(image,((width-w)//2,(height-h)//2))
    return av.VideoFrame.from_image(canvas).reformat(width,height,'yuv420p')


def stitch(sources,folder,keys,progress=lambda fraction,message:None):
    sources=[Path(p) for p in sources]
    if len(sources)<2:raise ValueError('Choose at least two shots.')
    info=[]
    for source in sources:
        if not source.is_file():raise ValueError('A selected shot is no longer available.')
        with av.open(str(source)) as media:
            if not media.streams.video:raise ValueError(f'{source.name} has no video track.')
            v=media.streams.video[0]
            info.append(dict(width=v.width,height=v.height,rate=v.average_rate or Fraction(24),audio=bool(media.streams.audio)))
    width,height=[max(2,n//2*2) for n in (info[0]['width'],info[0]['height'])]
    rate=24;sample_rate=48000
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    target=folder/f'Lineup-{uuid4().hex[:12]}.mp4'
    temporary=target.with_suffix('.partial.mp4')
    receipt=[];video_pts=audio_pts=0
    with av.open(str(temporary),'w',format='mp4',options={'movflags':'+faststart'}) as dest:
        video=dest.add_stream('libx264',rate=rate,options={'preset':'veryfast','crf':'18'})
        video.width,video.height,video.pix_fmt=width,height,'yuv420p'
        video.codec_context.thread_count=2
        has_audio=any(row['audio'] for row in info)
        if has_audio:
            audio=dest.add_stream('aac',rate=sample_rate)
            audio.layout='stereo';audio.bit_rate=192000

        def write_audio(samples):
            nonlocal audio_pts
            frame=av.AudioFrame.from_ndarray(np.ascontiguousarray(samples,dtype=np.float32),format='fltp',layout='stereo')
            frame.sample_rate=sample_rate;frame.time_base=Fraction(1,sample_rate);frame.pts=audio_pts
            audio_pts+=frame.samples
            dest.mux(audio.encode(frame))

        for index,(source,meta) in enumerate(zip(sources,info)):
            progress(index/len(sources),f'Stitching shot {index+1} of {len(sources)}')
            start=video_pts;count=0
            with av.open(str(source)) as media:
                for number,frame in enumerate(media.decode(video=0),1):
                    until=max(1,round(Fraction(number*rate,1)/meta['rate']))
                    if count>=until:continue
                    sample=fitted(frame,width,height)
                    while count<until:
                        sample.pts=video_pts;sample.time_base=Fraction(1,rate)
                        dest.mux(video.encode(sample));video_pts+=1;count+=1
            if not count:raise ValueError(f'{source.name} contains no decodable video frames.')
            if has_audio:
                remaining=count*(sample_rate//rate)
                if meta['audio']:
                    with av.open(str(source)) as media:
                        resampler=av.AudioResampler(format='fltp',layout='stereo',rate=sample_rate)
                        for frame in media.decode(audio=0):
                            frame.pts=None
                            for chunk in resampler.resample(frame):
                                take=min(remaining,chunk.samples)
                                if take:write_audio(chunk.to_ndarray()[:,:take]);remaining-=take
                            if not remaining:break
                        if remaining:
                            for chunk in resampler.resample(None):
                                take=min(remaining,chunk.samples)
                                if take:write_audio(chunk.to_ndarray()[:,:take]);remaining-=take
                while remaining:
                    take=min(remaining,4096)
                    write_audio(np.zeros((2,take),dtype=np.float32));remaining-=take
            receipt.append(dict(key=keys[index],source=source.name,start_seconds=start/rate,duration_seconds=count/rate))
        dest.mux(video.encode())
        if has_audio:dest.mux(audio.encode())
    temporary.rename(target)
    details=dict(file=target.name,width=width,height=height,fps=rate,duration_seconds=video_pts/rate,shots=receipt)
    target.with_suffix('.json').write_text(json.dumps(details,indent=2),encoding='utf-8')
    progress(1,'Stitched video ready')
    return target,details
