"""Stream a generated video to an animated GIF with a palette for each frame."""
from fractions import Fraction
from uuid import uuid4

import av
import numpy as np
from PIL import Image


def indexed_frame(frame,width,height):
    image=frame.reformat(width,height,format='rgb24').to_image()
    indexed=image.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    colors=np.asarray(indexed.getpalette(),dtype=np.uint8).reshape(-1,3)
    palette=np.zeros((256,4),dtype=np.uint8)
    palette[:,0]=255
    palette[:len(colors),1:]=colors
    return av.VideoFrame.from_ndarray((np.asarray(indexed),palette),format='pal8')


def export(source,loop=True):
    if not source.is_file():
        raise ValueError('The selected video is no longer available.')
    folder=source.parent/'GIFs'
    folder.mkdir(exist_ok=True)
    target=folder/f'{source.stem}-{uuid4().hex[:10]}.gif'
    temporary=target.with_suffix('.partial')
    with av.open(str(source)) as media:
        stream=media.streams.video[0]
        rate=stream.average_rate or Fraction(24)
        stride=max(1,int(np.ceil(rate/12)))
        scale=min(1,640/max(stream.width,stream.height))
        width,height=[max(1,round(n*scale)) for n in (stream.width,stream.height)]
        options={'loop':'0' if loop else '-1'}
        if stream.frames:
            last=((stream.frames-1)//stride)*stride
            options['final_delay']=str(max(1,round(100*stream.frames/rate)-round(100*last/rate)))
        with av.open(str(temporary),'w',format='gif',options=options) as dest:
            video=dest.add_stream('gif',rate=rate/stride)
            video.width,video.height=width,height
            video.pix_fmt='pal8'
            video.codec_context.time_base=Fraction(1,100)
            count=0
            for index,frame in enumerate(media.decode(stream)):
                if index%stride:continue
                sample=indexed_frame(frame,width,height)
                sample.pts=round(100*index/rate)
                sample.time_base=Fraction(1,100)
                dest.mux(video.encode(sample))
                count+=1
            if not count:raise ValueError('No video frames could be decoded.')
            dest.mux(video.encode())
    temporary.rename(target)
    return target
