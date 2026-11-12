"""Stream a generated video to an animated GIF with a palette for each frame."""
from fractions import Fraction
from uuid import uuid4
from pathlib import Path
import os

import av
import numpy as np
from PIL import Image,ImageDraw,ImageFont

FONTS={'Arial':'arialbd.ttf','Times New Roman':'timesbd.ttf','Courier New':'courbd.ttf'}


def caption_layer(width,height,text,font_name):
    if font_name not in FONTS:raise ValueError('Choose Arial, Times New Roman or Courier New.')
    text=str(text or '').strip()
    if not text:return None
    if len(text)>1000:raise ValueError('Keep the GIF caption under 1,000 characters.')
    padding=max(4,round(width*.025));available=max(1,width-padding*2)
    layer=Image.new('RGBA',(width,height));draw=ImageDraw.Draw(layer)
    path=Path(os.environ.get('WINDIR','C:/Windows'))/'Fonts'/FONTS[font_name]
    for size in range(max(10,min(42,round(height*.08))),7,-1):
        try:font=ImageFont.truetype(str(path),size)
        except OSError:font=ImageFont.load_default(size=size)
        lines=[]
        for paragraph in text.splitlines():
            current=''
            for word in paragraph.split():
                candidate=(current+' '+word).strip()
                if current and draw.textlength(candidate,font=font)>available:
                    lines.append(current);current=''
                for char in word:
                    candidate=current+char
                    if current and draw.textlength(candidate,font=font)>available:
                        lines.append(current);current=''
                    current+=char
                current+=' '
            lines.append(current.rstrip())
        wrapped='\n'.join(lines)
        spacing=max(2,size//6)
        box=draw.multiline_textbbox((0,0),wrapped,font=font,spacing=spacing,stroke_width=1)
        text_height=box[3]-box[1]
        if text_height+padding*2<=max(28,int(height*.45)):
            top=height-text_height-padding*2
            draw.rounded_rectangle((0,top,width-1,height-1),radius=padding,fill=(0,0,0,185))
            draw.multiline_text((width/2,(top+height)/2),wrapped,font=font,fill='white',stroke_width=1,stroke_fill='black',anchor='mm',align='center',spacing=spacing)
            return layer
    raise ValueError('The caption is too long for this GIF. Use fewer words.')


def indexed_frame(frame,width,height,overlay=None):
    image=frame.reformat(width,height,format='rgb24').to_image()
    if overlay is not None:image=Image.alpha_composite(image.convert('RGBA'),overlay).convert('RGB')
    indexed=image.quantize(colors=256,method=Image.Quantize.MEDIANCUT)
    colors=np.asarray(indexed.getpalette(),dtype=np.uint8).reshape(-1,3)
    palette=np.zeros((256,4),dtype=np.uint8)
    palette[:,0]=255
    palette[:len(colors),1:]=colors
    return av.VideoFrame.from_ndarray((np.asarray(indexed),palette),format='pal8')


def export(source,loop=True,caption='',font='Arial'):
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
        overlay=caption_layer(width,height,caption,font)
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
                sample=indexed_frame(frame,width,height,overlay)
                sample.pts=round(100*index/rate)
                sample.time_base=Fraction(1,100)
                dest.mux(video.encode(sample))
                count+=1
            if not count:raise ValueError('No video frames could be decoded.')
            dest.mux(video.encode())
    temporary.rename(target)
    return target
