"""Upload previews only. The creator always receives the unchanged original files."""
from __future__ import annotations

from fractions import Fraction
from html import escape
from io import BytesIO
from pathlib import Path
from urllib.parse import quote

import av
from gradio.processing_utils import save_bytes_to_cache
from PIL import Image

PICTURES = ('.png', '.jpg', '.jpeg', '.webp', '.bmp')
VIDEOS = ('.mp4', '.mov', '.webm', '.mkv', '.avi')
AUDIO = ('.wav', '.mp3', '.m4a', '.flac', '.ogg')
CLIP_SECONDS = 15


def url(block, path):
    data = block.serve_static_file(str(path))
    prefix, address = data['url'].split('file=', 1)
    return escape(prefix + 'file=' + quote(address.replace('\\', '/'), safe='/:'))


def video_clip(source, vs, sound):
    """Encode a bounded H.264/AAC excerpt when the upload is not browser-ready."""
    buf = BytesIO()
    scale = min(1, 640 / max(vs.width, vs.height))
    with av.open(buf, 'w', format='mp4') as dest:
        rate = min(vs.average_rate or Fraction(24), Fraction(30))
        video = dest.add_stream('libx264', rate=rate, options={'preset': 'veryfast'})
        video.width, video.height = [max(2, int(n * scale) // 2 * 2) for n in (vs.width, vs.height)]
        video.pix_fmt = 'yuv420p'
        if sound:
            audio = dest.add_stream('aac', rate=48000)
            audio.layout = 'stereo'
            resampler = av.AudioResampler(format='fltp', layout='stereo', rate=48000)
        streams = [vs, sound] if sound else [vs]
        starts, done, last = {}, set(), -1
        for packet in source.demux(streams):
            stream = packet.stream
            if stream.index in done:
                continue
            for frame in packet.decode():
                if frame.time is None:
                    continue
                t = frame.time - starts.setdefault(stream.index, frame.time)
                if t >= CLIP_SECONDS:
                    done.add(stream.index)
                    break
                if stream == vs:
                    tick = round(t * rate)
                    if tick <= last:
                        continue
                    last = tick
                    frame = frame.reformat(video.width, video.height, 'yuv420p')
                    frame.time_base, frame.pts = 1 / rate, tick
                    dest.mux(video.encode(frame))
                else:
                    for samples in resampler.resample(frame):
                        samples.pts = None
                        dest.mux(audio.encode(samples))
            if len(done) == len(streams):
                break
        if last < 0:
            raise ValueError('No decodable video frames.')
        dest.mux(video.encode())
        if sound:
            for samples in resampler.resample(None):
                samples.pts = None
                dest.mux(audio.encode(samples))
            dest.mux(audio.encode())
    return buf.getvalue()


def player_source(path, kind, block):
    with av.open(str(path)) as source:
        sound = next(iter(source.streams.audio), None)
        if kind == 'audio':
            if not sound:
                raise ValueError('No audio track.')
            if (path.suffix.lower(), sound.codec_context.name) in (
                ('.wav','pcm_s16le'), ('.mp3','mp3'), ('.mp3','mp3float'),
                ('.m4a','aac'), ('.ogg','opus'), ('.ogg','vorbis'), ('.flac','flac')
            ):
                return path, ''
            # Other codecs in an accepted audio container get a playable WAV excerpt.
            buf = BytesIO()
            with av.open(buf, 'w', format='wav') as dest:
                out = dest.add_stream('pcm_s16le', rate=44100)
                out.layout = 'stereo'
                resampler = av.AudioResampler(format='s16', layout='stereo', rate=44100)
                count = 0
                for frame in source.decode(sound):
                    for chunk in resampler.resample(frame):
                        chunk.pts = None
                        dest.mux(out.encode(chunk))
                        count += chunk.samples
                    if count >= 44100 * CLIP_SECONDS:
                        break
                if not count:
                    raise ValueError('No decodable audio samples.')
                dest.mux(out.encode())
            data, ext = buf.getvalue(), 'wav'
        else:
            vs = next(iter(source.streams.video), None)
            if not vs:
                raise ValueError('No video track.')
            name = vs.codec_context.name
            audio_codec = sound.codec_context.name if sound else None
            if ((path.suffix.lower() == '.mp4' and name == 'h264' and vs.codec_context.pix_fmt in ('yuv420p','yuvj420p') and audio_codec in (None,'aac','mp3','mp3float'))
                    or (path.suffix.lower() == '.webm' and name in ('vp8','vp9') and audio_codec in (None,'opus','vorbis'))):
                return path, ''
            data, ext = video_clip(source, vs, sound), 'mp4'
        cached = save_bytes_to_cache(data, 'preview.' + ext, cache_dir=block.GRADIO_CACHE)
        return Path(cached), f' · Preview: up to {CLIP_SECONDS} seconds'


def render(files, block, audio=False):
    cards = []
    for file in files or []:
        path = Path(file)
        caption = escape(path.name)
        try:
            suffix = path.suffix.lower()
            if not audio and suffix in PICTURES:
                with Image.open(path) as pic:
                    pic.verify()
                media = f'<img src="{url(block, path)}" alt="{caption}">'
            else:
                if suffix not in (AUDIO if audio else VIDEOS):
                    raise ValueError('Unsupported file type.')
                tag = 'audio' if audio else 'video'
                playable, note = player_source(path, tag, block)
                media = f'<{tag} controls playsinline preload="auto" src="{url(block, playable)}"></{tag}>'
                caption += note
        except Exception as err:
            media = f'<p role="status">Could not preview this file: {escape(str(err))}</p>'
        cards.append(f'<figure>{media}<figcaption>{caption}</figcaption></figure>')
    return '<div class="upload-previews">' + ''.join(cards) + '</div>' if cards else ''
