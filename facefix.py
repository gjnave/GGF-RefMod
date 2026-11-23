"""Refine a tracked face with H3 and mux the original audio packets into a new MP4."""
from pathlib import Path
import uuid

import av
import cv2
import numpy as np

from config import APP, OUTPUTS, model_name, require_models
import video_profile

SIZE = 384
CHUNK = 22
DETECTOR = APP / 'assets' / 'face_detection_yunet_2023mar.onnx'


class FaceTrack:
    def __init__(self):
        self.detector = cv2.FaceDetectorYN.create(str(DETECTOR), '', (320, 320), 0.70)
        self.previous = None
        self.thumbnail = None

    def locate(self, rgb):
        height, width = rgb.shape[:2]
        thumb = cv2.resize(rgb, (32, 32), interpolation=cv2.INTER_AREA)
        cut = self.thumbnail is not None and cv2.absdiff(thumb, self.thumbnail).mean() > 42
        self.thumbnail = thumb
        if cut:
            self.previous = None
        factor = min(1, 960 / max(width, height))
        small = cv2.resize(rgb, (max(1, round(width*factor)), max(1, round(height*factor))))
        self.detector.setInputSize((small.shape[1], small.shape[0]))
        _, faces = self.detector.detect(cv2.cvtColor(small, cv2.COLOR_RGB2BGR))
        if faces is None:
            return None, cut
        boxes = []
        for x, y, w, h in faces[:, :4] / factor:
            boxes.append(np.array([x+w/2, y+h/2, max(w, h)], dtype=float))
        if self.previous is None:
            face = max(boxes, key=lambda box: box[2])
        else:
            previous = self.previous
            boxes = [b for b in boxes if 0.5 < b[2]/previous[2] < 2 and
                     np.linalg.norm(b[:2]-previous[:2]) < previous[2]*0.85]
            if not boxes:
                return None, cut
            nearest = min(boxes, key=lambda b: np.linalg.norm(b[:2]-previous[:2]))
            face = previous*0.25 + nearest*0.75
        self.previous = face
        side = min(width, height, max(16, round(face[2]*1.65)))
        x = int(np.clip(round(face[0]-side/2), 0, width-side))
        y = int(np.clip(round(face[1]-side/2), 0, height-side))
        return (x, y, side), cut


class H3Refiner:
    def __init__(self, engine):
        self.torch, self.nodes, _, self.h3, self.sampling, _, _ = engine
        profile = video_profile.settings()
        hybrid = profile['active'] == video_profile.HYBRID
        require_models(['encoder', 'video_vae', 'turbo'] + ([] if hybrid else ['base']))
        if hybrid:
            import comfy.sd
            base = comfy.sd.load_diffusion_model(str(video_profile.linked_path(profile)))
        else:
            base = self.nodes.UNETLoader().load_unet(model_name('base'), 'default')[0]
        model = self.nodes.LoraLoaderModelOnly().load_lora_model_only(base, model_name('turbo'), 1.0)[0]
        self.model = self.h3.MiniMaxH3SigmaShift.execute(model, 12.0, 3.0)[0]
        clip = self.nodes.CLIPLoader().load_clip(model_name('encoder'), 'minimax', 'default')[0]
        self.vae = self.nodes.VAELoader().load_vae(model_name('video_vae'))[0]
        self.positive = clip.encode_from_tokens_scheduled(clip.tokenize(
            'A detailed face, consistent facial features, natural eyes and skin texture, original expression.'))

    def refine(self, crops):
        from comfy.nested_tensor import NestedTensor
        torch = self.torch
        count = len(crops)
        length = self.h3.align_frame_count(max(5, count))
        padded = crops + [crops[-1]]*(length-count)
        with torch.no_grad():
            pixels = torch.from_numpy(np.stack(padded)).float().div_(255)
            v = self.vae.encode(pixels)
            latent, _ = self.h3._empty_av_latent(SIZE, SIZE, length)
            _, a = latent['samples'].unbind()
            latent['samples'] = NestedTensor((v, a))
            latent['noise_mask'] = NestedTensor((torch.ones_like(v), torch.zeros_like(a)))
            sample = self.sampling
            schedule = sample.BasicScheduler.execute(self.model, 'simple', 8, 1.0)[0][-3:]
            result = sample.SamplerCustomAdvanced.execute(
                sample.RandomNoise.execute(17)[0], sample.BasicGuider.execute(self.model, self.positive)[0],
                sample.KSamplerSelect.execute('res_multistep')[0], schedule, latent)[1]
            import comfy.model_management as mm
            mm.unload_all_models()
            decoded = self.nodes.VAEDecode().decode(self.vae, result)[0]
            if len(decoded) < count:
                raise ValueError('H3 decoded fewer frames than the source crop.')
            return decoded[:count].clamp(0, 1).mul(255).byte().cpu().numpy()


def run(request, engine, emit):
    track = FaceTrack()
    refiner = None
    frames = fixed = 0
    pending = []
    job = Path(request['job_dir'])
    tmp = job / ('facefix-' + uuid.uuid4().hex + '.mp4')
    target = OUTPUTS / (job.name + '-FaceFix-' + uuid.uuid4().hex[:8] + '.mp4')
    axis = np.linspace(-1, 1, SIZE)
    radius = np.sqrt((axis[None, :]/0.90)**2 + (axis[:, None]/0.97)**2)
    mask = np.clip((1-radius)/0.28, 0, 1).astype('float32')
    with av.open(request['media'][0]) as src, av.open(str(tmp), 'w') as dest:
        if not src.streams.video:
            raise ValueError('Choose an MP4 with a video stream.')
        video = src.streams.video[0]
        out = dest.add_stream('libx264', rate=video.average_rate or 24, options={'crf':'16','preset':'fast'})
        out.width, out.height = video.width, video.height
        out.pix_fmt = 'yuv420p' if not (video.width%2 or video.height%2) else 'yuv444p'
        out.codec_context.time_base = video.time_base
        audios = {s.index:dest.add_stream_from_template(s) for s in src.streams.audio}

        def write(rgb, frame):
            nonlocal frames
            encoded = av.VideoFrame.from_ndarray(rgb, format='rgb24')
            encoded.pts, encoded.time_base = frame.pts, frame.time_base
            encoded.duration = frame.duration
            for packet in out.encode(encoded):
                dest.mux(packet)
            frames += 1

        def flush():
            nonlocal refiner, fixed
            if not pending:
                return
            if refiner is None:
                emit('stage', message='Loading H3 face refinement models')
                refiner = H3Refiner(engine)
            emit('stage', message=f'Refining faces at frame {frames+1}')
            crops = [cv2.resize(rgb[y:y+s,x:x+s], (SIZE,SIZE), interpolation=cv2.INTER_CUBIC)
                     for rgb, _, (x,y,s) in pending]
            for (rgb, frame, (x,y,s)), refined in zip(pending, refiner.refine(crops)):
                area = rgb[y:y+s, x:x+s]
                alpha = 0.65 * cv2.resize(mask, (s, s))[:, :, None]
                source = area.astype('float32')
                detail = source - cv2.GaussianBlur(source, (0, 0), 1.0)
                repair = cv2.resize(refined, (s, s)).astype('float32') + detail
                area[:] = np.clip(np.rint(source*(1-alpha) + repair*alpha), 0, 255).astype('uint8')
                write(rgb, frame)
                fixed += 1
            pending.clear()

        for packet in src.demux(video, *src.streams.audio):
            if packet.stream.index in audios:
                if packet.dts is not None:
                    packet.stream = audios[packet.stream.index]
                    dest.mux(packet)
                continue
            for frame in packet.decode():
                rgb = frame.to_ndarray(format='rgb24')
                box, cut = track.locate(rgb)
                if cut or box is None:
                    flush()
                if box is None:
                    write(rgb, frame)
                else:
                    pending.append((rgb, frame, box))
                    if len(pending) == CHUNK:
                        flush()
        flush()
        if not fixed:
            raise ValueError('No trackable face was detected in this clip.')
        for packet in out.encode():
            dest.mux(packet)
    tmp.rename(target)
    return dict(video=str(target), files=[str(target)], details=f'FaceFix refined {fixed} of {frames} frames.')
