# Extracted unchanged from pinned ComfyUI nodes_audio.py; GPL-3.0. No server import.
import torch

def vae_decode_audio(vae, samples, tile=None, overlap=None):
    latent = samples["samples"]
    if latent.is_nested:
        latent = latent.unbind()[-1]

    if tile is not None:
        audio = vae.decode_tiled(latent, tile_x=tile, tile_y=tile, overlap=overlap).movedim(-1, 1)
    else:
        audio = vae.decode(latent).movedim(-1, 1)

    std = torch.std(audio, dim=[1, 2], keepdim=True) * 5.0
    std[std < 1.0] = 1.0
    audio /= std
    vae_sample_rate = getattr(vae, "audio_sample_rate_output", getattr(vae, "audio_sample_rate", 44100))
    return {"waveform": audio, "sample_rate": vae_sample_rate if "sample_rate" not in samples else samples["sample_rate"]}
