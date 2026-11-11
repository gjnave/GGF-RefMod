# Third-party notices

The inference core under `vendor/comfy_core` is selected source from ComfyUI, commit `1332fabfcc507aa7c548cd5152071ed494c207e8`, licensed under GNU GPL version 3. Its license is retained at `vendor/comfy_core/LICENSE`. The app code is distributed under the same GPL-3.0 license in `LICENSE`.

Included are the model loaders, model implementations required by their dynamic imports, API types, execution helpers, selected sampling/video/upscale nodes, and supporting utilities. `comfy_extras/nodes_audio.py` contains the unchanged `vae_decode_audio` function extracted from that pinned source, plus its torch import. The server-dependent audio UI nodes are omitted. The ComfyUI server, frontend, manager, and launch entry point are not shipped or invoked.

RefMod source under `vendor/refmod` is from Luisacaotica/ComfyUI-MiniMaxH3Mod, commit `517e8c419a8584fbf69fceb487bc39d83e6a6ec8`. Its MIT license and upstream README are retained there. Its route definitions may be imported as ordinary code; registration is skipped because no ComfyUI server exists. No upstream browser JavaScript is shipped.

Model weights are downloaded separately from their publishers. Their licenses are independent of the Python source licenses. The pinned repositories, revisions, filenames, sizes, and hashes are in `models.lock.json`.

Pika H3 Flex adapters are published by Pika Labs and distributed in the pinned Comfy-Org/MiniMax-H3 repository. See https://huggingface.co/Pika-Labs/Pika-H3-Flex and its model license. The optional Pika presets follow the native six-step LCM schedule. No third-party workflow extensions, sparse-attention gates or Semantic Bridge code are bundled.

INSTALL.bat retrieves the official [aria2 1.37.0 Windows x64 release](https://github.com/aria2/aria2/releases/tag/release-1.37.0) and verifies its archive SHA-256 before extraction. The upstream license and documentation accompany the executable under `runtime/tools`.

Get Going Fast branding and the supplied about/disclaimer assets belong to their respective owner. They do not alter the upstream code or model licenses.
