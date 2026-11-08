# What the two supplied workflows do

## minimax_h3_ref_mod_creator.json

This is the RefMod **creator**. A folder loader reads image/video references, and `MiniMaxH3RefModMasterExtract` encodes them using the H3 video VAE. It is configured for Full Reference, a clothing concept named `zombie_clothing`, 2048px reference resolution, and a 5120-token budget. Saving is enabled with separate files. Inspection nodes report the contents/reference map and show a preview.

The folder `G:/motion` is an example machine-local input path. It is not a required destination and is not used by the app. The input files are not included in the JSON.

An audio VAE is connected, but the **audio input is disconnected**, so this saved graph is not creating an audio RefMod. The master node supports audio when an audio source is supplied. Pooling and refinement-step controls do not affect Full Reference in the same way they affect Compressed Reference.

A RefMod is an encoded reference representation saved to `.safetensors`. It is not a separately trained diffusion model or a LoRA. Full Reference can retain substantial latent data; the much smaller compressed mode trades detail for token cost.

## minimax_with_upscale.json

This is the RefMod **usage/video-generation** workflow. Most of it is inside a subgraph.

1. Load the FL2VA pruned INT8 H3 diffusion model, Qwen3-VL-32B NVFP4 text encoder, and video/audio VAEs.
2. Encode the text prompt and prepare a joint video/audio latent. Optional first/last-frame sockets are empty in the supplied graph.
3. Load two existing RefMods, `zombie_visual` and `zombie_clothing_visual`, apply them to conditioning, and sample video plus audio.
4. Separate the generated video/audio latents. Upscale the video latent to a 1.2-megapixel target. Mask the original audio latent so it is not re-noised.
5. Recombine, sample another pass, decode video/audio, and save the final video.

The example prompt assigns identity and wardrobe to two different video references. Those RefMods themselves are not included. The app starts with a blank prompt and an empty library instead of pretending these example assets are installed.

The active resolution selector is **portrait 9:16 at 0.5 megapixels**. Its wires override the 1344×768 values stored on the subgraph. Duration is six nominal seconds, rounded to the H3 frame grid. Turbo is on with eight steps.

## Problems in the supplied graph

- Nodes 153 → 151 load DMD and refiner LoRAs, but the output of node 151 has **no connection to the sampler**. The actual second-pass guider gets its model from nodes 152 → 154. The LoRA refinement branch therefore has no effect.
- The selected DMD file is the pruned **Ref2VA** variant while the loaded base is **FL2VA**. These variants should be matched, as documented by the DMD adapter source.
- The group title says “2× + 2 Steps,” but the upscaler is set to **1.2 megapixels**, and the second sampler receives the **same schedule as the first sampler**. With Turbo enabled here, that is eight steps, not two.
- The ordinary first-pass decoding/output nodes are bypassed; the saved output is the second-pass video.
- Monitoring, VRAM-debug, and experimental attention nodes are auxiliary tools, not requirements for a standalone interface.

The app preserves the core create/use pipeline while connecting the refinement adapters, selecting the matching FL2VA DMD variant, and explicitly exposing upscale area and refinement-step controls. This is a deliberate corrected implementation, not a claim that the supplied graph already performed those corrected operations. Its final generation quality still requires a real model run.
