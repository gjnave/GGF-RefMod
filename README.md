# MiniMax RefMod

[![MiniMax RefMod](assets/ggf-refmod.png)](https://getgoingfast.pro/tools/ggf-refmod/)

A standalone Get Going Fast app for creating reusable MiniMax H3 references and generating video with native audio. Navy/gold GGF interface; app-owned inference worker; no ComfyUI installation, server, node editor, API account, or API key.

[Get Going Fast tool page and Windows quick installer](https://getgoingfast.pro/tools/ggf-refmod/)

## Manual Windows install

Use Windows x64 with an NVIDIA GPU and a driver compatible with CUDA 13.0. Install [Git for Windows](https://git-scm.com/downloads/win) and [64-bit Python 3.11](https://www.python.org/downloads/windows/) with the Python launcher. Open **Command Prompt** in a writable folder with enough space for the app, dependencies and models.

1. Clone this release from GitHub and enter its folder.

   ```bat
   git clone --branch v0.9.7 --depth 1 https://github.com/gjnave/GGF-RefMod.git Minimax-Refmod
   cd Minimax-Refmod
   ```

   The same release is on [Codeberg](https://codeberg.org/Cognibuild/MiniMax-RefMod-GGF). To use that mirror, substitute `https://codeberg.org/Cognibuild/MiniMax-RefMod-GGF.git` in the clone command.

2. Create the private Python environment and install the CUDA build of PyTorch and the app requirements.

   ```bat
   py -3.11 -m venv .venv
   .venv\Scripts\python.exe -m pip install torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
   .venv\Scripts\python.exe -m pip install -r requirements.txt
   .venv\Scripts\python.exe -m pip check
   ```

3. Check your GPU and the app, then launch it. For later launches, open Command Prompt in this same folder and repeat the last command.

   ```bat
   .venv\Scripts\python.exe app.py --self-check
   .venv\Scripts\python.exe app.py
   ```

4. Open **Settings** to download or link models. Model files are separate from the code. The default local address is `http://127.0.0.1:7900`. The application can choose another port if it is occupied. An optional [aria2c Windows x64 release](https://github.com/aria2/aria2/releases/tag/release-1.37.0) enables parallel downloads. Extract its folder inside `runtime/tools/` so the executable is at `runtime/tools/aria2-1.37.0-win-64bit-build1/aria2c.exe`. A manual install without aria2c uses Windows curl.

## Models

To use GrizzlyMax INT4, choose **Settings → Link and Use GrizzlyMax Models** and select its models folder. You can also paste the checkpoint path in the GrizzlyMax field and press Enter. A verified hybrid becomes the active video model. This selection applies to future jobs using the Settings default.

The RefMod creator set is approximately **5.81 GB**: video and audio VAEs. Creating only visual RefMods needs the video VAE; audio-only references need the audio VAE.

The complete set is approximately **47.56 GB**, including those VAEs, the FL2VA INT8 video model, Qwen3-VL-32B NVFP4 text encoder, Turbo adapter, learned latent upscaler, matching FL2VA DMD adapter, and refinement adapter. Additional disk is needed for dependencies, inputs, jobs, and outputs. Weights are not included in the ZIP.

The installer sets up aria2c 1.37.0 in the app runtime. Model downloads start in Settings. RefMod downloads one model at a time using up to four connections and shows the current file, downloaded bytes, speed, and estimated time remaining. **Reconnect and resume** opens fresh connections for the current file. **Stop model operation** saves progress for the next **Download missing models** run. Leave the partial file and its aria2 control file together.

Existing sequential curl partials can resume in aria2c. Older installs can use curl for sequential transfers until the updated installer is run. Each file is checked against its pinned size and SHA-256 in `models.lock.json` before being marked installed. Hash failures preserve the received file under an `.unverified-*` name.

For standard MiniMax models, choose **Settings → Browse and Use MiniMax Models** and select a folder. You can also paste a folder or a single checkpoint in the MiniMax field and press Enter. RefMod scans subfolders and links verified files in place. Linking a standard video checkpoint switches the active model to MiniMax H3 INT8. Linking only supporting files keeps the current video model.

The MiniMax H3 INT8 video model accepts the official FL2VA pruned (20.97 GB) and full (34.04 GB) ConvRot checkpoints. For example, selecting `M:\models` will find `M:\models\diffusion_models\Model_minimax\minimax_h3_fl2va_int8_convrot.safetensors`. The full checkpoint is an optional link; the standard model download uses the pruned checkpoint. Use the separate GrizzlyMax picker only for its INT4 hybrid.


## Create RefMod

Upload images/videos and optionally an audio clip. Full Reference retains encoded detail. Compressed Reference pools the latent representation and can refine that representation; it does not train the MiniMax diffusion model or create a LoRA. Exact facial identity can deteriorate with compression.

Choose a name and purpose such as identity, clothing, style, motion, or background. Advanced controls expose reference resolution, input-frame sampling, token budgets, pooling, latent-frame limits, audio duration/type, and separate-file versus bundle saving. Images are normalized using their displayed EXIF orientation. Videos are sampled over the clip rather than simply taking the opening frames. Embedded video audio is not implicitly extracted; provide an audio reference separately.

Each uploaded reference appears directly below its input. Images display inline; videos and audio have playback controls. For clips that need conversion for browser playback, the app shows a labeled preview of up to 15 seconds. Creation always uses the original file. The optional first and last frame inputs also show their selected images.

Saved files live under `models/refmods/`. Names include a unique job suffix to preserve earlier results. When importing a saved RefMod, the app shows its metadata before you add it to the library.

## Generate Video

Under **Generation preset**, choose **Turbo (8 steps)** for the default workflow, **Fast Preview (4 steps)** for a quicker draft, or **Custom steps** to choose a step count without a built-in speed adapter. Fast Preview uses the LightX2V FL2V v1.2 adapter with Euler, a simple schedule and video/audio shifts of 6/3. To install just this optional adapter, go to **Settings → Fast Preview adapter only (1.96 GB) → Download missing models**, or use **Browse and Use MiniMax Models** with the folder containing the file. It is separate from the standard 47.56 GB model set. Existing installations can reuse their video model, including the linked GrizzlyMax INT4 hybrid.

Open **LoRAs (optional)** to select up to two additional MiniMax H3 video adapters and set their strengths independently. Use **Browse for LoRA** and **Link LoRA** on the PC to reuse a local `.safetensors` file in place. Files already under `models/loras` appear with **Refresh LoRA list**. Choose **No LoRA** to clear a slot. The selected adapters affect both generation and optional refinement. A strength of zero skips the adapter. Choose **Custom steps** when using your own speed LoRA.

Choose up to eight RefMods. Beside **Scene and sound**, click a numbered reference shortcut to insert a phrase such as `The Character from <Picture 2>` at the cursor. The links follow your selection order and number pictures, videos, and audio separately. Include motion, camera direction, dialogue, SFX, and music. Optional first/last images provide keyframes. A shared strength and retention control adjust the chosen references.

Click **Generate Video** for the first submission. While it runs, change the prompt or settings and click **Add to queue**. Each submission saves its inputs and seed, and the app runs them in order on one GPU. Every finished clip appears in the thumbnail strip, newest first, including the clip in the main player. Click a thumbnail to select it. New clips enter the strip while your selected video stays in the player.

The **Generation queue** box beneath the clip thumbnails shows the running submission and waiting positions. Use **Remove** on a waiting row or **Stop current job** to stop the active worker. Open **Recent submissions** for the last 20. Closing the browser leaves the queue running. After an app restart, waiting submissions resume and an interrupted generation is marked stopped.

Double-click a finished thumbnail or the main video, or select **View project details**. This shows the prompt, model, seed, output length, dimensions, and timing. New clips record individual render times; job time excludes time spent waiting in the queue. Older clips show the information that is available. **Reuse these settings** fills in the form for one clip using its actual seed. Review or edit the form, then click **Generate Video** yourself. **Close details** leaves the form alone.

**Thumbnails shown** sets the strip size from 1 to 20. The browser remembers your choice. The newest clips fill the available places automatically. Click **×** on a thumbnail or **× Delete current video** under the player and confirm to send that MP4 directly to the Windows Recycle Bin. Use **Restore** in the Recycle Bin to return it to the gallery. The **Settings → Output gallery → Send deleted videos to the Windows Recycle Bin** checkbox applies only to Refmod. Uncheck it if you prefer to move clips to `outputs/Deleted videos/`.

Turn on **Loop playback** to repeat the selected video in the main player. The player remembers this setting when you switch clips or refresh.

To export the selected video, leave **Loop GIF continuously** checked for an endlessly repeating GIF, or uncheck it for a GIF that plays once. Click **Convert to GIF**, then **Preview GIF** or **Download GIF**. GIFs contain the full clip without audio and are saved in `outputs/GIFs/` at up to 12 frames per second and 640 pixels on the longest side. The original video stays in the gallery.

Turbo uses the supplied workflow's FL2V eight-step adapter, `res_multistep` sampler, simple schedule, and 12/3 video/audio sigma shifts. The seed defaults to **Random**. Leave it blank for a fresh seed on every Generate, or enter a number to reuse it across runs and reloads. The generated seed appears in the result details. Sequential videos increment the starting seed within a batch.

Duration snaps upward to `17k+5` frames at 24 fps. For example, 3 seconds becomes 73 frames (3.04 s); 6 seconds becomes 158 frames (6.58 s). Long durations are accepted but are not a promise of quality or memory compatibility outside the model's trained range.

Optional upscale uses the same learned LBH latent-upscaler weights through the native Comfy inference implementation. It applies a matching FL2VA DMD adapter and the 2K refiner to a fresh base, then uses the final requested steps of the eight-step schedule. Audio is masked during refinement and the original audio latent is retained. Default upscale target is 1.2 megapixels; this is an area target, not a promise of exactly 2× width and height.

The supplied graph's disconnected refinement branch and incompatible DMD variant are explained in `WORKFLOW-ANALYSIS.md`. Experimental sparse attention and graphical monitoring/debug nodes are not required by this app.

## Files and controls

- `outputs/`: generated MP4 files, each with a unique filename.
- `models/refmods/`: reusable RefMod files.
- `jobs/<job-id>/`: durable input copies, request JSON, final status, and worker log.
- `jobs/workspaces/`: browser-workspace state. Reopening the same origin restores the last job and selected generation controls. Browser storage identifies the workspace; changing origin creates a different workspace.
- `settings.json`: phone access and gallery settings; keep private.
- `model-links.json`: local model paths; keep private.

Only one worker runs at a time across the local and phone interfaces. Stop terminates only that job's owned process tree. Each inference worker exits after its job, releasing its models. A failed job preserves its input files and diagnostic log. Ctrl+Enter invokes the visible Create/Generate action. Stop server requires confirmation and refuses while a job is active.

Phone access is optional and off by default. Settings can enable a separate LAN interface on ports 7910–7919, with an optional username/password. A blank password disables remote authentication. The local loopback interface stays login-free. Model management is restricted to the local interface. The PC performs all generation and must stay on. No public tunnel is started.

## Upstream sources

- [ComfyUI](https://github.com/Comfy-Org/ComfyUI), commit `1332fabfcc507aa7c548cd5152071ed494c207e8` — GPL-3.0.
- [ComfyUI-MiniMaxH3Mod](https://github.com/Luisacaotica/ComfyUI-MiniMaxH3Mod), commit `517e8c419a8584fbf69fceb487bc39d83e6a6ec8` — MIT.
- [MiniMax H3 model files](https://huggingface.co/Comfy-Org/MiniMax-H3).
- [Latent upscaler](https://huggingface.co/LBH-123-AI/Minimax_h3_latent_Upscaler).
- [DMD adapters](https://huggingface.co/drbaph/MiniMax-H3-Turbo-Lora-ComfyUI).
- [LightX2V Fast Preview adapter and recommended settings](https://huggingface.co/lightx2v/Minimax-h3-Turbo/discussions/52).
- [Refiner](https://huggingface.co/Kijai/MiniMax-H3-experimental).

See `THIRD-PARTY-NOTICES.md` and bundled upstream licenses. Model licenses remain separate.
