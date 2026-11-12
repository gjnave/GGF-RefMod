"""Shared validation and sampling policy for the two Pika H3 Flex tasks."""
from collections import Counter
import json
from pathlib import Path
from types import SimpleNamespace

PRESETS = ('pika_auto', 'pika', 'pika_refs')
SIGMAS = (1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0)
DOWNLOAD = 'Pika text / first-last models'
REFERENCE_DOWNLOAD = 'Pika multimodal RefMod models'
DEFAULT_DOWNLOAD = 'Pika default models (text + RefMods)'


def resolve(preset,references):
    return ('pika_refs' if references else 'pika') if preset=='pika_auto' else preset


def enabled(preset):
    return preset in PRESETS


def model_key(preset):
    return 'pika_base' if preset == 'pika_refs' else 'base'


def note(preset):
    if not enabled(preset):
        return ''
    common = (
        '**Pika Flex uses its trained six-step LCM schedule.** Compatible LoRAs, '
        'endpoint frames and the existing upscale/refine pass remain available. By default Pika selects its matching native '
        'INT8 checkpoint for this generation; your default model in Settings stays as saved. '
        'Start at 0.6 MP and 5 seconds on a 24 GB NVIDIA GPU. Smaller cards and '
        'longer clips need separate testing. Shorter clips are allowed here; Pika’s '
        'published duration range is 5–15 seconds. '
    )
    if preset == 'pika_auto':
        return common + (
            'Automatic chooses the reference task when RefMods are selected, otherwise text / first-last. '
            'Install Pika default models in Settings. Image, video, audio and mixed bundles are supported. '
            'All active members of selected bundles are used, even without a prompt mention. '
            'The linked GrizzlyMax Ref2VA hybrid can also be selected explicitly for the reference task.'
        )
    if preset == 'pika_refs':
        return common + (
            'Install the Pika multimodal RefMod models in Settings. Image, video, audio and mixed bundles '
            'use the existing native reference-latent pipeline. Generated audio remains enabled. Semantic Bridge and VSA '
            'are not applied by this preset.'
        )
    return common + (
        'Install the Pika text / first-last models in Settings. Use text or opening '
        'and ending frames here. Choose Pika References for saved RefMods.'
    )


def validate(request):
    preset = request.get('preset')
    if not enabled(preset):
        return
    profile = request.get('video_profile', {})
    if profile.get('active') not in ('MiniMax H3 INT8','GrizzlyMax INT4 Hybrid'):
        raise ValueError('Choose an available MiniMax H3 video model.')
    if profile.get('active')=='GrizzlyMax INT4 Hybrid' and preset=='pika':
        raise ValueError('The GrizzlyMax checkpoint is Ref2VA. Choose native INT8 for Pika text / first-last, or use the Pika reference task.')
    if not 5 <= request.get('frames', 0) <= 362:
        raise ValueError('Choose a positive duration up to 15 seconds for Pika Flex. The app rounds up to the supported frame grid.')
    if preset == 'pika' and any(value>0 for value in request.get('strengths',[])):
        raise ValueError('Choose Pika References to use saved RefMods, or clear the RefMod selection.')
    if preset == 'pika_refs':
        if not request.get('mods') or not any(value > 0 for value in request.get('strengths', [])):
            raise ValueError('Choose at least one active RefMod for the Pika reference task.')


def validate_references(mods):
    active = [mod for mod, strength in mods if strength > 0]
    counts = Counter(mod.kind for mod in active)
    if not active:raise ValueError('Choose at least one active RefMod.')
    if set(counts)-{'image','video','audio'}:raise ValueError('A RefMod has an unsupported reference type.')
    if counts['image']>9 or counts['video']>3 or counts['audio']>3 or len(active)>12:
        raise ValueError('Pika accepts up to 9 image, 3 video and 3 audio references, with 12 total. Bundles count each member.')


def reference_metadata(path):
    """Read the same embedded/legacy metadata as the loader, without tensor/GPU imports."""
    from safetensors import safe_open, SafetensorError
    path=Path(path)
    try:
        with safe_open(str(path),framework='numpy') as file:
            raw=file.metadata() or {}
            for key in ('refmod_meta','audio_refmod_meta'):
                if key in raw:return json.loads(raw[key])
    except (OSError,ValueError,SafetensorError):
        pass
    legacy=path.with_suffix('.json')
    if legacy.is_file():return json.loads(legacy.read_text(encoding='utf-8'))
    raise ValueError(f'{path.name} has no RefMod metadata.')


def preflight(request):
    """Reject incompatible references before a job or queue folder is created."""
    if len(request.get('mods',[]))!=len(request.get('strengths',[])):
        raise ValueError('Each selected RefMod needs a strength value. Refresh the selection and submit again.')
    active_paths=[path for path,strength in zip(request.get('mods',[]),request.get('strengths',[])) if strength>0]
    preset=resolve(request.get('preset'),active_paths)
    if not enabled(preset):return
    validate(dict(request,preset=preset))
    if preset!='pika_refs':return
    refs=[]
    for path,strength in zip(request.get('mods',[]),request.get('strengths',[])):
        if strength<=0:continue
        try:
            meta=reference_metadata(path)
            members=meta.get('members',[]) if meta.get('kind')=='bundle' else [meta]
            if meta.get('kind')=='bundle' and meta.get('_format_version')!=5:
                raise ValueError('Unsupported RefMod bundle format. Re-import it using the current RefMod creator.')
            if not members:raise ValueError('The RefMod bundle is empty.')
            refs.extend((SimpleNamespace(kind=member.get('kind')),strength) for member in members)
        except (OSError,ValueError,TypeError,AttributeError) as exc:
            raise ValueError(f'Cannot use RefMod {Path(path).name}: {exc}') from exc
    validate_references(refs)
