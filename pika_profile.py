"""Shared validation and sampling policy for the two Pika H3 Flex tasks."""
from collections import Counter

PRESETS = ('pika_auto', 'pika', 'pika_refs')
SIGMAS = (1.0, 0.972973, 0.923077, 0.8, 0.571429, 0.333333, 0.0)
DOWNLOAD = 'Pika text / first-last models'
REFERENCE_DOWNLOAD = 'Pika image RefMod models'
DEFAULT_DOWNLOAD = 'Pika default models (text + image RefMods)'


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
        '**Pika Flex uses its trained six-step LCM schedule.** Extra LoRAs and the '
        'current upscale/refine pass are disabled. Pika selects its matching native '
        'INT8 checkpoint for this generation; your default model in Settings stays as saved. '
        'Start at 0.6 MP and 5 seconds on a 24 GB NVIDIA GPU. Smaller cards and '
        'longer clips need separate testing. Shorter clips are allowed here; Pika’s '
        'published duration range is 5–15 seconds. '
    )
    if preset == 'pika_auto':
        return common + (
            'Automatic chooses image RefMods when references are selected, otherwise text / first-last. '
            'Install Pika default models in Settings. Video/audio RefMods need Turbo or Fast Preview. '
            'First/last frames are unavailable when RefMods are selected.'
        )
    if preset == 'pika_refs':
        return common + (
            'Install the Pika image RefMod models in Settings and choose an image RefMod. '
            'This release supports image RefMods only; video and voice RefMods are '
            'blocked pending separate validation. First/last frames are unavailable '
            'in this mode. Generated audio still works. Semantic Bridge and VSA '
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
    if request.get('upscale'):
        raise ValueError('Pika currently supports its native output only. Turn off Upscale and refine video.')
    if request.get('loras'):
        raise ValueError('Pika cannot stack with extra LoRAs, including Turbo or Fast Preview adapters. Clear both LoRA slots.')
    profile = request.get('video_profile', {})
    if profile.get('active') != 'MiniMax H3 INT8':
        raise ValueError('Pika uses its matching native INT8 checkpoint. Choose Use Settings or MiniMax H3 INT8 for this generation.')
    if not 5 <= request.get('frames', 0) <= 362:
        raise ValueError('Choose a positive duration up to 15 seconds for Pika Flex. The app rounds up to the supported frame grid.')
    if preset == 'pika' and request.get('mods'):
        raise ValueError('Choose Pika References to use saved RefMods, or clear the RefMod selection.')
    if preset == 'pika_refs':
        if request.get('first_frame') or request.get('last_frame'):
            raise ValueError('Pika References uses RefMods. Clear the first/last frames or choose Pika text / first-last.')
        if not request.get('mods') or not any(value > 0 for value in request.get('strengths', [])):
            raise ValueError('Choose at least one visual RefMod at a strength above zero for Pika References.')


def validate_references(mods):
    active = [mod for mod, strength in mods if strength > 0]
    counts = Counter(mod.kind for mod in active)
    if counts['video'] or counts['audio']:
        raise ValueError('Pika RefMods currently supports image references only. Use Turbo or Fast Preview for video/audio RefMods; those Pika paths need separate validation.')
    if counts['image'] == 0:
        raise ValueError('Choose at least one active image RefMod for Pika References.')
    if len(active) > 9:
        raise ValueError('This Pika mode supports up to nine image references. Bundles count each member.')
