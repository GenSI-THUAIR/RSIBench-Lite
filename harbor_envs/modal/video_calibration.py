"""Bind separately calibrated Modal video references without changing the source protocol."""
from copy import deepcopy


def apply_video_calibration(scoring: dict, calibration: dict) -> dict:
    """Only reference identity, calibrated quality limits and provenance may differ."""
    if scoring['gate']['kind'] != 'video-reference':
        raise ValueError('Modal video calibration requires a video-reference task')
    if set(calibration) - {'gate', 'video_benchmark'}:
        raise ValueError('Unknown Modal video calibration fields')
    gate = calibration.get('gate', {})
    video = calibration.get('video_benchmark', {})
    if set(gate) - {'data', 'source'}:
        raise ValueError('Video calibration cannot change the correctness requirement')
    if set(video) - {'identity', 'reference_manifest_sha256', 'calibration', 'quality'}:
        raise ValueError('Video calibration cannot change the workload')
    result = deepcopy(scoring)
    result['gate'].update(deepcopy(gate))
    # Identity and receipt are complete records, never partial merges with H20 provenance.
    for key, value in video.items():
        result['video_benchmark'][key] = deepcopy(value)
    old, new = scoring['video_benchmark'], result['video_benchmark']
    media_keys = {key for key in old['quality'] if key.startswith('expected_')}
    media_keys.update(('duration_tolerance_sec', 'timestamp_tolerance_sec'))
    if any(new['quality'].get(key) != old['quality'][key] for key in media_keys):
        raise ValueError('Video calibration cannot change the media contract')
    return result


def validate_hardware(expected: dict, gpu: str, count: int, observed: str | None = None) -> None:
    """Reject a different GPU/driver before comparing against a hardware-bound reference."""
    if expected['model'] != f'NVIDIA {gpu}' or expected['count'] != count:
        raise ValueError('Video reference hardware differs from the requested Modal GPU')
    if observed is not None:
        rows = [row.split(',') for row in observed.strip().splitlines()]
        if (len(rows) != count or any(row[0].strip() != expected['model']
                or row[-1].strip() != expected['driver'] for row in rows)):
            raise ValueError('Video reference hardware/driver differs from the actual allocation')
