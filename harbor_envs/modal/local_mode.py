"""Modal single-H200 resource policy; evaluation protocols remain unchanged."""
from copy import deepcopy


def local_allocations(allocations: dict) -> dict:
    """CPU editing and trusted driver, one H200 for the isolated model service."""
    result = deepcopy(allocations)
    for role in ('agent', 'benchmark'):
        result[role].update(gpus=0, gpu_types=[])
    result['serve'].update(gpus=1, gpu_types=['H200'])
    return result


def configure_local_job(config: dict) -> None:
    """Fail before dispatch if native Harbor overrides break the GPU budget."""
    if config.get('n_concurrent_trials', 1) != 1:
        raise ValueError('local-mode requires n_concurrent_trials: 1')
    config['n_concurrent_trials'] = 1
    environment = config.setdefault('environment', {})
    if environment.get('override_gpus') not in (None, 0):
        raise ValueError('local-mode forbids GPU overrides for Agent/Benchmark')
    kwargs = environment.setdefault('kwargs', {})
    if kwargs.get('local_mode') is False:
        raise ValueError('local-mode cannot disable the Modal resource guard')
    kwargs['local_mode'] = True
