import os


def object_to_dict(o):
    result = {}
    for k, v in vars(o).items():
        try:
            result[k] = format_value(k, v)
        except ValueError:
            pass
    return result


def format_value(k, v):
    if k in ('features', 'in_reader', 'logger', 'out_writer', '_arguments', 'scenario', 'rng', 'rh', '_tae', 'solver'):
        raise ValueError('This value is not to be published in Neptune.')
    if k in ('algo_runs_timelimit', 'cutoff', 'memory_limit', 'ta_run_limit', 'wallclock_limit'):
        # These are float parameters that may have the value `math.inf` or `None`.
        return str(v)
    if k == 'cs':
        return configspace_to_dict(v)
    if k in ('train_insts', 'test_insts', 'val_set'):
        if len(v) >= 1 and isinstance(v[0], list):
            assert all(isinstance(r, list) and len(r) == 1 for r in v)
            v = [r[0] for r in v]
        if v == [None]:
            return {
                'n': 0,
                'commonpath': None
            }
        return {
            'n': len(v),
            'commonpath': os.path.commonpath(v)
        }
    if k == 'feature_dict':
        return {'n': len(v)}
    if k == 'feature_array':
        return {'shape': v.shape}
    return v


def configspace_to_dict(cs):
    if cs is None:
        return {}
    result = {
        'hyperparameters': cs.get_hyperparameters_dict(),
        'conditions': cs.get_conditions(),
        'forbidden_clauses': cs.get_forbiddens()
    }
    for k, v in list(result.items()):
        result[f'{k}_count'] = len(v)
    return result
