import logging
import os

import neptune.new as neptune
from neptune.new import NeptuneUninitializedException
from neptune.new.integrations.python_logger import NeptuneHandler


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
        if v is None:
            return {'shape': None}
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


def get_run(namespace=None, local_monitoring=False, **kwargs):
    try:
        run = neptune.get_last_run()
    except NeptuneUninitializedException:
        if local_monitoring:
            kwargs['monitoring_namespace'] = f'{namespace}/monitoring'
        run = neptune.init(**kwargs)
        logging.getLogger().addHandler(NeptuneHandler(run=run))
    return Run(run, namespace=namespace)


class Run:
    def __init__(self, run, namespace=None):
        self.run = run
        self.namespace = namespace

    def __getitem__(self, path):
        if self.namespace is not None:
            path = f'{self.namespace}/{path}'
        return self.run[path]

    def __setitem__(self, path, value):
        if self.namespace is not None:
            path = f'{self.namespace}/{path}'
        self.run.__setitem__(path, value)

    def child(self, namespace, **kwargs):
        return get_run(f'{self.namespace}/{namespace}', **kwargs)
