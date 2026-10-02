"""The routine end to end, in a lyse worker process started as lyse starts one."""

import warnings

import pytest

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import lyse.analysis_subprocess
    from labscript_utils.ls_zprocess import ProcessTree
    from lyse.routine import RoutineSettings
    from lyse.utils import LYSE_DIR

from test_worker import CONFIG

ROUTINE = '''
from labscript_optimization.routine import OptimizationRoutine


class Runmanager:
    def __init__(self, config):
        pass

    def check_ready(self):
        pass

    def shot_status(self, shot_ids):
        return {}


class Optimization(OptimizationRoutine):
    config_path = 'config.toml'
    interface_factory = Runmanager
'''


@pytest.fixture
def worker(tmp_path):
    routine = tmp_path / 'optimization.lyse'
    routine.mkdir()
    (routine / 'lyse_routine.py').write_text(ROUTINE)
    (routine / 'config.toml').write_text(CONFIG)
    settings = RoutineSettings(routine, lyse.analysis_subprocess.config_dir)
    to_worker, from_worker, process = ProcessTree.instance().subprocess(
        str(LYSE_DIR / 'analysis_subprocess.py'), startup_timeout=30
    )
    to_worker.put(str(routine))

    def analyse(path, paths):
        to_worker.put(['analyse', (path, paths)])
        return from_worker.get(timeout=60)

    yield analyse
    process.kill()
    process.wait()
    settings.path.unlink(missing_ok=True)


def test_the_routine_runs_multishot_passes_and_refuses_singleshot_ones(
    worker, tmp_path
):
    assert worker(None, []) == ['done', {}]
    assert worker(str(tmp_path / 'shot.h5'), None) == ['error', {}]
