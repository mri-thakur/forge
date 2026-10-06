import pytest

from forge.thermal import Nvml, ThermalGuard


def scripted(temperatures):
    readings = iter(temperatures)
    return lambda: next(readings)


def test_cool_gpu_never_pauses():
    sleeps = []
    guard = ThermalGuard(85, 75, scripted([60, 84]), sleep=sleeps.append, log=lambda _: None)
    assert guard.wait_if_hot() == 0.0
    assert guard.wait_if_hot() == 0.0
    assert sleeps == [] and guard.pauses == 0


def test_hot_gpu_pauses_until_it_cools_to_the_resume_temperature():
    sleeps, logs = [], []
    guard = ThermalGuard(85, 75, scripted([87, 84, 78, 75]), sleep=sleeps.append, log=logs.append)
    guard.wait_if_hot()
    assert len(sleeps) == 3  # 84 and 78 are still above 75; 75 resumes
    assert guard.pauses == 1
    assert '"thermal_pause"' in logs[0] and '"thermal_resume"' in logs[1]


def test_unreadable_sensor_never_blocks_training():
    sleeps = []
    guard = ThermalGuard(85, 75, scripted([None, 90, None]), sleep=sleeps.append, log=lambda _: 0)
    assert guard.wait_if_hot() == 0.0
    guard.wait_if_hot()  # 90 pauses, then the sensor fails: resume rather than wait forever
    assert len(sleeps) == 1


def test_disabled_off_cuda_or_by_environment(monkeypatch):
    assert ThermalGuard.for_device("cpu") is None
    monkeypatch.setenv("FORGE_GPU_TEMP_LIMIT", "0")
    assert ThermalGuard.for_device("cuda") is None


def test_reads_a_plausible_gpu_temperature():
    try:
        nvml = Nvml()
    except OSError:
        pytest.skip("no NVIDIA driver")
    assert 10 <= nvml.temperature() <= 110
