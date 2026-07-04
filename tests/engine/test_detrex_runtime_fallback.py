from canopyrs.engine.models.detector.runtime_fallback import (
    ALLOW_DETREX_CPU_FALLBACK_ENV,
    build_detrex_cuda_support_error,
    is_detrex_gpu_support_error,
    should_fallback_detrex_to_cpu,
)


def test_detects_detrex_gpu_support_error():
    error = RuntimeError("Not compiled with GPU support")

    assert is_detrex_gpu_support_error("dino_detrex", "cuda", error)


def test_does_not_detect_detrex_gpu_support_error_when_conditions_do_not_match():
    error = RuntimeError("some other runtime failure")

    assert not is_detrex_gpu_support_error("dino_detrex", "cuda", error)
    assert not is_detrex_gpu_support_error("dino_detrex", "cpu", RuntimeError("Not compiled with GPU support"))
    assert not is_detrex_gpu_support_error(
        "faster_rcnn_detectron2",
        "cuda",
        RuntimeError("Not compiled with GPU support"),
    )


def test_only_falls_back_to_cpu_when_explicitly_enabled(monkeypatch):
    error = RuntimeError("Not compiled with GPU support")

    monkeypatch.delenv(ALLOW_DETREX_CPU_FALLBACK_ENV, raising=False)
    assert not should_fallback_detrex_to_cpu("dino_detrex", "cuda", error)

    monkeypatch.setenv(ALLOW_DETREX_CPU_FALLBACK_ENV, "1")
    assert should_fallback_detrex_to_cpu("dino_detrex", "cuda", error)


def test_builds_actionable_detrex_cuda_support_error_message():
    message = build_detrex_cuda_support_error()

    assert "FORCE_CUDA=1" in message
    assert ALLOW_DETREX_CPU_FALLBACK_ENV in message