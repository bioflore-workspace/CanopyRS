import os


DETREX_GPU_SUPPORT_ERROR = "Not compiled with GPU support"
ALLOW_DETREX_CPU_FALLBACK_ENV = "CANOPYRS_ALLOW_DETREX_CPU_FALLBACK"


def is_detrex_gpu_support_error(configured_model: str, device_type: str, error: BaseException) -> bool:
    return (
        configured_model.endswith("detrex")
        and device_type == "cuda"
        and DETREX_GPU_SUPPORT_ERROR in str(error)
    )


def should_fallback_detrex_to_cpu(configured_model: str, device_type: str, error: BaseException) -> bool:
    return is_detrex_gpu_support_error(configured_model, device_type, error) and os.getenv(
        ALLOW_DETREX_CPU_FALLBACK_ENV, ""
    ).lower() in {"1", "true", "yes", "on"}


def build_detrex_cuda_support_error() -> str:
    return (
        "detrex was selected for CUDA inference, but its native extension was built without GPU kernels. "
        "Reinstall CanopyRS on this host with the CUDA toolkit available so detectron2 and detrex compile with "
        "FORCE_CUDA=1, or rerun commands/install_canopyrs.sh after confirming nvcc is on PATH. "
        "Set CANOPYRS_ALLOW_DETREX_CPU_FALLBACK=1 only if you explicitly want to accept CPU inference."
    )