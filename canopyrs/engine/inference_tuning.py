import os

import torch


DEFAULT_GPU_TARGET_UTILIZATION = 0.9
DEFAULT_GPU_MEMORY_RESERVE_GIB = 6.0
DEFAULT_MAX_DETECTOR_BATCH_SIZE = 8
DEFAULT_MAX_SEGMENTER_IMAGE_BATCH_SIZE = 4
DEFAULT_MAX_SEGMENTER_BOX_BATCH_SIZE = 512


def auto_scale_pipeline_inference(config, log=print):
    if _env_flag("CANOPYRS_DISABLE_AUTO_BATCH"):
        log("Automatic GPU batch tuning disabled by CANOPYRS_DISABLE_AUTO_BATCH.")
        return config

    if not torch.cuda.is_available():
        return config

    try:
        free_bytes, total_bytes = torch.cuda.mem_get_info()
    except (AssertionError, RuntimeError):
        return config

    free_gib = _bytes_to_gib(free_bytes)
    total_gib = _bytes_to_gib(total_bytes)
    target_utilization = _clamp(
        _env_float("CANOPYRS_GPU_TARGET_UTILIZATION", DEFAULT_GPU_TARGET_UTILIZATION),
        min_value=0.1,
        max_value=0.98,
    )
    reserve_gib = max(0.0, _env_float("CANOPYRS_GPU_MEMORY_RESERVE_GIB", DEFAULT_GPU_MEMORY_RESERVE_GIB))
    usable_gib = max(0.0, free_gib * target_utilization - reserve_gib)
    if usable_gib <= 0:
        return config

    updates = []
    for component_type, component_config in config.components_configs:
        if component_type == "detector":
            updates.extend(_scale_detector(component_config, usable_gib))
        elif component_type == "segmenter":
            updates.extend(_scale_segmenter(component_config, usable_gib))

    if updates:
        log(
            "GPU auto-tuning enabled "
            f"({free_gib:.1f}/{total_gib:.1f} GiB free, target usable {usable_gib:.1f} GiB): "
            + ", ".join(updates)
        )

    return config


def _scale_detector(component_config, usable_gib: float) -> list[str]:
    current_batch_size = getattr(component_config, "batch_size", None)
    if not current_batch_size:
        return []

    explicit_batch_size = _env_int("CANOPYRS_DETECTOR_BATCH_SIZE")
    if explicit_batch_size is not None:
        new_batch_size = explicit_batch_size
    else:
        max_batch_size = _env_int(
            "CANOPYRS_MAX_DETECTOR_BATCH_SIZE",
            DEFAULT_MAX_DETECTOR_BATCH_SIZE,
        )
        new_batch_size = _scaled_value(
            current_value=current_batch_size,
            usable_gib=usable_gib,
            reference_gib=_detector_reference_vram_gib(component_config),
            max_value=max_batch_size,
        )

    if new_batch_size == current_batch_size:
        return []

    component_config.batch_size = new_batch_size
    return [f"detector batch_size {current_batch_size}->{new_batch_size}"]


def _scale_segmenter(component_config, usable_gib: float) -> list[str]:
    updates = []

    current_image_batch_size = getattr(component_config, "image_batch_size", None)
    if current_image_batch_size:
        explicit_image_batch_size = _env_int("CANOPYRS_SEGMENTER_IMAGE_BATCH_SIZE")
        if explicit_image_batch_size is not None:
            new_image_batch_size = explicit_image_batch_size
        else:
            max_image_batch_size = _env_int(
                "CANOPYRS_MAX_SEGMENTER_IMAGE_BATCH_SIZE",
                DEFAULT_MAX_SEGMENTER_IMAGE_BATCH_SIZE,
            )
            new_image_batch_size = _scaled_value(
                current_value=current_image_batch_size,
                usable_gib=usable_gib,
                reference_gib=_segmenter_image_reference_vram_gib(component_config),
                max_value=max_image_batch_size,
            )

        if new_image_batch_size != current_image_batch_size:
            component_config.image_batch_size = new_image_batch_size
            updates.append(
                f"segmenter image_batch_size {current_image_batch_size}->{new_image_batch_size}"
            )

    current_box_batch_size = getattr(component_config, "box_batch_size", None)
    if current_box_batch_size:
        explicit_box_batch_size = _env_int("CANOPYRS_SEGMENTER_BOX_BATCH_SIZE")
        if explicit_box_batch_size is not None:
            new_box_batch_size = explicit_box_batch_size
        else:
            max_box_batch_size = _env_int(
                "CANOPYRS_MAX_SEGMENTER_BOX_BATCH_SIZE",
                DEFAULT_MAX_SEGMENTER_BOX_BATCH_SIZE,
            )
            new_box_batch_size = _scaled_value(
                current_value=current_box_batch_size,
                usable_gib=usable_gib,
                reference_gib=_segmenter_reference_vram_gib(component_config),
                max_value=max_box_batch_size,
            )

        if new_box_batch_size != current_box_batch_size:
            component_config.box_batch_size = new_box_batch_size
            updates.append(
                f"segmenter box_batch_size {current_box_batch_size}->{new_box_batch_size}"
            )

    return updates


def _scaled_value(current_value: int, usable_gib: float, reference_gib: float, max_value: int) -> int:
    if current_value < 1:
        return current_value
    if reference_gib <= 0:
        return current_value

    multiplier = max(1, int((usable_gib / reference_gib) + 0.5))
    scaled_value = current_value * multiplier
    max_value = max(1, max_value)

    return min(max_value, max(current_value, scaled_value))


def _detector_reference_vram_gib(component_config) -> float:
    model = str(getattr(component_config, "model", "")).lower()
    architecture = str(getattr(component_config, "architecture", "")).lower()

    if model == "dino_detrex" and "swin" in architecture:
        return 4.0
    if model == "dino_detrex":
        return 3.5
    if model in {"detectree2", "faster_rcnn_detectron2", "retinanet_detectron2"}:
        return 4.0
    if model == "deepforest":
        return 3.0
    return 6.0


def _segmenter_reference_vram_gib(component_config) -> float:
    model = str(getattr(component_config, "model", "")).lower()
    architecture = str(getattr(component_config, "architecture", "")).lower()

    if model in {"sam2", "sam3"} and architecture == "l":
        return 10.0
    if model in {"sam2", "sam3"}:
        return 8.0
    return 6.0


def _segmenter_image_reference_vram_gib(component_config) -> float:
    model = str(getattr(component_config, "model", "")).lower()
    architecture = str(getattr(component_config, "architecture", "")).lower()

    if model == "sam2" and architecture == "l":
        return 10.0
    if model == "sam2":
        return 8.0
    if model == "sam3" and architecture == "l":
        return 12.0
    if model == "sam3":
        return 10.0
    return _segmenter_reference_vram_gib(component_config)


def _env_flag(name: str) -> bool:
    value = os.getenv(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default=None):
    value = os.getenv(name)
    if value is None:
        return default

    try:
        parsed = int(value)
    except ValueError:
        return default

    return max(1, parsed)


def _env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    if value is None:
        return default

    try:
        return float(value)
    except ValueError:
        return default


def _clamp(value: float, min_value: float, max_value: float) -> float:
    return max(min_value, min(max_value, value))


def _bytes_to_gib(value: int) -> float:
    return value / (1024 ** 3)