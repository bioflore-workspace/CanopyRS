import importlib
from .registry import (
    DETECTOR_REGISTRY,
    SEGMENTER_REGISTRY,
    CLASSIFIER_REGISTRY,
    EMBEDDER_REGISTRY,
)


MODEL_MODULES = {
    "detector": {
        "deepforest": "deepforest_infer",
        "dino_detrex": "detectron2_infer",
        "faster_rcnn_detectron2": "detectron2_infer",
        "retinanet_detectron2": "detectron2_infer",
        "detectree2": "detectron2_infer",
        "faster_rcnn": "pt_faster_rcnn",
        "retina_net": "pt_retina_net",
    },
    "segmenter": {
        "detectree2": "detectron2_infer",
        "mask_rcnn_detectron2": "detectron2_infer",
        "mask2former_detrex": "detectron2_infer",
        "sam": "sam",
        "sam2": "sam2",
        "sam3": "sam3",
    },
    "classifier": {
        "resnet": "resnet_classifier",
        "swin": "swin_classifier",
    },
    "embedder": {},
}


def _get_registry(model_type: str):
    return {
        "detector": DETECTOR_REGISTRY,
        "segmenter": SEGMENTER_REGISTRY,
        "classifier": CLASSIFIER_REGISTRY,
        "embedder": EMBEDDER_REGISTRY,
    }[model_type]


def ensure_model_registered(model_type: str, model_name: str) -> None:
    registry = _get_registry(model_type)
    if model_name in registry:
        return

    module_name = MODEL_MODULES.get(model_type, {}).get(model_name)
    if module_name is None:
        return

    importlib.import_module(f"canopyrs.engine.models.{model_type}.{module_name}")

# Make registries available at package level
__all__ = [
    "DETECTOR_REGISTRY",
    "SEGMENTER_REGISTRY",
    "CLASSIFIER_REGISTRY",
    "EMBEDDER_REGISTRY",
    "ensure_model_registered",
]
