import torch

from canopyrs.engine.config_parsers.detector import DetectorConfig
from canopyrs.engine.config_parsers.segmenter import SegmenterConfig
from canopyrs.engine.inference_tuning import auto_scale_pipeline_inference


class DummyPipelineConfig:
    def __init__(self, components_configs):
        self.components_configs = components_configs


def test_auto_scale_pipeline_inference_uses_free_vram(monkeypatch):
    pipeline_config = DummyPipelineConfig(
        components_configs=[
            (
                'detector',
                DetectorConfig(
                    model='dino_detrex',
                    architecture='dino/configs/dino-swin/dino_swin_large_384_5scale_36ep.py',
                    batch_size=1,
                ),
            ),
            (
                'segmenter',
                SegmenterConfig(model='sam2', architecture='l', box_batch_size=50),
            ),
        ]
    )

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (22 * 1024 ** 3, 24 * 1024 ** 3))

    messages = []
    auto_scale_pipeline_inference(pipeline_config, log=messages.append)

    detector_config = pipeline_config.components_configs[0][1]
    segmenter_config = pipeline_config.components_configs[1][1]
    assert detector_config.batch_size == 2
    assert segmenter_config.box_batch_size == 100
    assert messages


def test_auto_scale_pipeline_inference_honors_env_overrides(monkeypatch):
    pipeline_config = DummyPipelineConfig(
        components_configs=[
            ('detector', DetectorConfig(model='dino_detrex', batch_size=1)),
            ('segmenter', SegmenterConfig(model='sam2', box_batch_size=50)),
        ]
    )

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'mem_get_info', lambda: (12 * 1024 ** 3, 24 * 1024 ** 3))
    monkeypatch.setenv('CANOPYRS_DETECTOR_BATCH_SIZE', '4')
    monkeypatch.setenv('CANOPYRS_SEGMENTER_BOX_BATCH_SIZE', '150')

    auto_scale_pipeline_inference(pipeline_config, log=lambda _: None)

    detector_config = pipeline_config.components_configs[0][1]
    segmenter_config = pipeline_config.components_configs[1][1]
    assert detector_config.batch_size == 4
    assert segmenter_config.box_batch_size == 120