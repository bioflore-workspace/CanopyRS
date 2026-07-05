import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np


geodataset_module = types.ModuleType("geodataset")
geodataset_dataset_module = types.ModuleType("geodataset.dataset")
geodataset_utils_module = types.ModuleType("geodataset.utils")


class _BaseDataset:
    pass


class _DetectionLabeledRasterCocoDataset(_BaseDataset):
    pass


class _UnlabeledRasterDataset(_BaseDataset):
    pass


geodataset_dataset_module.BaseDataset = _BaseDataset
geodataset_dataset_module.DetectionLabeledRasterCocoDataset = _DetectionLabeledRasterCocoDataset
geodataset_dataset_module.UnlabeledRasterDataset = _UnlabeledRasterDataset
geodataset_utils_module.mask_to_polygon = lambda *args, **kwargs: None

sys.modules.setdefault("geodataset", geodataset_module)
sys.modules["geodataset.dataset"] = geodataset_dataset_module
sys.modules["geodataset.utils"] = geodataset_utils_module

from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase


class _DummyQueue:
    def put(self, item):
        return None

    def join(self):
        return None

    def close(self):
        return None


class _DummyManager:
    def dict(self):
        return {}


class _DummyValue:
    def __init__(self, *args, **kwargs):
        self.value = 0


class _DummyProcess:
    def __init__(self, *args, **kwargs):
        return None

    def start(self):
        return None

    def join(self):
        return None


class _FakeDetectionDataset(_DetectionLabeledRasterCocoDataset):
    def __init__(self, n_tiles):
        self.tiles = {tile_idx: {"path": f"/tmp/tile_{tile_idx}.tif"} for tile_idx in range(n_tiles)}


class _RecordingSegmenter(SegmenterWrapperBase):
    REQUIRES_BOX_PROMPT = True

    def __init__(self, config):
        self.forward_calls = []
        super().__init__(config)

    def forward(self, images, boxes, boxes_object_ids, tiles_idx, queue):
        self.forward_calls.append(
            {
                "tiles_idx": tiles_idx,
                "images_len": len(images),
                "boxes_len": len(boxes),
            }
        )

    def infer_on_dataset(self, dataset):
        return self._infer_on_dataset(dataset, collate_fn=None)


def _build_detection_batches(n_tiles, image_batch_size):
    batches = []
    for batch_start in range(0, n_tiles, image_batch_size):
        batch_end = min(batch_start + image_batch_size, n_tiles)
        batch_len = batch_end - batch_start
        batches.append(
            (
                [np.zeros((4, 4, 3), dtype=np.uint8) for _ in range(batch_len)],
                [np.zeros((1, 4), dtype=np.float32) for _ in range(batch_len)],
                [[tile_idx] for tile_idx in range(batch_start, batch_end)],
            )
        )
    return batches


def test_detection_batches_use_actual_last_batch_size_for_tile_indexes():
    n_tiles = 85
    image_batch_size = 4
    dataset = _FakeDetectionDataset(n_tiles=n_tiles)
    segmenter = _RecordingSegmenter(
        SimpleNamespace(
            image_batch_size=image_batch_size,
            dataloader_num_workers=0,
            pp_n_workers=0,
            model="sam2",
            architecture="l",
        )
    )

    fake_batches = _build_detection_batches(n_tiles=n_tiles, image_batch_size=image_batch_size)

    with patch("canopyrs.engine.models.segmenter.segmenter_base.DataLoader", return_value=fake_batches), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.tqdm", side_effect=lambda iterable, **kwargs: iterable), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.multiprocessing.JoinableQueue", return_value=_DummyQueue()), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.multiprocessing.Manager", return_value=_DummyManager()), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.multiprocessing.Value", return_value=_DummyValue()), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.multiprocessing.Lock", return_value=object()), \
         patch("canopyrs.engine.models.segmenter.segmenter_base.multiprocessing.Process", _DummyProcess):
        tiles_paths, _, _, _ = segmenter.infer_on_dataset(dataset)

    assert len(tiles_paths) == n_tiles
    assert tiles_paths[-1] == "/tmp/tile_84.tif"
    assert segmenter.forward_calls[-1]["images_len"] == 1
    assert segmenter.forward_calls[-1]["tiles_idx"] == [84]