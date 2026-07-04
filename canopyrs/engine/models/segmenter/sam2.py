from typing import List

import numpy as np
import torch
from geodataset.dataset import DetectionLabeledRasterCocoDataset
import multiprocessing

from canopyrs.engine.config_parsers import SegmenterConfig
from canopyrs.engine.models.segmenter.segmenter_base import SegmenterWrapperBase
from canopyrs.engine.models.registry import SEGMENTER_REGISTRY
from canopyrs.engine.models.utils import collate_fn_infer_image_box
from pathlib import Path

@SEGMENTER_REGISTRY.register('sam2')
class Sam2PredictorWrapper(SegmenterWrapperBase):
    MODEL_MAPPING = {
        't': "facebook/sam2-hiera-tiny",
        's': "facebook/sam2-hiera-small",
        'b': "facebook/sam2-hiera-base-plus",
        'l': "facebook/sam2-hiera-large",
    }

    REQUIRES_BOX_PROMPT = True

    def __init__(self, config: SegmenterConfig):
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        super().__init__(config)

        self.model_name = self.MODEL_MAPPING[self.config.architecture]

        # Load SAM model and processor
        print(f"Loading model {self.model_name}")
        self.predictor = SAM2ImagePredictor.from_pretrained(self.model_name)
        self.supports_image_batching = all(
            hasattr(self.predictor, attr) for attr in ["set_image_batch", "_prep_prompts", "_predict"]
        )
        print(f"Model {self.model_name} loaded")

        checkpoint_path = getattr(self.config, 'checkpoint_path', None)
        if checkpoint_path:
            checkpoint_path = Path(checkpoint_path)
            if checkpoint_path.exists():
                print(f"Loading fine-tuned checkpoint:")
                print(f"  Path: {checkpoint_path}")
                
                # Load state dict
                state_dict = torch.load(checkpoint_path, map_location='cpu')
                
                # Handle different checkpoint formats
                if 'model_state_dict' in state_dict:
                    # Full checkpoint with optimizer, etc.
                    model_state_dict = state_dict['model_state_dict']
                    print(f"  Checkpoint type: Full training checkpoint")
                else:
                    # Just model weights
                    model_state_dict = state_dict
                    print(f"  Checkpoint type: Model weights only")
                
                # Load weights into predictor model
                self.predictor.model.load_state_dict(model_state_dict)
                print(f"Fine-tuned weights loaded successfully!")
            else:
                print(f"\nWARNING: Checkpoint not found: {checkpoint_path}")
                print(f"Using base pretrained model instead.\n")
                
    def forward(self,
                images: List[np.array],
                boxes: List[np.array],
                boxes_object_ids: List[int],
                tiles_idx: List[int],
                queue: multiprocessing.JoinableQueue):
        prepared_images = [self._prepare_image(image) for image in images]
        use_batched_embeddings = self.supports_image_batching and len(prepared_images) > 1

        if use_batched_embeddings:
            self._forward_with_batched_embeddings(
                prepared_images=prepared_images,
                boxes=boxes,
                boxes_object_ids=boxes_object_ids,
                tiles_idx=tiles_idx,
                queue=queue,
            )
            return

        self._forward_sequential(
            prepared_images=prepared_images,
            boxes=boxes,
            boxes_object_ids=boxes_object_ids,
            tiles_idx=tiles_idx,
            queue=queue,
        )

    @staticmethod
    def _prepare_image(image: np.ndarray) -> np.ndarray:
        image = image[:3, :, :]
        image = image.transpose((1, 2, 0))
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        return image

    def _forward_sequential(self,
                            prepared_images: List[np.ndarray],
                            boxes: List[np.ndarray],
                            boxes_object_ids: List[List[int]],
                            tiles_idx: List[int],
                            queue: multiprocessing.JoinableQueue):
        autocast_enabled = self.device.type == "cuda"

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            for image, image_boxes, image_boxes_object_ids, tile_idx in zip(
                prepared_images, boxes, boxes_object_ids, tiles_idx
            ):
                if len(image_boxes) == 0:
                    continue

                self.predictor.set_image(image)
                n_masks_processed = 0
                image_size = (image.shape[0], image.shape[1])
                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = np.asarray(image_boxes[i:i + self.config.box_batch_size], dtype=np.float32)
                    if len(box_batch) == 0:
                        continue

                    box_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]
                    masks, scores, _ = self.predictor.predict(
                        box=box_batch,
                        multimask_output=False,
                        normalize_coords=True,
                    )
                    n_masks_processed = self._queue_prediction_batch(
                        tile_idx=tile_idx,
                        image_size=image_size,
                        box_object_ids_batch=box_object_ids_batch,
                        masks=masks,
                        scores=scores,
                        n_masks_processed=n_masks_processed,
                        queue=queue,
                    )

    def _forward_with_batched_embeddings(self,
                                         prepared_images: List[np.ndarray],
                                         boxes: List[np.ndarray],
                                         boxes_object_ids: List[List[int]],
                                         tiles_idx: List[int],
                                         queue: multiprocessing.JoinableQueue):
        autocast_enabled = self.device.type == "cuda"

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            self.predictor.set_image_batch(prepared_images)

            for img_idx, (image, image_boxes, image_boxes_object_ids, tile_idx) in enumerate(
                zip(prepared_images, boxes, boxes_object_ids, tiles_idx)
            ):
                if len(image_boxes) == 0:
                    continue

                n_masks_processed = 0
                image_size = (image.shape[0], image.shape[1])

                for i in range(0, len(image_boxes), self.config.box_batch_size):
                    box_batch = np.asarray(image_boxes[i:i + self.config.box_batch_size], dtype=np.float32)
                    if len(box_batch) == 0:
                        continue

                    box_object_ids_batch = image_boxes_object_ids[i:i + self.config.box_batch_size]
                    mask_input, unnorm_coords, labels, unnorm_box = self.predictor._prep_prompts(
                        point_coords=None,
                        point_labels=None,
                        box=box_batch,
                        mask_logits=None,
                        normalize_coords=True,
                        img_idx=img_idx,
                    )
                    masks, scores, _ = self.predictor._predict(
                        point_coords=unnorm_coords,
                        point_labels=labels,
                        boxes=unnorm_box,
                        mask_input=mask_input,
                        multimask_output=False,
                        return_logits=False,
                        img_idx=img_idx,
                    )
                    n_masks_processed = self._queue_prediction_batch(
                        tile_idx=tile_idx,
                        image_size=image_size,
                        box_object_ids_batch=box_object_ids_batch,
                        masks=masks,
                        scores=scores,
                        n_masks_processed=n_masks_processed,
                        queue=queue,
                    )

    def _queue_prediction_batch(self,
                                tile_idx: int,
                                image_size: tuple[int, int],
                                box_object_ids_batch: List[int],
                                masks,
                                scores,
                                n_masks_processed: int,
                                queue: multiprocessing.JoinableQueue) -> int:
        if isinstance(masks, torch.Tensor):
            masks = masks.float().detach().cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.float().detach().cpu().numpy()

        masks = np.asarray(masks)
        scores = np.asarray(scores).reshape(-1)

        if masks.ndim == 4:
            masks = masks[:, 0, :, :]
        elif masks.ndim == 2:
            masks = masks[None, :, :]

        return self.queue_masks(
            box_object_ids_batch,
            masks,
            image_size,
            scores,
            tile_idx,
            n_masks_processed,
            queue,
        )

    def infer_on_dataset(self, dataset: DetectionLabeledRasterCocoDataset):
        return self._infer_on_dataset(dataset, collate_fn_infer_image_box)
