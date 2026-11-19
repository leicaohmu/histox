"""Native PLIP image feature extraction for HistoX."""

from typing import Any, Optional, Union

import torch
from torchvision import transforms

from histox.model import torch_utils

from ._factory_torch import TorchFeatureExtractor


class _HuggingFacePLIPVisionModel(torch.nn.Module):
    """Return the projected PLIP image embedding for every tile."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values)
        image_embeds = getattr(output, "image_embeds", None)
        if image_embeds is None:
            raise RuntimeError("PLIP did not return image_embeds.")
        return image_embeds


class PLIPFeatures(TorchFeatureExtractor):
    """PLIP tile feature extractor backed by Hugging Face Transformers.

    The extractor returns the 512-dimensional projected image embedding used
    by PS3. Model weights are loaded at runtime and are never bundled with
    HistoX.

    Parameters
    ----------
    model_id : str
        Hugging Face model ID or local snapshot path. Defaults to
        ``"vinid/plip"``.
    revision : str, optional
        Fixed Hugging Face revision for reproducible loading.
    local_files_only : bool
        If ``True``, load only files already present in the local cache.
    device : str or torch.device, optional
        Device used for inference.
    """

    tag = "plip"
    license = (
        "Weights are not distributed by HistoX. Review the current upstream "
        "model card and terms at https://huggingface.co/vinid/plip."
    )
    citation = """
@article{huang2023visual,
  title={A visual-language foundation model for pathology image analysis using medical Twitter},
  author={Huang, Zhi and Bianchi, Federico and Yuksekgonul, Mert and Montine, Thomas and Zou, James},
  journal={Nature Medicine},
  year={2023}
}
"""

    def __init__(
        self,
        model_id: str = "vinid/plip",
        revision: Optional[str] = None,
        local_files_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        try:
            from transformers import CLIPVisionModelWithProjection
        except ImportError as exc:
            raise ImportError(
                "PLIP requires Transformers. Install it with "
                "`pip install 'histox[huggingface]'`."
            ) from exc

        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self.device = torch_utils.get_device(device)
        try:
            backbone = CLIPVisionModelWithProjection.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
        except (OSError, RuntimeError, ValueError) as exc:
            raise RuntimeError(
                "Unable to load PLIP vision weights from {!r}. The current "
                "upstream repository may provide PyTorch .bin weights, which "
                "recent Transformers releases refuse to load with "
                "torch<2.6. Upgrade PyTorch to 2.6 or newer, or pass a "
                "trusted local safetensors snapshot with model_id.".format(
                    model_id
                )
            ) from exc
        self.num_features = int(backbone.config.projection_dim)
        self.model = _HuggingFacePLIPVisionModel(backbone)
        self.model.to(self.device)
        self.model.eval()
        self.transform = self._build_transform()
        self.preprocess_kwargs = dict(standardize=False)

    def _build_transform(self) -> transforms.Compose:
        options = {
            "resize": 224,
            "center_crop": 224,
            "interpolation": "bicubic",
            "antialias": True,
            "norm_mean": (0.48145466, 0.4578275, 0.40821073),
            "norm_std": (0.26862954, 0.26130258, 0.27577711),
        }
        options.update(self.transform_kwargs)
        pipeline = []
        resize = options["resize"]
        if resize:
            pipeline.append(transforms.Resize(
                224 if resize is True else resize,
                interpolation=self._get_interpolation(options["interpolation"]),
                antialias=options["antialias"],
            ))
        center_crop = options["center_crop"]
        if center_crop:
            pipeline.append(transforms.CenterCrop(
                224 if center_crop is True else center_crop
            ))
        pipeline.extend([
            transforms.Lambda(lambda x: x / 255.0),
            transforms.Normalize(
                mean=options["norm_mean"],
                std=options["norm_std"],
            ),
        ])
        return transforms.Compose(pipeline)

    def dump_config(self) -> dict:
        """Return configuration needed to reconstruct this extractor."""
        return self._dump_config(
            class_name="histox.model.extractors.plip.PLIPFeatures",
            model_id=self.model_id,
            revision=self.revision,
            local_files_only=self.local_files_only,
        )
