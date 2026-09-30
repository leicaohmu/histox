from typing import Any, Optional, Union

import torch
from torchvision import transforms

from histox.model import torch_utils

from ._factory_torch import TorchFeatureExtractor

# -----------------------------------------------------------------------------


class _HuggingFaceDinoV2Model(torch.nn.Module):
    """Return one embedding per image from a Hugging Face DINOv2 model."""

    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        output = self.model(pixel_values=pixel_values)
        pooler_output = getattr(output, 'pooler_output', None)
        if pooler_output is not None:
            return pooler_output

        last_hidden_state = getattr(output, 'last_hidden_state', None)
        if last_hidden_state is None:
            raise RuntimeError(
                "DINOv2 did not return pooler_output or last_hidden_state."
            )
        return last_hidden_state[:, 0]


class DinoV2Features(TorchFeatureExtractor):
    """DinoV2 feature extractor.

    The extractor supports either a Hugging Face model ID or the original
    DINOv2 configuration-and-weights pair. The embedding dimension is read
    from the loaded model (for example, 384 for ``facebook/dinov2-small``).

    GitHub: https://github.com/facebookresearch/dinov2
    """

    tag = 'dinov2'
    license = "Apache-2.0"
    citation = """
@misc{oquab2023dinov2,
  title={DINOv2: Learning Robust Visual Features without Supervision},
  author={Oquab, Maxime and Darcet, Timothée and Moutakanni, Theo and Vo, Huy V. and Szafraniec, Marc and Khalidov, Vasil and Fernandez, Pierre and Haziza, Daniel and Massa, Francisco and El-Nouby, Alaaeldin and Howes, Russell and Huang, Po-Yao and Xu, Hu and Sharma, Vasu and Li, Shang-Wen and Galuba, Wojciech and Rabbat, Mike and Assran, Mido and Ballas, Nicolas and Synnaeve, Gabriel and Misra, Ishan and Jegou, Herve and Mairal, Julien and Labatut, Patrick and Joulin, Armand and Bojanowski, Piotr},
  journal={arXiv:2304.07193},
  year={2023}
}
"""

    def __init__(
        self,
        cfg: Optional[str] = None,
        weights: Optional[str] = None,
        model_id: Optional[str] = None,
        revision: Optional[str] = None,
        local_files_only: bool = False,
        device: Optional[Union[str, torch.device]] = None,
        **kwargs: Any
    ) -> None:
        """Create a DINOv2 feature extractor.

        Parameters
        ----------
        cfg : str, optional
            Path to an original DINOv2 YAML configuration. Must be supplied
            together with ``weights``.
        weights : str, optional
            Path to original DINOv2 weights. Must be supplied with ``cfg``.
        model_id : str, optional
            Hugging Face model ID or local snapshot path, such as
            ``"facebook/dinov2-small"``.
        revision : str, optional
            Hugging Face revision used to make model loading reproducible.
        local_files_only : bool, optional
            If ``True``, never contact Hugging Face and load only local files.
        device : str or torch.device, optional
            Device used for inference.

        Notes
        -----
        Set the standard ``HF_ENDPOINT`` environment variable before model
        construction to use a Hugging Face mirror.
        """
        super().__init__(**kwargs)

        has_custom_cfg = cfg is not None or weights is not None
        if has_custom_cfg and (cfg is None or weights is None):
            raise ValueError("cfg and weights must be provided together.")
        if has_custom_cfg and model_id is not None:
            raise ValueError(
                "Choose either model_id or cfg and weights, not both."
            )
        if has_custom_cfg and (revision is not None or local_files_only):
            raise ValueError(
                "revision and local_files_only are only valid with model_id."
            )
        if not has_custom_cfg and model_id is None:
            raise ValueError(
                "Provide model_id, or provide both cfg and weights."
            )

        self.cfg = cfg
        self.weights = weights
        self.model_id = model_id
        self.revision = revision
        self.local_files_only = local_files_only
        self.device = torch_utils.get_device(device)

        if model_id is not None:
            try:
                from transformers import AutoModel
            except ImportError as exc:
                raise ImportError(
                    "Loading DINOv2 from Hugging Face requires transformers. "
                    "Install it with `pip install 'histox[huggingface]'`."
                ) from exc
            backbone = AutoModel.from_pretrained(
                model_id,
                revision=revision,
                local_files_only=local_files_only,
            )
            self.num_features = int(backbone.config.hidden_size)
            self.model = _HuggingFaceDinoV2Model(backbone)
            self.transform = self._build_huggingface_transform()
        else:
            try:
                from omegaconf import OmegaConf
                from dinov2.eval.setup import build_model_for_eval
            except ImportError as exc:
                raise ImportError(
                    "Loading original DINOv2 checkpoints requires the "
                    "facebookresearch/dinov2 source package and OmegaConf."
                ) from exc
            self.model = build_model_for_eval(OmegaConf.load(cfg), weights)
            self.num_features = int(getattr(self.model, 'embed_dim', 1024))
            self.transform = self.build_transform(img_size=224)

        self.model.to(self.device)
        self.model.eval()

        self.preprocess_kwargs = dict(standardize=False)

    def _build_huggingface_transform(self) -> transforms.Compose:
        """Build transforms from the official DINOv2 processor parameters."""
        options = {
            'resize': 256,
            'center_crop': 224,
            'interpolation': 'bicubic',
            'antialias': True,
            'norm_mean': (0.485, 0.456, 0.406),
            'norm_std': (0.229, 0.224, 0.225),
        }
        options.update(self.transform_kwargs)

        pipeline = []
        resize = options['resize']
        if resize:
            pipeline.append(transforms.Resize(
                224 if resize is True else resize,
                interpolation=self._get_interpolation(options['interpolation']),
                antialias=options['antialias'],
            ))
        center_crop = options['center_crop']
        if center_crop:
            pipeline.append(transforms.CenterCrop(
                224 if center_crop is True else center_crop
            ))
        pipeline.extend([
            transforms.Lambda(lambda x: x / 255.),
            transforms.Normalize(
                mean=options['norm_mean'],
                std=options['norm_std'],
            ),
        ])
        return transforms.Compose(pipeline)

    def dump_config(self) -> dict:
        """Return a dictionary of configuration parameters.

        These configuration parameters can be used to reconstruct the
        feature extractor, using ``histox.build_feature_extractor()``.

        """
        kwargs = {}
        if self.model_id is not None:
            kwargs.update(
                model_id=self.model_id,
                revision=self.revision,
                local_files_only=self.local_files_only,
            )
        else:
            kwargs.update(cfg=self.cfg, weights=self.weights)
        return self._dump_config(
            class_name=f'histox.model.extractors.dinov2.{self.__class__.__name__}',
            **kwargs
        )
