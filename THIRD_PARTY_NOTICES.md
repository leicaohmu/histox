# Third-Party Notices and Review Inventory

This file records known third-party origins in HistoX. It is an engineering
inventory, not legal advice and not a substitute for the applicable license
texts. Exact upstream revisions still need to be pinned before a public release.

## Source included or adapted in this repository

| HistoX path | Upstream project | Recorded license | Use in HistoX |
| --- | --- | --- | --- |
| Broad portions of `histox/` | [Slideflow](https://github.com/slideflow/slideflow) | Apache-2.0 | Forked, renamed and modified |
| `histox/_version.py` | [Versioneer](https://github.com/python-versioneer/python-versioneer) | Unlicense | Generated/version-management source |
| `histox/io/gaussian.py` | [TensorFlow Addons](https://github.com/tensorflow/addons) | Apache-2.0 | Adapted Gaussian-filter implementation |
| `histox/mil/models/att_mil.py` | [marugoto](https://github.com/KatherLab/marugoto) | MIT | Adapted attention-MIL implementation |
| `histox/mil/models/bistro/` | [HistoBistro](https://github.com/peng-lab/HistoBistro) | MIT | Adapted MIL implementation |
| `histox/model/extractors/vit.py` | [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | Adapted vision-transformer implementation |
| `histox/model/tensorflow_utils.py` | [pycox](https://github.com/havakv/pycox) | BSD-2-Clause | Adapted negative-log-likelihood function |
| Parts of `histox/norm/` | [StainTools](https://github.com/Peter554/StainTools) | MIT | Adapted stain-normalization utilities |
| `histox/norm/tensorflow/color.py` | [TensorFlow I/O](https://github.com/tensorflow/io) | Apache-2.0 | Adapted color-conversion implementation |
| `histox/norm/torch/color.py` | [colorization-pytorch](https://github.com/richzhang/colorization-pytorch) | MIT | Adapted color-conversion implementation |
| `histox/norm/torch/cyclegan.py` | [stain-transfer](https://github.com/Boehringer-Ingelheim/stain-transfer) | BSD-2-Clause | Adapted model implementation; notice retained in source |
| `histox/segment/_cp_utils.py` | [Cellpose](https://github.com/MouseLand/cellpose) | BSD-3-Clause | Adapted utility code; notice retained in source |
| `histox/stats/concordance.py` | [lifelines](https://github.com/CamDavidsonPilon/lifelines) | MIT | Adapted concordance code; notice retained in source |
| `histox/stats/delong.py` | [VMAF](https://github.com/Netflix/vmaf) | BSD-2-Clause-Patent | Adapted DeLong implementation |
| `histox/tfrecord/` | [tfrecord](https://github.com/vahidk/tfrecord) | MIT | Adapted TFRecord implementation |

The table reports licenses visible in the named upstream repositories during the
inventory. It does not yet prove which upstream revision was imported into every
HistoX file.

## External artifacts and optional integrations

Optional dependencies, plugins, model weights and datasets are distributed under
their own terms. Their presence in documentation or dependency metadata does not
make them Apache-2.0. This includes optional Slideflow extensions, pretrained
models and public pathology datasets downloaded by users.

## Items that must be resolved before a new release

1. **Multiprocessing logging handler:** `histox/util/log_utils.py` attributes its
   handler to
   [multiprocessing-logging](https://github.com/jruere/multiprocessing-logging).
   Confirm the exact imported revision, license and required redistribution
   notice, then either comply, replace or rewrite the copied implementation.
2. **Stain-normalization fork:** `histox/norm/utils.py` refers to
   [wanghao14/Stain_Normalization](https://github.com/wanghao14/Stain_Normalization),
   whose current repository view does not expose a license. Trace the relevant
   lines to a licensed source or replace them.
3. **Attention-flow reference:**
   `histox/mil/models/bistro/transformer.py` refers to
   [attention_flow](https://github.com/samiraabnar/attention_flow), whose current
   repository view does not expose a license. Determine whether any source was
   copied and replace it if redistribution cannot be established.
4. **Bundled assets:** create a file-level origin and license inventory for
   tracked fonts, logos, icons and reference images, including `*.ttf`, `*.png`
   and `*.jpg` files.
5. **Package license conflict:** `histox/__init__.py` declares GPL-3.0 while the
   top-level `LICENSE` declares Apache-2.0. Resolve this only after the preceding
   component review is complete.

Until these items are closed, the repository should not publish a new package
release that presents all bundled content as uniformly Apache-2.0.
