# Third-Party Notices and Review Inventory

This file records known third-party origins in HistoX. It is an engineering
inventory, not legal advice and not a substitute for the applicable license
texts. Pinned revisions below identify the source used for this audit.

## Source included or adapted in this repository

| HistoX path | Upstream project | Recorded license | Use in HistoX |
| --- | --- | --- | --- |
| Broad portions of `histox/` | [Slideflow](https://github.com/slideflow/slideflow) | Apache-2.0 | Forked, renamed and modified |
| `histox/_version.py` | [Versioneer](https://github.com/python-versioneer/python-versioneer) | Unlicense | Generated/version-management source |
| `histox/io/gaussian.py` | [TensorFlow Addons](https://github.com/tensorflow/addons) | Apache-2.0 | Adapted Gaussian-filter implementation |
| `histox/mil/models/att_mil.py` | [marugoto](https://github.com/KatherLab/marugoto) | MIT | Adapted attention-MIL implementation |
| `histox/mil/models/bistro/` | [HistoBistro `969d516`](https://github.com/peng-lab/HistoBistro/commit/969d516d23fb12046153f7a39ebb4dacb411d395) | MIT | Adapted MIL implementation; license text in `licenses/MIT-HistoBistro.txt` |
| `histox/mil/models/bistro/transformer.py::_compute_rollout` | [Transformer-Explainability `38071a7`](https://github.com/hila-chefer/Transformer-Explainability/commit/38071a7a0ae7836d0f2e6abc9ed47bf96ac149cd) | MIT | Adapted PyTorch attention-rollout implementation; license text in `licenses/MIT-Transformer-Explainability.txt` |
| `histox/model/extractors/vit.py` | [timm](https://github.com/huggingface/pytorch-image-models) | Apache-2.0 | Adapted vision-transformer implementation |
| `histox/model/tensorflow_utils.py` | [pycox](https://github.com/havakv/pycox) | BSD-2-Clause | Adapted negative-log-likelihood function |
| `histox/norm/utils.py` stain utility block | [StainTools `54e97bbd`](https://github.com/Peter554/StainTools/commit/54e97bbdce9d3a25289c96e5b732294d8ba49b6d) via [wanghao14/Stain_Normalization `1a40d41`](https://github.com/wanghao14/Stain_Normalization/commit/1a40d41992ba4e1c67f9f0ac8ad144be6b1f015b) | MIT | Adapted stain-normalization utilities; license text in `licenses/MIT-StainTools.txt` |
| `histox/norm/tensorflow/color.py` | [TensorFlow I/O](https://github.com/tensorflow/io) | Apache-2.0 | Adapted color-conversion implementation |
| `histox/norm/torch/color.py` | [colorization-pytorch](https://github.com/richzhang/colorization-pytorch) | MIT | Adapted color-conversion implementation |
| `histox/norm/torch/cyclegan.py` | [stain-transfer](https://github.com/Boehringer-Ingelheim/stain-transfer) | BSD-2-Clause | Adapted model implementation; notice retained in source |
| `histox/segment/_cp_utils.py` | [Cellpose](https://github.com/MouseLand/cellpose) | BSD-3-Clause | Adapted utility code; notice retained in source |
| `histox/stats/concordance.py` | [lifelines](https://github.com/CamDavidsonPilon/lifelines) | MIT | Adapted concordance code; notice retained in source |
| `histox/stats/delong.py` | [VMAF](https://github.com/Netflix/vmaf) | BSD-2-Clause-Patent | Adapted DeLong implementation |
| `histox/tfrecord/` | [tfrecord](https://github.com/vahidk/tfrecord) | MIT | Adapted TFRecord implementation |

The table reports licenses verified in the named upstream repositories. Rows
without pinned revisions remain inventory items for later file-level review.

## Replaced implementation

Earlier revisions of `histox/util/log_utils.py` contained a modified copy of
`MultiProcessingHandler` from
[multiprocessing-logging `3449c4e`](https://github.com/jruere/multiprocessing-logging/commit/3449c4e785bb61c99111c05e69c2dba0c2ab6cd1),
licensed LGPL-3.0. That copied implementation has been removed. The current
handler is composed from Python standard-library logging primitives.

## Bundled fonts

The following files declare Apache-2.0 in their embedded font metadata:

| File | Copyright | SHA-256 |
| --- | --- | --- |
| `histox/studio/gui/fonts/DroidSans.ttf` | Digitized data copyright 2007 Google Corporation | `f51b88945f4c1b236f44b8d55a2d304316869127e95248c435c23f1e4142a7db` |
| `histox/studio/gui/fonts/DroidSans-Bold.ttf` | Digitized data copyright 2007 Google Corporation | `2f529a3e60c007979d95d29794c3660694217fb882429fb33919d2245fe969e9` |
| `histox/studio/gui/fonts/open_sans.ttf` | Digitized data copyright 2010-2011 Google Corporation | `037236ed4bf58a85f67074c165d308260fd6be01c86d7df4e79ea16eb273f8c5` |

The Droid font files also carry the applicable upstream NOTICE attribution:
Copyright (c) 2005-2008, The Android Open Source Project. The source NOTICE is
[`platform/frameworks/base@c049f9a/data/fonts/NOTICE`](https://android.googlesource.com/platform/frameworks/base/+/c049f9a/data/fonts/NOTICE).
The Apache-2.0 text is the repository's top-level `LICENSE`.

## External artifacts and optional integrations

Optional dependencies, plugins, model weights and datasets are distributed under
their own terms. Their presence in documentation or dependency metadata does not
make them Apache-2.0. This includes optional Slideflow extensions, pretrained
models and public pathology datasets downloaded by users.

## Items that must still be resolved before a new release

1. **Unknown-origin pathology image:** `histox/norm/norm_tile.jpg` has no
   recoverable creator, dataset or reuse license. Replace the three tests that
   load it, then remove the image from package data.
2. **Legacy branding:** replace the inherited Slideflow logo and Studio
   logo/splash files with HistoX-owned artwork.
3. **Studio icon glyphs:** document the underlying source and license for each
   retained glyph, or replace the set with a named permissive icon source.
4. **Documentation images:** audit source examples separately from generated
   documentation output during the documentation rebuild.

Until these items are closed and wheel/sdist contents are verified, the
repository should not publish a new package release that presents all bundled
content as uniformly Apache-2.0.
