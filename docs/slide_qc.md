# histox.slide.qc

This module contains functions for slide-level quality control, including Otsu's thresholding and Gaussian blur filtering. Quality control methods are used by passing a list of callables to the `qc` argument of `.extract_tiles()`. They can also be directly applied to a slide with `histox.WSI.qc()`.

```
import histox as hx
from histox.slide import qc

# Define custom QC options
qc = [
 qc.Otsu(),
 qc.Gaussian(sigma=2)
]

# Use this QC during tile extraction
P.extract_tiles(qc=qc)

# Alternatively, you can use the same QC directly on a WSI object
wsi = hx.WSI(...)
wsi.qc(qc).show()
```

*class*histox.slide.qc.Otsu(*slide_level: int | None = None*)[[source]](_modules/histox/slide/qc/otsu.html#Otsu)

*class*histox.slide.qc.Gaussian(*mpp: float | None = None*, *sigma: int = 3*, *threshold: float = 0.02*)[[source]](_modules/histox/slide/qc/gaussian.html#Gaussian)

*class*histox.slide.qc.Save(*dest: str | None = None*)[[source]](_modules/histox/slide/qc/saver.html#Save)

*class*histox.slide.qc.Load(*source: str | None = None*)[[source]](_modules/histox/slide/qc/saver.html#Load)

*class*histox.slide.qc.StridedDL(*model: Callable*, *pred_idx: int*, *tile_px: int*, *tile_um: str | int*, ***, *buffer: int = 8*, *verbose: bool = False*, *pred_threshold: float = 0.5*, ***wsi_kwargs*)[[source]](_modules/histox/slide/qc/strided_dl.html#StridedDL)