# Tutorial 6: Custom slide filtering

**Status** Compatibility tutorial**Runtime verification** Pending[View source](https://github.com/leicaohmu/histox/blob/develop/docs/source/tutorial6.rst)

In this brief tutorial, we'll take a look at how you can implement and preview bespoke slide-level filtering methods.

The slide-level filtering (QC) methods HistoX currently supports include Otsu's thresholding and Gaussian blur filtering, which can be applied to a [`WSI`](slide.html#histox.slide.WSI) object with [`WSI.qc()`](slide.html#WSI.qc). If you have a custom filtering algorithm you would like to apply to a slide, you can now use [`WSI.apply_qc_mask()`](slide.html#WSI.apply_qc_mask) to apply a boolean mask to filter a slide.

For the purposes of this tutorial, we will generate a boolean mask using the already-available Otsu's thresholding algorithm, but you can replace this with whatever masking algorithm you like.

First, we'll load a slide:

```
import numpy as np
import histox as hx

wsi = hx.WSI('slide.svs', tile_px=299, tile_um=302)
```

Next, we'll apply Otsu's thresholding to get the boolean mask we'll use in subsequent steps, then remove the QC once we have the mask:

```
wsi.qc('otsu')
qc_mask = np.copy(wsi.qc_mask)
wsi.remove_qc()
```

Our mask should have two dimensions (y, x) and have a dtype of bool:

```
>>> qc_mask.shape
(1010, 2847)
>>> qc_mask.dtype
dtype('bool')
```

Our [`WSI`](slide.html#histox.slide.WSI) object now has no QC applied. We can manually apply this boolean mask with [`WSI.apply_qc_mask()`](slide.html#WSI.apply_qc_mask):

```
wsi.apply_qc_mask(qc_mask)
```

And that's it! We can preview how our mask affects tile filtering by using [`WSI.preview()`](slide.html#WSI.preview):

```
wsi.preview().show()
```