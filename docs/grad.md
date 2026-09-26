# histox.grad

This submodule contains tools for calculating and display pixel attribution, or
saliency, maps. See [Saliency Maps](saliency.html#saliency) for more information.

*class*histox.grad.SaliencyMap(*model: Callable*, *class_idx: int*)[[source]](_modules/histox/grad.html#SaliencyMap)

all(*img: ndarray*) → Dict[[source]](_modules/histox/grad.html#SaliencyMap.all)

Calculate all saliency map methods.

Parameters:

**img** (*np.ndarray*) - Pre-processed input image in W, H, C format.

Returns:

Dictionary mapping name of saliency method to saliency map.

Return type:

Dict

blur_integrated_gradients(*img: ndarray*, *batch_size: int = 20*, *smooth: bool = False*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.blur_integrated_gradients)

Calculate saliency map using blur integrated gradients.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **batch_size** (*int**,**optional*) - Batch size. Defaults to 20.
- **smooth** (*bool**,**optional*) - Smooth gradients. Defaults to False.

Returns:

Saliency map.

Return type:

np.ndarray

gradcam(*img: ndarray*, *layer: str*, *smooth: bool = False*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.gradcam)

Calculate gradient-based saliency map.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **smooth** (*bool**,**optional*) - Smooth gradients. Defaults to False.

Returns:

Saliency map.

Return type:

np.ndarray

guided_integrated_gradients(*img: ndarray*, *x_steps: int = 25*, *max_dist: float = 1.0*, *fraction: float = 0.5*, *smooth: bool = False*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.guided_integrated_gradients)

Calculate saliency map using guided integrated gradients.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **x_steps** (*int**,**optional*) - Steps for gradient calculation.
Defaults to 25.
- **max_dist** (*float**,**optional*) - Maximum distance for gradient
calculation. Defaults to 1.0.
- **fraction** (*float**,**optional*) - Fraction for gradient calculation.
Defaults to 0.5.
- **smooth** (*bool**,**optional*) - Smooth gradients. Defaults to False.

Returns:

Saliency map.

Return type:

np.ndarray

integrated_gradients(*img: ndarray*, *x_steps: int = 25*, *batch_size: int = 20*, *smooth: bool = False*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.integrated_gradients)

Calculate saliency map using integrated gradients.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **x_steps** (*int**,**optional*) - Steps for gradient calculation.
Defaults to 25.
- **max_dist** (*float**,**optional*) - Maximum distance for gradient
calculation. Defaults to 1.0.
- **smooth** (*bool**,**optional*) - Smooth gradients. Defaults to False.

Returns:

Saliency map.

Return type:

np.ndarray

vanilla(*img: ndarray*, *smooth: bool = False*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.vanilla)

Calculate gradient-based saliency map.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **smooth** (*bool**,**optional*) - Smooth gradients. Defaults to False.

Returns:

Saliency map.

Return type:

np.ndarray

xrai(*img: ndarray*, *batch_size: int = 20*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.xrai)

Calculate saliency map using XRAI.

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **batch_size** (*int**,**optional*) - Batch size. Defaults to 20.

Returns:

Saliency map.

Return type:

np.ndarray

xrai_fast(*img: ndarray*, *batch_size: int = 20*, ***kwargs*) → ndarray[[source]](_modules/histox/grad.html#SaliencyMap.xrai_fast)

Calculate saliency map using XRAI (fast implementation).

Parameters:

- **img** (*np.ndarray*) - Pre-processed input image in W, H, C format.
- **batch_size** (*int**,**optional*) - Batch size. Defaults to 20.

Returns:

Saliency map.

Return type:

np.ndarray

Submodule for calculating/displaying pixel attribution (saliency maps).

histox.grad.grayscale(*image_3d*, *vmax=None*, *vmin=None*, *percentile=99*)[[source]](_modules/histox/grad.html#grayscale)

Returns a 3D tensor as a grayscale 2D tensor.
This method sums a 3D tensor across the absolute value of axis=2, and then
clips values at a given percentile.

histox.grad.comparison_plot(*original: ndarray*, *maps: Dict[str, ndarray]*, *cmap: Any = 'plt.cm.gray'*, *n_rows: int = 3*, *n_cols: int = 3*) → None[[source]](_modules/histox/grad/plot_utils.html#comparison_plot)

Plots comparison of many saliency maps for a single image in a grid.

Parameters:

- **original** (*np.ndarray*) - Original (unprocessed) image.
- **maps** (*dict**(**str**,**np.ndarray**)*) - Dictionary mapping saliency map names
to the numpy array maps.
- **cmap** (*matplotlib colormap**,**optional*) - Colormap for maps.
Defaults to plt.cm.gray.

histox.grad.inferno(*img*)[[source]](_modules/histox/grad/plot_utils.html#inferno)

histox.grad.multi_plot(*raw_imgs: List[ndarray]*, *processed_imgs: List[ndarray]*, *method: Callable*, *cmap: str = 'inferno'*, *xlabels: List[str] | None = None*, *ylabels: List[str] | None = None*, ***kwargs*) → None[[source]](_modules/histox/grad/plot_utils.html#multi_plot)

Creates a plot of saliency maps and overlays for a given set of images.

The first row will be the raw images.
The second row will be an overlay of the saliency map and the raw image.
The third row will be the saliency maps.

Parameters:

- **raw_imgs** (*List**[**np.ndarray**]*) - Raw, unprocessed images.
- **processed_imgs** (*List**[**np.ndarray**]*) - Processed images.
- **method** (*Callable*) - Saliency method.
- **cmap** (*str**,**optional*) - Colormap. Defaults to 'inferno'.
- **xlabels** (*Optional**[**List**[**str**]**]**,**optional*) - Labels for x-axis.
Defaults to None.
- **ylabels** (*Optional**[**List**[**str**]**]**,**optional*) - Labels for y-axis.
Defaults to None.

Raises:

- **ValueError** - If length of raw_imgs, processed_imgs are not equal.
- **ValueError** - If xlabels is provided and not a list.
- **ValueError** - If ylabels is provided and not a list.
- **ValueError** - If xlabels is provided and length does not equal raw_imgs.
- **ValueError** - If ylabels is provided and length does not equal raw_imgs.

histox.grad.oranges(*img*)[[source]](_modules/histox/grad/plot_utils.html#oranges)

histox.grad.overlay(*image*, *mask*)[[source]](_modules/histox/grad/plot_utils.html#overlay)

histox.grad.saliency_map_comparison(*orig_imgs: List[ndarray]*, *saliency_fn: List[Callable]*, *process_fn: Callable*, *saliency_labels: List[str] | None = None*, *cmap: str = 'inferno'*, ***kwargs: Any*) → None[[source]](_modules/histox/grad/plot_utils.html#saliency_map_comparison)

Plots several saliency maps for a list of images.

Each row is a unique image.
The first column is the original image. Each column after is a saliency
map generated each of the functions provided to saliency_fn.

Parameters:

- **orig_imgs** (*list**(**np.ndarray**)*) - Original (unprocessed) images for
which to generate saliency maps.
- **saliency_fn** (*list**(**Callable**)*) - List of saliency map functions.
- **process_fn** (*Callable*) - Function for processing images. This function
will be applied to images before images are passed to the
saliency map function.
- **saliency_labels** (*list**(**str**)**,**optional*) - Labels for provided saliency
maps. Defaults to None.
- **cmap** (*str**,**optional*) - Colormap for saliency maps.
Defaults to 'inferno'.