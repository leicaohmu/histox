# histox.gan

Submodule used to interface with the PyTorch implementation of StyleGAN2
(maintained separately at [jamesdolezal/stylegan2-histox](https://github.com/jamesdolezal/stylegan2-histox)).

See [Generative Networks (GANs)](stylegan.html#stylegan) for more information on working with GANs.

## StyleGAN2 Interpolator

*class*histox.gan.StyleGAN2Interpolator(*gan_pkl: str*, *start: int*, *end: int*, ***, *device: torch.device | None = None*, *target_um: int | None = None*, *target_px: int | None = None*, *gan_um: int | None = None*, *gan_px: int | None = None*, *noise_mode: str = 'const'*, *truncation_psi: int = 1*, ***gan_kwargs*)[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator)

class_interpolate(*seed: int*, *steps: int = 100*) → Generator[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.class_interpolate)

Sets up a generator that returns images during class embedding
interpolation.

Parameters:

- **seed** (*int*) - Seed for random noise vector.
- **steps** (*int**,**optional*) - Number of steps for interpolation.
Defaults to 100.

Returns:

Generator which yields images (torch.tensor, uint8)
during interpolation.

Return type:

Generator

Yields:

*Generator* - images (torch.tensor, dtype=uint8)

generate(*seed: int | List[int]*, *embedding: torch.Tensor*) → torch.Tensor[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate)

Generate an image from a given embedding.

Parameters:

- **seed** (*int*) - Seed for noise vector.
- **embedding** (*torch.Tensor*) - Class embedding.

Returns:

Image (float32, shape=(1, 3, height, width))

Return type:

torch.Tensor

generate_end(*seed: int*) → torch.Tensor[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_end)

Generate an image from the ending class.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

Image (float32, shape=(1, 3, height, width))

Return type:

torch.Tensor

generate_np_end(*seed: int*) → ndarray[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_np_end)

Generate a numpy image from the ending class.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

Image (uint8, shape=(height, width, 3))

Return type:

np.ndarray

generate_np_from_embedding(*seed: int*, *embedding: torch.Tensor*) → ndarray[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_np_from_embedding)

Generate a numpy image from a given embedding.

Parameters:

- **seed** (*int*) - Seed for noise vector.
- **embedding** (*torch.Tensor*) - Class embedding.

Returns:

Image (uint8, shape=(height, width, 3))

Return type:

np.ndarray

generate_np_start(*seed: int*) → ndarray[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_np_start)

Generate a numpy image from the starting class.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

Image (uint8, shape=(height, width, 3))

Return type:

np.ndarray

generate_start(*seed: int*) → torch.Tensor[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_start)

Generate an image from the starting class.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

Image (float32, shape=(1, 3, height, width))

Return type:

torch.Tensor

generate_tf_end(*seed: int*) → Tuple[tf.Tensor, tf.Tensor][[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_tf_end)

Create a processed Tensorflow image from the GAN output of a given
seed and the ending class embedding.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

A tuple containing

> tf.Tensor: Unprocessed resized image, uint8.
> 
> 
> 
> 
> tf.Tensor: Processed resized image, standardized and normalized.

generate_tf_from_embedding(*seed: int | List[int]*, *embedding: torch.Tensor*) → Tuple[tf.Tensor, tf.Tensor][[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_tf_from_embedding)

Create a processed Tensorflow image from the GAN output from a given
seed and embedding.

Parameters:

- **seed** (*int*) - Seed for noise vector.
- **embedding** (*torch.tensor*) - Class embedding.

Returns:

A tuple containing

> tf.Tensor: Unprocessed resized image, uint8.
> 
> 
> 
> 
> tf.Tensor: Processed resized image, standardized and normalized.

generate_tf_start(*seed: int*) → Tuple[tf.Tensor, tf.Tensor][[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.generate_tf_start)

Create a processed Tensorflow image from the GAN output of a given
seed and the starting class embedding.

Parameters:

**seed** (*int*) - Seed for noise vector.

Returns:

A tuple containing

> tf.Tensor: Unprocessed image (tf.Tensor), uint8.
> 
> 
> 
> 
> tf.Tensor: Processed image (tf.Tensor), standardized and normalized.

interpolate_and_predict(*seed: int*, *steps: int = 100*, *outcome_idx: int = 0*) → Tuple[List, ...][[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.interpolate_and_predict)

Interpolates between starting and ending classes for a seed,
recording raw images, processed images, and predictions.

Parameters:

- **seed** (*int*) - Seed for random noise vector.
- **steps** (*int**,**optional*) - Number of steps during interpolation.
Defaults to 100.

Returns:

Raw images, processed images, and predictions.

Return type:

Tuple[List, ...]

linear_interpolate(*seed: int*, *steps: int = 100*) → Generator[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.linear_interpolate)

Sets up a generator that returns images during linear label
interpolation.

Parameters:

- **seed** (*int*) - Seed for random noise vector.
- **steps** (*int**,**optional*) - Number of steps for interpolation.
Defaults to 100.

Returns:

Generator which yields images (torch.tensor, uint8)
during interpolation.

Return type:

Generator

Yields:

*Generator* - images (torch.tensor, dtype=uint8)

plot_comparison(*seeds: int | List[int]*, *titles: List[str] | None = None*) → None[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.plot_comparison)

Plots side-by-side comparison of images from the starting
and ending interpolation classes.

Parameters:

**seeds** (*int**or**list**(**int**)*) - Seeds to display.

seed_search(*seeds: List[int]*, *batch_size: int = 32*, *verbose: bool = False*, *outcome_idx: int = 0*, *concordance_thresholds: Iterable[float] | None = None*) → DataFrame[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.seed_search)

Generates images for starting and ending classes for many seeds,
calculating layer activations from a set classifier.

Parameters:

- **seeds** (*List**[**int**]*) - Seeds.
- **batch_size** (*int**,**optional*) - Batch size for GAN during generation.
Defaults to 32.
- **verbose** (*bool**,**optional*) - Verbose output. Defaults to False.

Raises:

**Exception** - If classifier model has not been been set with
 .set_classifier()

Returns:

Dataframe of results.

Return type:

pd.core.frame.DataFrame

set_classifier(*path: str*, *layers: str | List[str] | None = None*, ***kwargs*) → None[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.set_classifier)

Configures a classifier model to be used for generating features
and predictions during interpolation.

Parameters:

- **path** (*str*) - Path to trained model.
- **layers** (*Union**[**str**,**List**[**str**]**]**,**optional*) - Layers from which to
calculate activations for interpolated images.
Defaults to None.

z(*seed: int | List[int]*) → torch.Tensor[[source]](_modules/histox/gan/interpolate.html#StyleGAN2Interpolator.z)

Returns a noise tensor for a given seed.

Parameters:

**seed** (*int*) - Seed.

Returns:

Noise tensor for the corresponding seed.

Return type:

torch.tensor

## Utility functions

histox.gan.utils.crop(*img: torch.Tensor*, *gan_um: int*, *gan_px: int*, *target_um: int*) → Any[[source]](_modules/histox/gan/utils.html#crop)

Process a batch of raw GAN output, converting to a Tensorflow tensor.

Parameters:

- **img** (*torch.Tensor*) - Raw batch of GAN images.
- **gan_um** (*int**,**optional*) - Size of gan output images, in microns.
- **gan_px** (*int**,**optional*) - Size of gan output images, in pixels.
- **target_um** (*int**,**optional*) - Size of target images, in microns.
Will crop image to meet this target.

Returns:

Cropped image.

histox.gan.utils.noise_tensor(*seed: int*, *z_dim: int*) → torch.Tensor[[source]](_modules/histox/gan/utils.html#noise_tensor)

Creates a noise tensor based on a given seed and dimension size.

Parameters:

- **seed** (*int*) - Seed.
- **z_dim** (*int*) - Dimension of noise vector to create.

Returns:

Noise vector of shape (1, z_dim)

Return type:

torch.Tensor