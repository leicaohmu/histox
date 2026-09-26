# Creating a HistoX Plugin

HistoX has been designed to be extensible, and we encourage users to contribute their own plugins to the HistoX ecosystem. Plugins can be used to add new functionality to HistoX, such as new feature extractors or new model architectures. This page provides an overview of how to create and use plugins with HistoX.

## MIL Model Registration

As discussed in [Custom Architectures](mil.html#custom-mil), HistoX supports the registration of custom MIL models. This is done by using the `register_model` decorator to register a custom MIL model.

For example, suppose you have a custom MIL model called `MyMILModel` that you want to register with HistoX. You've already designed the model such that it meets HistoX's MIL [requirements](custom_mil). Now you want to make it available for use directly within HistoX. You can accomplish this by using the `register_model` decorator:

```
from histox.model.mil import register_model

@register_model
def my_mil_model(**kwargs):
 from . import MyMILModel
 return MyMILModel(**kwargs)
```

Once this code is run, the custom MIL model will be available for use with HistoX:

```
import histox as hx

model = hx.build_mil_model("my_mil_model")
```

## Feature Extractors

Similarly, HistoX supports the integration of custom feature extractors via the `register_torch` and `register_tf` decorators. Please see our detailed [developer note](custom_extractors) for more information on how to create and register custom extractors. Briefly, you can register a custom feature extractor with HistoX as follows:

```
from histox.model.extractors import register_torch

@register_torch
def my_foundation_model(**kwargs):
 from . import MyFoundationModel
 return MyFoundationModel(**kwargs)
```

## Creating a Plugin

Once you have a custom MIL model or feature extractor that you want to integrate with HistoX, you can create a plugin to make it available to other users.

HistoX supports external plugins via standard Python entry points, allowing you to publish your own package that integrates with HistoX.

In your package's `setup.py` file, use the "entry_points" key to connect with the HistoX plugin interface:

```
...,
entry_points={
 'histox.plugins': [
 'extras = my_package:register_extras',
 ],
},
```

Then, in your package's root `__init__.py` file, write a `register_extras()` function that does any preparation needed to initialize or import your model.

(in `my_package/__init__.py`)

```
def register_extras():
 # Import the model, and do any other necessary preparation.
 # If my_module contains the @register_model decorator,
 # the model will be registered with HistoX automatically.
 from . import my_module

print("Registered MyFoundationModel")
```

You can then build and distribute your plugin, and once installed, the registration with HistoX will happen automatically:

```
pip install my_package
```

```
import histox as hx

model = hx.build_feature_extractor("my_foundation_model")
```

For compatibility examples, see the upstream [Slideflow-GPL](https://github.com/slideflow/slideflow-gpl) and [Slideflow-NonCommercial](https://github.com/slideflow/slideflow-noncommercial) repositories. Their package and import names remain unchanged until HistoX-owned replacements are published.