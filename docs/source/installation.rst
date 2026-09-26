Installation
============

HistoX is under active development. The current ``0.2.x`` API is pre-stable,
so pin the package version in reproducible projects. Python 3.9 is the current
development and continuous-integration baseline; the package metadata still
declares Python 3.7 or newer while compatibility is being audited.

Development installation
------------------------

Installing the repository in editable mode is the recommended way to test the
current code before a formal release:

.. code-block:: bash

   git clone https://github.com/leicaohmu/histox.git
   cd histox
   conda create -n histox python=3.9
   conda activate histox
   python -m pip install --upgrade pip
   python -m pip install -e ".[torch]"

Changes made inside the cloned repository are then available immediately in
the ``histox`` environment; reinstalling after every edit is not required.

Published package
-----------------

To evaluate the most recent package published on PyPI with the PyTorch
backend:

.. code-block:: bash

   python -m pip install "histox[torch]"

The repository may contain changes newer than the published package. Check
``hx.__version__`` and pin the version used by an experiment.

Installation groups
-------------------

.. list-table:: Optional dependency groups
   :header-rows: 1
   :widths: 38 62

   * - Command
     - Purpose
   * - ``python -m pip install histox``
     - Base dependencies; install a model backend separately.
   * - ``python -m pip install "histox[torch]"``
     - PyTorch backend and related training tools. Recommended for new work.
   * - ``python -m pip install "histox[tf]"``
     - Legacy TensorFlow compatibility for inherited workflows.
   * - ``python -m pip install "histox[torch,cucim]"``
     - PyTorch and cuCIM; install the matching CuPy build separately.
   * - ``python -m pip install "histox[torch,cucim-cuda12]"``
     - PyTorch, cuCIM, and the CUDA 12 CuPy build.
   * - ``python -m pip install "histox[torch,cucim-cuda11]"``
     - PyTorch, cuCIM, and the CUDA 11 CuPy build.

Do not install multiple ``cupy-*`` variants in one environment. Match the
CuPy build to the CUDA runtime reported by ``nvidia-smi``.

System requirements
-------------------

HistoX is developed primarily on Linux. macOS can be used for development and
CPU workflows; Windows support is experimental.

Whole-slide image reading requires one of the following:

* `libvips <https://www.libvips.org/>`_ with its Python binding, for broad
  scanner-format support; or
* `cuCIM <https://docs.rapids.ai/api/cucim/stable/>`_ in a compatible NVIDIA
  CUDA environment.

Model training also requires the selected deep-learning backend. A CUDA-capable
GPU is strongly recommended for training, but it is not required to import
HistoX or create a project.

Verify the installation
-----------------------

Run this before creating a project:

.. code-block:: python

   import histox as hx

   print("HistoX:", hx.__version__)
   print("model backend:", hx.backend())
   print("slide backend:", hx.slide_backend())

For the current development line, the first line should report ``0.2.1``.
The backend values depend on the packages installed in the environment.

Select backends explicitly
--------------------------

HistoX prefers PyTorch when both supported model backends are installed. Set
the model and slide readers before starting Python when an explicit choice is
needed:

.. code-block:: bash

   export HX_BACKEND=torch
   export HX_SLIDE_BACKEND=libvips

Use ``HX_BACKEND=tensorflow`` only for legacy workflows that require it. The
``HX_BACKEND`` and ``HX_SLIDE_BACKEND`` names are the supported HistoX
environment interface.

Next, follow the :doc:`quickstart` to create a local project without
downloading a public pathology cohort.
