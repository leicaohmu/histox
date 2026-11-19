"""Single source of truth for the HistoX documentation navigation.

Both the top navigation bar (``_templates/histox_navbar.html``) and the left
sidebar (``_templates/histox_sidebar.html``) are rendered from ``NAV``.

* ``menu`` lists the links shown in a section's drop-down. Menu links may
  point anywhere, including pages owned by another section or external URLs.
* ``sidebar`` lists ``(caption, [(page, label), ...])`` groups. The first
  section whose sidebar lists a page *owns* it: that section is highlighted
  in the navbar and its sidebar is shown on the page.
* ``hidden`` pages belong to a section without appearing in its sidebar.

Every document must be owned by a section; ``conf.py`` emits a Sphinx warning
(which fails CI) for documents missing from this file and for entries that
point to documents that do not exist.
"""

NAV = [
    {
        'key': 'get-started',
        'title': 'Get started',
        'menu': [
            ('installation', 'Installation', 'Choose core and optional dependencies.'),
            ('quickstart', 'Quickstart', 'Run a first HistoX workflow.'),
            ('overview', 'Concepts', 'Understand the HistoX workflow.'),
            ('project_setup', 'Project setup', 'Organize slides, data, and outputs.'),
        ],
        'sidebar': [
            ('Get started', [
                ('docs', 'Documentation overview'),
                ('installation', 'Installation'),
                ('quickstart', 'Quickstart'),
                ('overview', 'Concepts'),
                ('project_setup', 'Project setup'),
            ]),
        ],
    },
    {
        'key': 'learn',
        'title': 'Learn',
        'menu': [
            ('auto_examples/index', 'Examples gallery', 'Run complete, downloadable workflows.'),
            ('tutorials', 'Tutorials', 'Task-oriented guides from data to evidence.'),
            ('tutorials/ps3_coadread_reproduction', 'Reproduction protocols',
             'Multi-fold, multi-seed validation of implemented methods.'),
        ],
        'sidebar': [
            ('Examples', [
                ('auto_examples/index', 'Examples gallery'),
                ('auto_examples/basics/plot_open_wsi', 'Open your first WSI'),
                ('auto_examples/feature_extraction/plot_dinov2_wsi', 'DINOv2 features on a WSI'),
                ('auto_examples/multimodal/plot_ps3_crc', 'Train PS3 with HistoX'),
            ]),
            ('Tutorials', [
                ('tutorials', 'Tutorials home'),
                ('tutorials/feature_bags', 'Generate feature bags'),
                ('tutorials/ps3_data_preparation', 'Prepare multimodal PS3 inputs'),
                ('tutorials/ps3_coadread_reproduction', 'Reproduce PS3 on COADREAD'),
            ]),
            ('Compatibility tutorials', [
                ('tutorial1', 'Train a first model'),
                ('tutorial2', 'Control the training loop'),
                ('tutorial3', 'Use a custom architecture'),
                ('tutorial7', 'Add custom augmentations'),
                ('tutorial4', 'Evaluate and build heatmaps'),
                ('tutorial5', 'Create a mosaic map'),
                ('tutorial6', 'Write a custom slide filter'),
                ('tutorial8', 'Train a MIL model'),
            ]),
        ],
        'hidden': [
            'tutorials_training', 'tutorials_evaluation', 'tutorials_slides',
            'tutorials_mil', 'auto_examples/basics/index',
            'auto_examples/feature_extraction/index',
            'auto_examples/multimodal/index',
        ],
    },
    {
        'key': 'user-guide',
        'title': 'User guide',
        'menu': [
            ('user_guide', 'User guide', 'The end-to-end workflow, chapter by chapter.'),
            ('data', '1. Data and cohorts', 'Public data and study datasets.'),
            ('slide_processing', '2. Slide processing', 'Tiling, segmentation, and features.'),
            ('workflows', '3. Modeling', 'Workflows, training, MIL, and SSL.'),
            ('evaluation', '4. Evaluation and interpretation', 'Metrics, uncertainty, and saliency.'),
            ('studio', '5. Tools', 'Inspect slides and models in HistoX Studio.'),
        ],
        'sidebar': [
            (None, [('user_guide', 'User guide home')]),
            ('1. Data and cohorts', [
                ('data', 'Public data'),
                ('datasets_and_val', 'Study datasets'),
            ]),
            ('2. Slide processing', [
                ('slide_processing', 'Slide processing'),
                ('segmentation', 'Tissue segmentation'),
                ('cellseg', 'Cell segmentation'),
                ('features', 'Feature extraction'),
            ]),
            ('3. Modeling', [
                ('workflows', 'Experiment workflows'),
                ('training', 'Supervised training'),
                ('mil', 'Multiple-instance learning'),
                ('ssl', 'Self-supervised learning'),
                ('stylegan', 'Generative models'),
                ('custom_loops', 'Custom training loops'),
            ]),
            ('4. Evaluation and interpretation', [
                ('evaluation', 'Evaluation'),
                ('posthoc', 'Post-hoc analysis'),
                ('uq', 'Uncertainty'),
                ('saliency', 'Saliency and interpretation'),
            ]),
            ('5. Tools', [
                ('studio', 'HistoX Studio'),
            ]),
        ],
    },
    {
        'key': 'methods',
        'title': 'Methods',
        'wide': True,
        'menu': [
            ('methods', 'Methods hub', 'Browse methods by task, maturity, and origin.'),
            ('methods/encoders/index', 'Image encoders', 'Foundation and pretrained tile encoders.'),
            ('methods/multimodal/index', 'Multimodal & survival', 'Combine histology, reports, and omics.'),
            ('mil', 'MIL & weak supervision', 'Learn from slide-level labels.'),
            ('training', 'Classification', 'Tile- and slide-level prediction.'),
            ('segmentation', 'Segmentation', 'Tissue and cell segmentation.'),
            ('stylegan', 'Generative models', 'Synthesis and representation learning.'),
            ('norm', 'Stain normalization', 'Prepare consistent slide inputs.'),
            ('saliency', 'Interpretation', 'Inspect evidence and uncertainty.'),
        ],
        'sidebar': [
            (None, [('methods', 'Methods hub')]),
            ('Image encoders', [
                ('methods/encoders/index', 'Overview'),
                ('methods/encoders/dinov2', 'DINOv2'),
                ('methods/encoders/vit', 'Vision Transformer'),
                ('methods/encoders/virchow', 'Virchow'),
            ]),
            ('Multimodal and survival', [
                ('methods/multimodal/index', 'Overview'),
                ('methods/multimodal/ps3', 'PS3'),
            ]),
        ],
    },
    {
        'key': 'data',
        'title': 'Data',
        'right': True,
        'menu': [
            ('data', 'Public data', 'Discover, download, and cache public cohorts.'),
            ('datasets_and_val', 'Study datasets', 'Build, filter, and split cohorts.'),
            ('data_api', 'histox.data', 'Registry, download planning, and cache API.'),
            ('dataset', 'histox.Dataset', 'Dataset class reference.'),
        ],
        'sidebar': [],
    },
    {
        'key': 'api',
        'title': 'API',
        'right': True,
        'menu': [
            ('api', 'API reference', 'Look up modules, classes, and arguments.'),
            ('project', 'histox.Project', 'Study configuration and outputs.'),
            ('slide', 'histox.WSI', 'Open, tile, and query whole-slide images.'),
            ('workflows_api', 'histox.workflows', 'Tasks, methods, and experiments.'),
            ('mil_module', 'histox.mil', 'Multiple-instance learning.'),
        ],
        'sidebar': [
            (None, [('api', 'API index')]),
            ('Core objects', [
                ('histox', 'histox'),
                ('project', 'histox.Project'),
                ('dataset', 'histox.Dataset'),
                ('model_params', 'histox.ModelParams'),
            ]),
            ('Data and slides', [
                ('data_api', 'histox.data'),
                ('dataset_features', 'DatasetFeatures'),
                ('slide', 'histox.WSI'),
                ('slide_qc', 'histox.slide.qc'),
                ('io', 'histox.io'),
                ('io_torch', 'histox.io.torch'),
            ]),
            ('Models and workflows', [
                ('workflows_api', 'histox.workflows'),
                ('model', 'histox.model'),
                ('model_torch', 'histox.model.torch'),
                ('mil_module', 'histox.mil'),
                ('simclr', 'histox.simclr'),
                ('gan', 'histox.gan'),
                ('grad', 'histox.grad'),
                ('norm', 'histox.norm'),
            ]),
            ('Pathology and utilities', [
                ('heatmap', 'histox.Heatmap'),
                ('mosaic', 'histox.Mosaic'),
                ('slidemap', 'histox.SlideMap'),
                ('histox_cellseg', 'histox.cellseg'),
                ('biscuit', 'histox.biscuit'),
                ('stats', 'histox.stats'),
                ('util', 'histox.util'),
                ('studio_module', 'histox.studio'),
            ]),
            ('Legacy TensorFlow', [
                ('io_tensorflow', 'histox.io.tensorflow'),
                ('model_tensorflow', 'histox.model.tensorflow'),
            ]),
        ],
    },
    {
        'key': 'community',
        'title': 'Community',
        'right': True,
        'menu': [
            ('developer', 'Developer notes', 'Extend HistoX and its internals.'),
            ('extensions', 'Extensions', 'Optional packages built on HistoX.'),
            ('plugins', 'Plugins', 'Register your own components.'),
            ('troubleshooting', 'Troubleshooting', 'Diagnose common problems.'),
            ('https://github.com/leicaohmu/histox', 'GitHub', 'Source code, issues, and pull requests.'),
        ],
        'sidebar': [
            (None, [('developer', 'Developer notes')]),
            ('Extend HistoX', [
                ('extensions', 'Extensions'),
                ('plugins', 'Plugins'),
                ('custom_extractors', 'Custom feature extractors'),
                ('dataloaders', 'Dataloaders'),
                ('tfrecords', 'TFRecords'),
                ('tile_labels', 'Tile labels'),
            ]),
            ('Support', [
                ('troubleshooting', 'Troubleshooting'),
            ]),
        ],
    },
]

# Documents rendered without a section sidebar.
UNOWNED = {
    'index', 'search', 'genindex', 'py-modindex', 'sg_execution_times',
    'auto_examples/sg_execution_times',
}


def page_owners():
    """Map each document name to the key of the section that owns it."""
    owners = {}
    for section in NAV:
        pages = [page for _, items in section['sidebar'] for page, _ in items]
        pages += section.get('hidden', [])
        for page in pages:
            owners.setdefault(page, section['key'])
    return owners


def linked_pages():
    """Every internal document named anywhere in ``NAV``."""
    pages = set(page_owners())
    for section in NAV:
        pages.update(p for p, _, _ in section['menu'] if '://' not in p)
    return pages
