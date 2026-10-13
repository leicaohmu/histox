Troubleshooting
===============

Logging
*******

HistoX logs to the terminal by default. Importing HistoX does not create a log
file in the current working directory.

To include debug messages in the terminal output, set
``HX_LOGGING_LEVEL=10`` before starting Python:

.. code-block:: bash

    export HX_LOGGING_LEVEL=10

File logging is opt-in. Set ``HX_LOG_FILE`` to an explicit destination:

.. code-block:: bash

    export HX_LOG_FILE=/path/to/logs/histox.log

Alternatively, enable it programmatically:

.. code-block:: python

    from histox.util import addLoggingFileHandler

    addLoggingFileHandler("/path/to/logs/histox.log")

To check for errors in your environment or installation, you can also use the test script ``test.py``, which uses the testing suite ``histox.test.TestSuite`` to execute all pipeline functions on a set of provided slides.

Testing
*******

To troubleshoot environment or installation issues, start by running unit tests,
which do not require any sample slides. Use the ``test.py`` script without any
arguments:

.. code-block:: bash

    $ python3 test.py

For a more comprehensive test of all pipeline functions, provide a path to a directory containing sample slides via ``--slides``, setting ``--all=True`` to run all tests:

.. code-block:: bash

    $ python3 test.py --slides=/path/to/slides --all=True

Individual tests can be manually skipped with the following syntax:

.. code-block:: bash

    $ python3 test.py --slides=/path/to/slides --all=True --reader=False

To view a list of all tests that will be run (and thus can be skipped), pass the argument ``--help``.

Issue Reporting
***************

If the issue is still unclear, please submit an Issue on the `project Github page <https://github.com/leicaohmu/histox/issues>`_. Be sure to include the following information:

* The version of HistoX you are using, which can be displayed with ``hx.about()``:

.. code-block:: bash

    $ python3 -c "import histox; histox.about()"
    ╭=======================╮
    │       HistoX       │
    │    Version: 2.1.0     │
    │  Backend: tensorflow  │
    │ Slide Backend: cucim  │
    │ https://histox.readthedocs.io/en/latest │
    ╰=======================╯

* The active deep learning backend (``hx.backend()``) and slide backend (``hx.slide_backend()``)
* The version of Python you are using (``python3 --version``)
* The operating system you are using (``uname -a``)
