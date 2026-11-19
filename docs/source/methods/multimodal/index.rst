Multimodal methods
==================

These methods combine histology with non-image modalities. Each guide states
the exact tensor contract, endpoint definition, provenance, and validation
boundary.

PS3 learning path
-----------------

* :doc:`ps3` — understand the architecture, equations, tensor contract, and
  public classes.
* :doc:`Prepare multimodal inputs </tutorials/ps3_data_preparation>` — convert
  WSI files, report text, RNA, and outcomes into patient-aligned inputs.
* :doc:`Train one patient-level fold
  </auto_examples/multimodal/plot_ps3_crc>` — learn the HistoX workflow through
  Python objects.
* :doc:`Reproduce TCGA COADREAD
  </tutorials/ps3_coadread_reproduction>` — run the complete five-fold,
  three-seed protocol.

.. toctree::
   :maxdepth: 1

   ps3
