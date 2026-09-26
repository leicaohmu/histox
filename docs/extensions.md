# Extensions

The core `histox` package is licensed under the Apache-2.0 license.
Additional functionality is distributed in separate packages according
to their licensing terms.

## HistoX-NonCommercial (CC BY-NC 4.0)

Non-commercial extensions including BISCUIT, StyleGAN3, HistoSSL, PLIP,
GigaPath, and UNI.

```
pip install histox[noncommercial]
# 或直接安装
pip install slideflow-noncommercial
```

After installation:

```
import histox
histox.biscuit # BISCUIT uncertainty quantification
histox.stylegan3 # StyleGAN3 GAN training
```

## HistoX-GPL (GPL-3.0)

GPL-licensed extensions including CLAM, RetCCL, and CTransPath.

```
pip install histox[gpl]
# 或直接安装
pip install slideflow-gpl
```

After installation:

```
import histox
histox.clam # CLAM MIL model
```

Note

These extensions are **not** included in the default `histox`
package due to their licensing terms. Please review the license
of each extension before use.