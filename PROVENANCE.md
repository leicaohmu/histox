# HistoX Provenance

HistoX is a modified fork of
[Slideflow](https://github.com/slideflow/slideflow), with the upstream Git history
retained in this repository. A renamed path is not, by itself, evidence that the
file was independently authored by HistoX contributors.

## Repository lineage

- Upstream project: Slideflow
- Upstream repository: <https://github.com/slideflow/slideflow>
- HistoX package-rename commit:
  `2fcd81a44d24c4c7dfaceb67543ede114225f584` (2026-04-11). Earlier versions
  of this file gave this commit as `2fcd66d…`, which does not exist in the
  repository; the file counts below match `2fcd81a4`.
- Module alias commit (`sf` → `hx` throughout the source):
  `58901d97b09d9bce7b08df561eb72a9d51b1bcc0` (2026-04-11)
- Revision used for this provenance inventory:
  `d4318a63ddc9399fade2e92757b351e53ba697f2`

The package-rename commit records 240 renamed files, 9 added files, 5 deleted
files and 9 modified files. Later HistoX commits include both new work and
changes to inherited code.

## Share of Slideflow-derived source

Measured on `develop` at HistoX 0.3.0 (`cb986f13`) with `git blame -w`. Each
line of Python under `histox/` is attributed to the commit that last changed it:

| | Lines | Share |
|---|---:|---:|
| All Python lines under `histox/` | 94,880 | |
| Last changed by a Slideflow commit (before the rename commit) | 78,649 | 83% |
| Last changed by a HistoX commit | 16,231 | 17% |

The HistoX share includes lines touched only by the rename and alias commits
above, so 83% is a lower bound on Slideflow-derived lines. Modules written for
HistoX, such as `histox/workflows/` and `histox/data/`, have no
Slideflow-derived lines. Re-run the measurement when reclassifying modules.

## Upstream release tags

The repository used to carry Slideflow's release tags (`0.8.2` to `3.0.2`).
They were removed from the HistoX repository after HistoX 0.3.0. Read the Docs
and version-sorting tools treated `3.0.2` as the newest release, ahead of
HistoX's own `v*` tags. The tagged commits are part of Slideflow's history and
remain available, with their tags, in the
[upstream repository](https://github.com/slideflow/slideflow/tags). Before the
removal, all 68 tags were checked against the upstream repository; each one
exists there and points to the same commit.

Tags marked "side branch" point to Slideflow release-branch commits that are not
ancestors of the rename commit. After the tags were removed, these commits are
no longer referenced in the HistoX repository.

To restore a tag here: `git tag <name> <commit> && git push origin <name>`.

| Tag | Commit | Note |
|---|---|---|
| `3.0.2` | `c596d2cf1bc1a22784ab77df4b506273c4c0684b` | |
| `3.0.1` | `004d9d0379db83772fd4ba60ef56135cde221073` | |
| `3.0.0` | `106190191e261730d1993539e20c851185870199` | |
| `2.3.2` | `67db53bfde2564b975d49b1b4701bbdc01640327` | |
| `2.3.1` | `b9982f478f13842030dd67e80de49efc82ba1f27` | |
| `2.3.0` | `b7781b9c48e726d5892120a4757f45185f073213` | |
| `2.2.2` | `3b8d16f054991117e055ef653699f80343768b02` | |
| `2.2.1-post1` | `e62005a122717f5bef061c1b4b87dec8a9aca457` | |
| `2.2.1` | `6f9919b7039ae88ad1b09f0eb46a7aec0e6c4e9b` | |
| `2.2.0` | `b756fc0adb6c298ab441382464b30475e0287ca2` | |
| `2.1.1` | `3fd84da3a8a7b2b6bc202e74bec3fe8d63a9c6ef` | |
| `2.1.0` | `0c158549cb4b8f0a25f113b914367d999948e1e7` | |
| `2.0.5` | `ae6ad0e8937207efe60d23a400e88bf12f5db719` | |
| `2.0.4` | `26ce5cf0d34b65514223967ffc0628d0b5031d05` | |
| `2.0.3-post1` | `95f96082e8a3a81ae8bd15b7b9fef5aa5f0b08ef` | |
| `2.0.3` | `b64be73fa730ab23f9434cab6b857c9c3ec0c8b0` | |
| `2.0.2.post1` | `5ea8c7ab2f30763a20c1cb069e6cd60dc471c5ef` | |
| `2.0.2` | `204a302fa0942fc84b611828b43f4412761e33cd` | |
| `2.0.1` | `3495a878316e9e6ba4525c3a225827817f5198e9` | |
| `2.0.0-rc0` | `01780363db053c9ee6e990748ca85468f4312d5d` | |
| `2.0.0-beta2` | `8095030e34ac71c71ac9db5d0c573656bd6dd3d2` | |
| `2.0.0-beta1` | `6f56a6aa2a6da6260b08bac9d0808f713f9bd129` | |
| `2.0.0` | `79294168e51e9f5d49ee18ca7cc08db09f8bd078` | |
| `1.5.4` | `6dd80a38e8aa1e4f31686e5afcafed5b9fcfdb9c` | |
| `1.5.3` | `1527b2847472fc8416300de70152d82a2640b53c` | |
| `1.5.2` | `3c28115a6f26d722b298fb97e93c477b23b9b95e` | |
| `1.5.1` | `5bade1aae9ef6c0acfb54d14e50bf97bc513dbde` | |
| `1.5.0` | `4cb45f44a92eb7da63ba4b80553309afb15758cf` | |
| `1.4.4` | `d9dc9975d3e69909cce0922a009c87671d2e7677` | side branch |
| `1.4.3` | `0a6d0ad159b9b6ec45b621a35f68fbc38b3f10cd` | side branch |
| `1.4.2` | `04c16b055b38ad7d36fd910376a07caf4d9b12df` | side branch |
| `1.4.1` | `c42adfc16551530c2802dfd062c070f81daf1d83` | |
| `1.4.0-rc0` | `2da96d40ac8e77f06e30c6b00d4a58e399637c7e` | |
| `1.4.0.post1` | `1a65eee09b92d57a459764238dfaf3dbd4723410` | |
| `1.4.0` | `c322478e3caaa81a7ad22c4294d33c66a1471f33` | |
| `1.3.3` | `b02b940269361d5ea3cfb62d23a973c847e5f2df` | |
| `1.3.2` | `bb7d5de965c6d05952521d9e0770fc5036692059` | |
| `1.3.1` | `269c2f3ae42ff336af9d19b1e3856ba11b0d6505` | |
| `1.3.0-rc0` | `2685825db5226c66cd97b683ee5118b1843538ab` | |
| `1.3.0` | `1cfe580dab72a67d7bf9396be0654e36dd008c94` | side branch |
| `1.2.5` | `3a3c4cb616e992518da8541695fe4ae3e7f3a85e` | |
| `1.2.4` | `206d23b68d2bcb8b392e908f9f81030d8100916f` | |
| `1.2.3` | `cf1e1eb0980cc7111ac50dee848fda965f537f76` | |
| `1.2.2` | `865324e44c54d431e124ed65b6b65357cae7722c` | |
| `1.2.1` | `9bd0396e18ceffb9e58df62bc4cc12dc3a8a078f` | |
| `1.2.0-rc0` | `f6811d728238d896c2f080eae4292b1d12be9a2f` | |
| `1.2.0` | `d03424d6b0e444456cde595cf0472c7e1499ea29` | |
| `1.1.4` | `839d1890595b0a7b428a77e9523ab3869f60b3c4` | side branch |
| `1.1.3` | `40f59e09bf65ec46533541a890f2803752832617` | |
| `1.1.2` | `5797857fe47beab53fd6ed4efa11b99ef5f7ebd0` | |
| `1.1.1` | `01d271d5ed81d1f8da9d0d12946e05e0787159fe` | |
| `1.1.0-rc2` | `242d1a526b0a56cedae78d64c2f73bfa66173d55` | side branch |
| `1.1.0-rc1` | `7b365b9589173d0be35b9f3fc6b0feace7098af8` | |
| `1.1.0` | `76b88120b06647eef768d41977dd2d4123378e6e` | |
| `1.0.8` | `65126a7dfdc8bc498983a54baea57a249e49d5c7` | side branch |
| `1.0.7` | `0f4ba1f10269f15b93975fb74f97fb0205a50619` | side branch |
| `1.0.6` | `0df3e9664130752e477c1213714fae0c43786e71` | |
| `1.0.5` | `08e994a561470b968c8a768b96eb28977fd358c1` | |
| `1.0.4` | `be4f3725ad9ac37ea3c0849bf72b954fc26de262` | |
| `1.0.3` | `652981cedd3293a482af963381af0944d6224aa1` | |
| `1.0.2` | `73468e64848028d50b58f5aca0c8033d15d0f41a` | |
| `1.0.1` | `66847ebbe4ee38614bbf82ed9f9fb31ece243040` | |
| `1.0.0` | `cfda049d9a14d80d43458bf65a26617ac55ab043` | |
| `0.12.2` | `824d8b1491d2391ce5f9269ba586d1e57978e7ec` | |
| `0.11.2` | `a3de27922f6e673dbc08c3ed1ffcc43f84e26012` | |
| `0.10.1` | `7ea38a5bac7b71b2ff61bbd8c3424d7845a95733` | |
| `0.9.2` | `17d74537cfbb8a66cf74cbfac75696f7b8f682a6` | side branch |
| `0.8.2` | `0d01ad8b939f6e76f3f9846f22c714e5b3c4330b` | side branch |

## How source is classified

Repository content belongs to one of four provenance classes:

1. **Slideflow-derived code** — inherited from or modified after the Slideflow
   fork.
2. **HistoX-authored code** — introduced by HistoX contributors after the fork.
3. **Embedded third-party code** — copied or adapted from another named project.
4. **External artifacts and integrations** — dependencies, plugins, model
   weights and datasets obtained separately from this source tree.

The top-level [LICENSE](LICENSE) records the repository's Apache-2.0 license
declaration. Embedded third-party components remain subject to their own license
terms. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for the current
component inventory and unresolved review items.

## Contribution requirement

Every contribution that copies or adapts third-party material must record:

- the upstream project and source URL;
- the exact upstream version or commit when known;
- the upstream license and required notice;
- whether the imported material was modified; and
- tests that cover the imported behavior.

Do not add copied source, model weights, datasets, fonts, icons or other assets
without a documented redistribution basis. External datasets and weights are
not automatically covered by the HistoX repository license.

## Independent paper-derived implementations

`histox/mil/models/ps3.py` is a HistoX-authored implementation of the prototype
construction and multimodal fusion equations described by Raza et al., "PS3: A
Multimodal Transformer Integrating Pathology Reports with Histology Images and
Biological Pathways for Cancer Survival Prediction" (ICCV 2025). No source was
copied from `manahilr/PS3` or `leicaohmu/PS3-CRC-reproduction`; neither
repository contained a detected source-code license at the time of integration.
The HistoX implementation does not claim compatibility with upstream
checkpoints or reproduce the paper's reported performance without a separate
controlled experiment.
