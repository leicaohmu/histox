# HistoX Provenance

HistoX is a modified fork of
[Slideflow](https://github.com/slideflow/slideflow), with the upstream Git history
retained in this repository. A renamed path is not, by itself, evidence that the
file was independently authored by HistoX contributors.

## Repository lineage

- Upstream project: Slideflow
- Upstream repository: <https://github.com/slideflow/slideflow>
- HistoX package-rename commit:
  `2fcd66d56e9ccfe88345205199ab78e54c4aa44c` (2026-04-11)
- Revision used for this provenance inventory:
  `d4318a63ddc9399fade2e92757b351e53ba697f2`

The package-rename commit records 240 renamed files, 9 added files, 5 deleted
files and 9 modified files. Later HistoX commits include both new work and
changes to inherited code.

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
