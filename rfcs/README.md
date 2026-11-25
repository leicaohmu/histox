# HistoX RFCs

An RFC (request for comments) is a short design document for a change that
affects HistoX's public API, its core abstractions, or how components fit
together. Small fixes and self-contained features do not need one.

## Process

1. Copy the structure of an existing RFC into `rfcs/NNNN-short-title.md`, using the next free number.
2. Open a pull request with the RFC alone, before writing the implementation. Discussion happens on the pull request.
3. When maintainers agree, set the status to **Accepted** and merge the RFC.
4. Implement in separate pull requests that link back to the RFC. When the work is complete, set the status to **Implemented**.

An RFC that is not pursued is merged with status **Rejected** or **Withdrawn**
and a one-line reason, so the reasoning stays on record.

## Index

| RFC | Title | Status |
|---|---|---|
| [0001](0001-core-abstractions.md) | Core abstractions | Accepted |
| [0002](0002-framework-building-blocks.md) | From application to framework | Draft |
