# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.1] - 2026-06-23

### Added

- Optional `hf-hub` feature that enables `POSTagger::load_from_hub` and `Chunker::load_from_hub`, allowing models to be downloaded directly from the Hugging Face Hub without managing local paths.
- `Error::Hub(String)` variant (gated behind `hf-hub` feature) for Hub download failures.

## [0.1.0] - 2026-06-23

### Added

- Initial release with `POSTagger` and `Chunker` backed by CRF models via `crfrs`.
