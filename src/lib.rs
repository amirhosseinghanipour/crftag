//! # crftag
//!
//! CRF-based POS tagger and shallow parser (chunker) built on [`crfrs`].
//!
//! Fully independent — drop it into any Rust project.
//!
//! ## Features
//!
//! - [`POSTagger`] — structured-perceptron tagger
//! - [`Chunker`] — IOB chunker (shallow parser)
//! - [`RuleBasedChunker`] — grammar-based chunker, no model file required
//! - [`tree2brackets`] — formats a parse result as a bracketed string
//! - [`features`] — raw feature extraction (for custom models)
//! - [`crfrs`] — re-exported; use for training/loading models directly
//!
//! ## Quick start
//!
//! ```no_run
//! use crftag::POSTagger;
//!
//! let mut tagger = POSTagger::new();
//! tagger.load_model("pos_tagger.model").unwrap();
//! let tagged = tagger.tag(&["من", "به", "مدرسه", "رفتم", "."]).unwrap();
//! ```
//!
//! ## Training
//!
//! ```no_run
//! use crftag::{POSTagger, crfrs::TrainConfig};
//!
//! let corpus: Vec<Vec<(String, String)>> = vec![];  // (word, tag) pairs
//! POSTagger::train_and_save(&corpus, "pos_tagger.model", TrainConfig::default()).unwrap();
//! ```

#![warn(missing_docs)]

pub mod chunker;
/// Error types for crftag.
pub mod error;
pub mod features;
pub mod tagger;

// Re-export crfrs so callers can use TrainConfig without an explicit dep.
pub use crfrs;

pub use chunker::{iob_to_tree, tree2brackets, Chunker, ChunkedSentence, ChunkedToken, ParseNode, RuleBasedChunker};
pub use error::{Error, Result};
pub use tagger::POSTagger;
