use thiserror::Error;

/// Errors produced by crftag operations.
#[derive(Debug, Error)]
pub enum Error {
    /// The model has not been loaded yet.
    #[error("model is not loaded — call load_model() first")]
    ModelNotLoaded,
    /// Underlying CRF error.
    #[error(transparent)]
    Crf(#[from] crfrs::Error),
    /// Hugging Face Hub download failed.
    #[cfg(feature = "hf-hub")]
    #[error("Hugging Face Hub error: {0}")]
    Hub(String),
}

/// A specialized `Result` type for crftag operations.
pub type Result<T> = std::result::Result<T, Error>;
