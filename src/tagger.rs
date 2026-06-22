//! CRF-based Persian POS tagger.

use std::path::Path;
use crfrs::{train, PerceptronModel, TrainConfig};
use crate::{error::{Error, Result}, features::sentence_pos_features};

/// Part-of-speech tagger for Persian text.
///
/// Uses a structured perceptron (via [`crfrs`]) with features matching
/// the Collins 2002 averaged perceptron.  The model must be trained and saved separately,
/// then loaded with [`POSTagger::load_model`].
///
/// # Examples
///
/// ```no_run
/// use crftag::POSTagger;
///
/// let mut tagger = POSTagger::new();
/// tagger.load_model("pos_tagger.model").unwrap();
///
/// let tags = tagger.tag(&["من", "به", "مدرسه", "رفتم", "."]).unwrap();
/// // → [("من", "PRON"), ("به", "ADP"), ("مدرسه", "NOUN"), ("رفتم", "VERB"), (".", "PUNCT")]
/// ```
pub struct POSTagger {
    model: Option<PerceptronModel>,
    universal: bool,
}

impl POSTagger {
    /// Creates a tagger with no loaded model.
    pub fn new() -> Self {
        Self { model: None, universal: false }
    }

    /// Creates a tagger that maps POS tags to the universal tagset.
    pub fn universal() -> Self {
        Self { model: None, universal: true }
    }

    /// Loads a model saved with [`train_and_save`](Self::train_and_save).
    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.model = Some(PerceptronModel::load(path.as_ref())?);
        Ok(())
    }

    /// Tags a single sentence.  Returns `(word, pos_tag)` pairs.
    pub fn tag<'a>(&self, tokens: &[&'a str]) -> Result<Vec<(&'a str, String)>> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        if tokens.is_empty() {
            return Ok(vec![]);
        }
        let features = sentence_pos_features(tokens);
        let tags = model.tag(&features);
        let tags = if self.universal {
            tags.into_iter().map(|t| to_universal(&t)).collect()
        } else {
            tags
        };
        Ok(tokens.iter().copied().zip(tags).collect())
    }

    /// Tags multiple sentences.
    pub fn tag_sents<'a>(
        &self,
        sentences: &[Vec<&'a str>],
    ) -> Result<Vec<Vec<(&'a str, String)>>> {
        sentences.iter().map(|s| self.tag(s)).collect()
    }

    /// Trains from labeled data and saves the model to `model_path`.
    ///
    /// `tagged` — slice of sentences; each sentence is `Vec<(word, pos_tag)>`.
    pub fn train_and_save(
        tagged: &[Vec<(String, String)>],
        model_path: impl AsRef<Path>,
        config: TrainConfig,
    ) -> Result<()> {
        let (examples, labels) = prepare_training(tagged);
        let model = train(&examples, labels, config);
        model.save(model_path.as_ref())?;
        Ok(())
    }

    /// Evaluates token-level accuracy.
    pub fn evaluate(&self, tagged_sents: &[Vec<(String, String)>]) -> Result<f64> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        let (mut correct, mut total) = (0usize, 0usize);
        for sent in tagged_sents {
            let words: Vec<&str> = sent.iter().map(|(w, _)| w.as_str()).collect();
            let gold: Vec<&str> = sent.iter().map(|(_, t)| t.as_str()).collect();
            let pred = model.tag(&sentence_pos_features(&words));
            for (p, g) in pred.iter().zip(gold.iter()) {
                if p == g { correct += 1; }
                total += 1;
            }
        }
        Ok(if total == 0 { 1.0 } else { correct as f64 / total as f64 })
    }
}

impl Default for POSTagger { fn default() -> Self { Self::new() } }

fn prepare_training(
    tagged: &[Vec<(String, String)>],
) -> (Vec<(Vec<Vec<String>>, Vec<String>)>, Vec<String>) {
    let mut label_set = std::collections::BTreeSet::new();
    for sent in tagged {
        for (_, tag) in sent { label_set.insert(tag.clone()); }
    }
    let labels: Vec<String> = label_set.into_iter().collect();
    let examples = tagged.iter().map(|sent| {
        let words: Vec<&str> = sent.iter().map(|(w, _)| w.as_str()).collect();
        let gold: Vec<String> = sent.iter().map(|(_, t)| t.clone()).collect();
        (sentence_pos_features(&words), gold)
    }).collect();
    (examples, labels)
}

fn to_universal(tag: &str) -> String {
    tag.split(',').next().unwrap_or(tag).to_string()
}
