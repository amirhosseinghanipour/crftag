//! Shallow parser (chunker) for Persian text.

use std::path::Path;
use crfrs::{train, PerceptronModel, TrainConfig};
use crate::{error::{Error, Result}, features::sentence_chunk_features};

/// A single token with POS and IOB chunk tags.
pub type ChunkedToken = (String, String, String);
/// A fully parsed sentence.
pub type ChunkedSentence = Vec<ChunkedToken>;

/// A shallow parse tree node.
#[derive(Debug, Clone)]
pub enum ParseNode {
    /// A phrase (NP, VP, PP, …) grouping one or more tokens.
    Phrase {
        /// Phrase type label (e.g. `"NP"`, `"VP"`).
        label: String,
        /// `(word, POS)` pairs belonging to the phrase.
        tokens: Vec<(String, String)>,
    },
    /// An un-chunked token.
    Token {
        /// Surface form.
        word: String,
        /// POS tag.
        tag: String,
    },
}

/// Converts an IOB-tagged sentence into a parse tree.
pub fn iob_to_tree(chunked: &[ChunkedToken]) -> Vec<ParseNode> {
    let mut tree: Vec<ParseNode> = Vec::new();
    for (word, pos, iob) in chunked {
        if iob.starts_with('B') {
            tree.push(ParseNode::Phrase {
                label: iob[2..].to_string(),
                tokens: vec![(word.clone(), pos.clone())],
            });
        } else if iob.starts_with('I') {
            match tree.last_mut() {
                Some(ParseNode::Phrase { tokens, .. }) => {
                    tokens.push((word.clone(), pos.clone()));
                }
                _ => tree.push(ParseNode::Token { word: word.clone(), tag: pos.clone() }),
            }
        } else {
            tree.push(ParseNode::Token { word: word.clone(), tag: pos.clone() });
        }
    }
    tree
}

/// Formats a parse tree as a bracketed string.
///
/// # Examples
///
/// ```
/// use crftag::chunker::{iob_to_tree, tree2brackets};
///
/// let chunked = vec![
///     ("نامه".into(), "NOUN,EZ".into(), "B-NP".into()),
///     ("ایشان".into(), "PRON".into(),   "I-NP".into()),
///     ("را".into(),    "ADP".into(),    "B-POSTP".into()),
///     ("داشتم".into(), "VERB".into(),   "B-VP".into()),
///     (".".into(),     "PUNCT".into(),  "O".into()),
/// ];
/// let tree = iob_to_tree(&chunked);
/// assert_eq!(tree2brackets(&tree), "[نامه ایشان NP] [را POSTP] [داشتم VP] .");
/// ```
pub fn tree2brackets(tree: &[ParseNode]) -> String {
    tree.iter()
        .map(|node| match node {
            ParseNode::Phrase { label, tokens } => {
                let words: Vec<&str> = tokens.iter().map(|(w, _)| w.as_str()).collect();
                format!("[{} {}]", words.join(" "), label)
            }
            ParseNode::Token { word, .. } => word.clone(),
        })
        .collect::<Vec<_>>()
        .join(" ")
}

// ---------------------------------------------------------------------------
// CRF Chunker
// ---------------------------------------------------------------------------

/// CRF-based IOB chunker for Persian text.
pub struct Chunker {
    model: Option<PerceptronModel>,
}

impl Chunker {
    /// Creates a chunker with no loaded model.
    pub fn new() -> Self { Self { model: None } }

    /// Loads a model saved with [`train_and_save`](Self::train_and_save).
    pub fn load_model(&mut self, path: impl AsRef<Path>) -> Result<()> {
        self.model = Some(PerceptronModel::load(path.as_ref())?);
        Ok(())
    }

    /// Assigns IOB chunk tags to a POS-tagged sentence.
    pub fn tag<'a>(&self, tagged: &[(&'a str, &'a str)]) -> Result<ChunkedSentence> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        if tagged.is_empty() { return Ok(vec![]); }
        let words: Vec<&str> = tagged.iter().map(|(w, _)| *w).collect();
        let pos: Vec<&str>   = tagged.iter().map(|(_, p)| *p).collect();
        let iob = model.tag(&sentence_chunk_features(&words, &pos));
        Ok(tagged.iter().zip(iob).map(|((w, p), iob)| (w.to_string(), p.to_string(), iob)).collect())
    }

    /// Tags multiple sentences.
    pub fn tag_sents<'a>(&self, sents: &[Vec<(&'a str, &'a str)>]) -> Result<Vec<ChunkedSentence>> {
        sents.iter().map(|s| self.tag(s)).collect()
    }

    /// Parses a tagged sentence into a tree.
    pub fn parse<'a>(&self, tagged: &[(&'a str, &'a str)]) -> Result<Vec<ParseNode>> {
        Ok(iob_to_tree(&self.tag(tagged)?))
    }

    /// Trains and saves a chunker model.
    pub fn train_and_save(
        tagged: &[ChunkedSentence],
        model_path: impl AsRef<Path>,
        config: TrainConfig,
    ) -> Result<()> {
        let (examples, labels) = prepare_chunk_training(tagged);
        let model = train(&examples, labels, config);
        model.save(model_path.as_ref())?;
        Ok(())
    }

    /// Evaluates token-level IOB accuracy.
    pub fn evaluate(&self, test: &[ChunkedSentence]) -> Result<f64> {
        let model = self.model.as_ref().ok_or(Error::ModelNotLoaded)?;
        let (mut correct, mut total) = (0usize, 0usize);
        for sent in test {
            let words: Vec<&str> = sent.iter().map(|(w, _, _)| w.as_str()).collect();
            let pos: Vec<&str>   = sent.iter().map(|(_, p, _)| p.as_str()).collect();
            let gold: Vec<&str>  = sent.iter().map(|(_, _, i)| i.as_str()).collect();
            let pred = model.tag(&sentence_chunk_features(&words, &pos));
            for (p, g) in pred.iter().zip(gold.iter()) {
                if p == g { correct += 1; }
                total += 1;
            }
        }
        Ok(if total == 0 { 1.0 } else { correct as f64 / total as f64 })
    }
}

impl Default for Chunker { fn default() -> Self { Self::new() } }

// ---------------------------------------------------------------------------
// Rule-based chunker
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
enum Cat { N, Ne, V, AJ, AJe, DET, DETe, NUM, NUMe, P, ADV, ADVe, PRO, CL, RES, RESe, POSTP, Other }

fn cat(tag: &str) -> Cat {
    match tag {
        // Specific EZ forms must come before the starts_with guards that would shadow them.
        "N,EZ" | "NOUN,EZ" => Cat::Ne,
        t if t == "N" || t.starts_with("N,") || t == "NOUN" || t.starts_with("NOUN,") => Cat::N,
        t if t.starts_with('V') || t.starts_with("VERB") => Cat::V,
        "AJ,EZ" | "ADJ,EZ" => Cat::AJe,
        t if t == "AJ" || t.starts_with("AJ,") || t == "ADJ" || t.starts_with("ADJ,") => Cat::AJ,
        "DET" => Cat::DET, "DET,EZ" => Cat::DETe,
        "NUM" => Cat::NUM, "NUM,EZ" => Cat::NUMe,
        t if t.starts_with('P') && !t.starts_with("PR") && !t.starts_with("PU") => Cat::P,
        "ADV" => Cat::ADV, "ADV,EZ" => Cat::ADVe,
        t if t.starts_with("PRO") || t.starts_with("PRON") => Cat::PRO,
        "CL" => Cat::CL, "RES" => Cat::RES, "RES,EZ" => Cat::RESe,
        "POSTP" => Cat::POSTP,
        _ => Cat::Other,
    }
}

fn is_np(c: &Cat) -> bool {
    matches!(c, Cat::DET | Cat::DETe | Cat::N | Cat::Ne | Cat::NUM | Cat::NUMe | Cat::AJe | Cat::PRO | Cat::CL | Cat::RES | Cat::RESe)
}

/// Grammar-rule chunker — requires no trained model.
pub struct RuleBasedChunker;

impl RuleBasedChunker {
    /// Creates a new rule-based chunker.
    pub fn new() -> Self { Self }

    /// Parses a POS-tagged sentence into a tree.
    pub fn parse(&self, tagged: &[(&str, &str)]) -> Vec<ParseNode> {
        let cats: Vec<Cat> = tagged.iter().map(|(_, t)| cat(t)).collect();
        let n = tagged.len();
        let mut iob: Vec<String> = vec!["O".to_string(); n];
        let mut i = 0;
        while i < n {
            if cats[i] == Cat::V {
                iob[i] = "B-VP".to_string();
                i += 1;
                continue;
            }
            if is_np(&cats[i]) {
                iob[i] = "B-NP".to_string();
                i += 1;
                while i < n && is_np(&cats[i]) { iob[i] = "I-NP".to_string(); i += 1; }
                continue;
            }
            if cats[i] == Cat::AJ { iob[i] = "B-ADJP".to_string(); i += 1; continue; }
            if matches!(cats[i], Cat::ADV | Cat::ADVe) { iob[i] = "B-ADVP".to_string(); i += 1; continue; }
            if cats[i] == Cat::P {
                iob[i] = "B-PP".to_string(); i += 1;
                while i < n && cats[i] == Cat::P { iob[i] = "I-PP".to_string(); i += 1; }
                continue;
            }
            if cats[i] == Cat::POSTP { iob[i] = "B-POSTP".to_string(); i += 1; continue; }
            i += 1;
        }
        let chunked: ChunkedSentence = tagged.iter().zip(iob).map(|((w, p), iob)| (w.to_string(), p.to_string(), iob)).collect();
        iob_to_tree(&chunked)
    }
}

impl Default for RuleBasedChunker { fn default() -> Self { Self::new() } }

fn prepare_chunk_training(tagged: &[ChunkedSentence]) -> (Vec<(Vec<Vec<String>>, Vec<String>)>, Vec<String>) {
    let mut label_set = std::collections::BTreeSet::new();
    for sent in tagged { for (_, _, iob) in sent { label_set.insert(iob.clone()); } }
    let labels: Vec<String> = label_set.into_iter().collect();
    let examples = tagged.iter().map(|sent| {
        let words: Vec<&str> = sent.iter().map(|(w, _, _)| w.as_str()).collect();
        let pos: Vec<&str>   = sent.iter().map(|(_, p, _)| p.as_str()).collect();
        let gold: Vec<String> = sent.iter().map(|(_, _, iob)| iob.clone()).collect();
        (sentence_chunk_features(&words, &pos), gold)
    }).collect();
    (examples, labels)
}
