use crftag::{POSTagger, crfrs::TrainConfig};

// Helpers ----------------------------------------------------------------

/// Repeats a fixed 3-token Persian sentence `n` times as a training corpus.
fn corpus(n: usize) -> Vec<Vec<(String, String)>> {
    let sent = vec![
        ("علی".to_string(), "N".to_string()),
        ("رفت".to_string(), "V".to_string()),
        (".".to_string(), "PUNC".to_string()),
    ];
    vec![sent; n]
}

fn tmp_path(name: &str) -> std::path::PathBuf {
    std::env::temp_dir().join(name)
}

// Tests ------------------------------------------------------------------

#[test]
fn train_save_load_tag_cycle() {
    let path = tmp_path("crftag_train_tag.json");
    POSTagger::train_and_save(&corpus(40), &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = POSTagger::new();
    tagger.load_model(&path).unwrap();

    let result = tagger.tag(&["علی", "رفت", "."]).unwrap();
    assert_eq!(result.len(), 3);
    assert_eq!(result[0], ("علی", "N".to_string()));
    assert_eq!(result[1], ("رفت", "V".to_string()));
    assert_eq!(result[2], (".", "PUNC".to_string()));

    std::fs::remove_file(&path).ok();
}

#[test]
fn evaluate_accuracy_on_training_data() {
    let data = corpus(40);
    let path = tmp_path("crftag_eval.json");
    POSTagger::train_and_save(&data, &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = POSTagger::new();
    tagger.load_model(&path).unwrap();

    let acc = tagger.evaluate(&data).unwrap();
    // After 10 epochs on a perfectly consistent mini-corpus, accuracy must be perfect.
    assert_eq!(acc, 1.0, "expected 1.0 accuracy on training data, got {acc}");

    std::fs::remove_file(&path).ok();
}

#[test]
fn tag_empty_sentence_returns_empty() {
    let path = tmp_path("crftag_empty.json");
    POSTagger::train_and_save(&corpus(10), &path, TrainConfig { epochs: 3 }).unwrap();

    let mut tagger = POSTagger::new();
    tagger.load_model(&path).unwrap();

    assert!(tagger.tag(&[]).unwrap().is_empty());
    std::fs::remove_file(&path).ok();
}

#[test]
fn model_not_loaded_returns_error() {
    let tagger = POSTagger::new();
    let err = tagger.tag(&["سلام"]);
    assert!(err.is_err());
}

#[test]
fn tag_sents_matches_individual_tag_calls() {
    let path = tmp_path("crftag_sents.json");
    POSTagger::train_and_save(&corpus(40), &path, TrainConfig { epochs: 10 }).unwrap();

    let mut tagger = POSTagger::new();
    tagger.load_model(&path).unwrap();

    let sents = vec![
        vec!["علی", "رفت", "."],
        vec!["علی", "."],
    ];
    let batch = tagger.tag_sents(&sents).unwrap();
    let singles: Vec<_> = sents.iter().map(|s| tagger.tag(s).unwrap()).collect();

    assert_eq!(batch, singles);
    std::fs::remove_file(&path).ok();
}
