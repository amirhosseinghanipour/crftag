# crftag

CRF-based POS tagger and shallow parser (chunker), written in Rust.

Built on [`crfrs`](https://github.com/amirhosseinghanipour/crfrs) — bring your own corpus and label set.

## Features

- `POSTagger` — structured-perceptron POS tagger
- `Chunker` — IOB shallow parser (CRF-based)
- `RuleBasedChunker` — grammar-rule chunker; no model file required
- `tree2brackets` — formats a parse tree as a bracketed string
- `features` module — raw feature extraction for custom models
- Re-exports [`crfrs`] for training and model management

## Usage

```toml
[dependencies]
crftag = "0.1"

```

### POS Tagging

```rust
use crftag::POSTagger;

let mut tagger = POSTagger::new();
tagger.load_model("pos_tagger.model").unwrap();

let tagged = tagger.tag(&["من", "به", "مدرسه", "رفتم", "."]).unwrap();
// → [("من", "PRON"), ("به", "ADP"), ("مدرسه", "NOUN"), ("رفتم", "VERB"), (".", "PUNCT")]
```

### Training a POS Model

```rust
use crftag::{POSTagger, crfrs::TrainConfig};

// Each sentence is a Vec of (word, tag) pairs
let corpus: Vec<Vec<(String, String)>> = vec![
    vec![("من".into(), "PRON".into()), ("رفتم".into(), "VERB".into())],
];

POSTagger::train_and_save(&corpus, "pos_tagger.model", TrainConfig::default()).unwrap();
```

### Chunking

```rust
use crftag::Chunker;

let mut chunker = Chunker::new();
chunker.load_model("chunker.model").unwrap();

let tagged = &[("نامه", "NOUN,EZ"), ("ایشان", "PRON"), ("را", "ADP"), ("داشتم", "VERB")];
let chunked = chunker.tag(tagged).unwrap();
// → [("نامه", "NOUN,EZ", "B-NP"), ("ایشان", "PRON", "I-NP"), ("را", "ADP", "B-POSTP"), ("داشتم", "VERB", "B-VP")]
```

### Rule-Based Chunking (no model)

```rust
use crftag::{RuleBasedChunker, tree2brackets};

let chunker = RuleBasedChunker::new();
let tagged = &[
    ("نامه", "NOUN,EZ"),
    ("ایشان", "PRON"),
    ("را", "ADP"),
    ("داشتم", "VERB"),
    (".", "PUNCT"),
];
let tree = chunker.parse(tagged);
println!("{}", tree2brackets(&tree));
// → [نامه ایشان NP] [را POSTP] [داشتم VP] .
```

## API

| Item | Description |
|---|---|
| `POSTagger::new()` | Create a tagger (no model loaded) |
| `POSTagger::universal()` | Create a tagger that maps to universal POS tags |
| `POSTagger::load_model(path)` | Load a trained model |
| `POSTagger::tag(tokens)` | Tag a single sentence |
| `POSTagger::tag_sents(sentences)` | Tag multiple sentences |
| `POSTagger::train_and_save(corpus, path, config)` | Train and persist a model |
| `POSTagger::evaluate(corpus)` | Token-level accuracy |
| `Chunker::new()` | Create a chunker (no model loaded) |
| `Chunker::load_model(path)` | Load a trained model |
| `Chunker::tag(tagged)` | Assign IOB chunk tags |
| `Chunker::parse(tagged)` | Parse into a tree |
| `Chunker::train_and_save(corpus, path, config)` | Train and persist a model |
| `Chunker::evaluate(corpus)` | Token-level IOB accuracy |
| `RuleBasedChunker::new()` | Grammar-rule chunker, no model needed |
| `RuleBasedChunker::parse(tagged)` | Parse into a tree |
| `tree2brackets(tree)` | Format a parse tree as a bracketed string |
| `features::pos_features(sentence, i)` | Extract POS features for token `i` |
| `features::chunk_features(words, pos, i)` | Extract chunk features for token `i` |

## License

MIT
