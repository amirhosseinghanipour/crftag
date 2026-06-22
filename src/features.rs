//! Feature extraction for Persian POS tagging and chunking.
//!
//! Feature extraction for POS tagging and chunking.

const PUNCTUATION: &[&str] = &[
    "\"", "#", "(", ")", "*", ",", "-", ".", "/", ":", "[", "]",
    "«", "»", "،", ";", "?", "!",
];

fn is_punc(word: &str) -> bool {
    PUNCTUATION.contains(&word)
}

fn all_digits(word: &str) -> bool {
    !word.is_empty() && word.chars().all(|c| c.is_ascii_digit() || matches!(c, '۰'..='۹'))
}

/// Extracts POS-tagging features for `sentence[index]`.
///
/// Returns a `Vec<String>` of active feature strings in the format
/// `"key=value"` (or just `"key"` for boolean features).  These strings are
/// passed directly to the CRF model.
pub fn pos_features(sentence: &[&str], index: usize) -> Vec<String> {
    let word = sentence[index];
    let n = sentence.len();

    let prev = if index > 0 { sentence[index - 1] } else { "" };
    let two_prev = if index > 1 { sentence[index - 2] } else { "" };
    let next = if index + 1 < n { sentence[index + 1] } else { "" };
    let two_next = if index + 2 < n { sentence[index + 2] } else { "" };

    // Character-level prefix/suffix (up to 3 chars, safe for multi-byte)
    let chars: Vec<char> = word.chars().collect();
    let prefix1 = chars.iter().take(1).collect::<String>();
    let prefix2 = chars.iter().take(2).collect::<String>();
    let prefix3 = chars.iter().take(3).collect::<String>();
    let suffix1 = chars.iter().rev().take(1).collect::<String>();
    let suffix2: String = chars.iter().rev().take(2).collect::<Vec<_>>().into_iter().rev().collect();
    let suffix3: String = chars.iter().rev().take(3).collect::<Vec<_>>().into_iter().rev().collect();

    let mut f: Vec<String> = vec![
        format!("word={}", word),
        format!("prefix-1={}", prefix1),
        format!("prefix-2={}", prefix2),
        format!("prefix-3={}", prefix3),
        format!("suffix-1={}", suffix1),
        format!("suffix-2={}", suffix2),
        format!("suffix-3={}", suffix3),
        format!("prev_word={}", prev),
        format!("two_prev_word={}", two_prev),
        format!("next_word={}", next),
        format!("two_next_word={}", two_next),
        format!("is_numeric={}", all_digits(word)),
        format!("prev_is_numeric={}", all_digits(prev)),
        format!("next_is_numeric={}", all_digits(next)),
        format!("is_punc={}", is_punc(word)),
        format!("prev_is_punc={}", is_punc(prev)),
        format!("next_is_punc={}", is_punc(next)),
    ];

    if index == 0 {
        f.push("is_first".to_string());
    }
    if index == n - 1 {
        f.push("is_last".to_string());
    }

    f
}

/// Extracts chunking features for `words[index]` given pre-computed `pos_tags`.
///
/// Includes all POS features plus POS context.
pub fn chunk_features(words: &[&str], pos_tags: &[&str], index: usize) -> Vec<String> {
    let n = words.len();
    let mut f = pos_features(words, index);

    let pos = pos_tags[index];
    let prev_pos = if index > 0 { pos_tags[index - 1] } else { "" };
    let next_pos = if index + 1 < n { pos_tags[index + 1] } else { "" };

    f.push(format!("pos={}", pos));
    f.push(format!("prev_pos={}", prev_pos));
    f.push(format!("next_pos={}", next_pos));
    f
}

/// Converts a full sentence into the feature-per-token representation for POS tagging.
pub fn sentence_pos_features(sentence: &[&str]) -> Vec<Vec<String>> {
    (0..sentence.len())
        .map(|i| pos_features(sentence, i))
        .collect()
}

/// Converts a tagged sentence into chunk features.
pub fn sentence_chunk_features(
    words: &[&str],
    pos_tags: &[&str],
) -> Vec<Vec<String>> {
    (0..words.len())
        .map(|i| chunk_features(words, pos_tags, i))
        .collect()
}
