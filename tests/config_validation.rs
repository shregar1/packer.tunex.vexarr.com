use tunex_packer::{LabelMode, PackerConfig, PackingStrategy, ValidatePackerConfig};

#[test]
fn valid_packer_config_passes_validation() {
    let config = PackerConfig {
        strategy: PackingStrategy::BestFit,
        max_seq_len: 4096,
        batch_size: 8,
        pad_to_multiple_of: Some(8),
        drop_oversized: true,
        label_mode: LabelMode::CausalLm,
        ignore_index: -100,
        assistant_only_loss: true,
        role_boundaries: true,
        deterministic: true,
        ..PackerConfig::default()
    };

    assert!(config.validation_report().is_valid());
    assert_eq!(config.layout().batch_size, 8);
    assert_eq!(config.masks().label_mode, LabelMode::CausalLm);
}

#[test]
fn config_validation_rejects_zero_sizes_and_bad_padding() {
    let config = PackerConfig {
        max_seq_len: 10,
        batch_size: 0,
        pad_to_multiple_of: Some(6),
        ..PackerConfig::default()
    };
    let report = config.validation_report();

    assert!(!report.is_valid());
    assert!(report.errors.iter().any(|error| error.path == "batch_size"));
    assert!(report
        .errors
        .iter()
        .any(|error| error.path == "pad_to_multiple_of"));
}

#[test]
fn config_validation_rejects_assistant_only_without_boundaries() {
    let config = PackerConfig {
        assistant_only_loss: true,
        role_boundaries: false,
        ..PackerConfig::default()
    };
    let report = config.validation_report();

    assert!(!report.is_valid());
    assert!(report
        .errors
        .iter()
        .any(|error| error.path == "assistant_only_loss"));
}

#[test]
fn config_validation_rejects_ambiguous_oversized_policy() {
    let config = PackerConfig {
        drop_oversized: false,
        skip_long_samples: true,
        ..PackerConfig::default()
    };
    let report = config.validation_report();

    assert!(!report.is_valid());
    assert!(report
        .errors
        .iter()
        .any(|error| error.path == "drop_oversized"));
}

#[test]
fn enum_parsers_accept_expected_names() {
    assert_eq!(
        PackingStrategy::parse("best_fit"),
        Some(PackingStrategy::BestFit)
    );
    assert_eq!(
        PackingStrategy::parse("pad-only"),
        Some(PackingStrategy::PadOnly)
    );
    assert_eq!(LabelMode::parse("causal_lm"), Some(LabelMode::CausalLm));
    assert_eq!(LabelMode::parse("ignore-all"), Some(LabelMode::IgnoreAll));
}
