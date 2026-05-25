use tunex_core::{plugin::TunexPlugin, types::TokenizedSample};
use tunex_packer::{
    BestFitPacker, GreedyPacker, PackerConfig, PackerPlugin, PackingStrategy, TunexSequencePacker,
    PACKER_PLUGIN_NAME,
};

#[test]
fn greedy_packer_builds_padded_batch() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 8,
        pad_token_id: 0,
        ignore_index: -100,
        ..PackerConfig::default()
    })
    .expect("packer");
    let samples = vec![
        TokenizedSample::new(vec![1, 2, 3]),
        TokenizedSample::new(vec![4, 5]),
    ];

    let output = packer.pack_output(&samples).expect("packed");
    let batch = &output.batch;

    assert_eq!(batch.token_ids, vec![1, 2, 3, 4, 5, 0, 0, 0]);
    assert_eq!(batch.sequence_offsets, vec![0, 3]);
    assert_eq!(batch.labels, vec![1, 2, 3, 4, 5, -100, -100, -100]);
    assert_eq!(output.layout.attention_mask, vec![1, 1, 1, 1, 1, 0, 0, 0]);
    assert_eq!(output.layout.position_ids, vec![0, 1, 2, 0, 1, 0, 0, 0]);
    assert_eq!(output.boundaries().len(), 2);
}

#[test]
fn greedy_packer_skips_long_samples() {
    let packer = GreedyPacker::new(PackerConfig {
        max_seq_len: 4,
        skip_long_samples: true,
        ..PackerConfig::default()
    })
    .expect("packer");
    let samples = vec![
        TokenizedSample::new(vec![1, 2, 3, 4, 5]),
        TokenizedSample::new(vec![6, 7]),
    ];

    let output = packer.pack_output(&samples).expect("packed");

    assert_eq!(output.batch.token_ids, vec![6, 7, 0, 0]);
    assert_eq!(output.skipped.len(), 1);
    assert_eq!(output.skipped[0].sample_index, 0);
}

#[test]
fn best_fit_packer_minimizes_padding_waste() {
    let packer = BestFitPacker::new(PackerConfig {
        strategy: PackingStrategy::BestFit,
        max_seq_len: 8,
        pad_token_id: 0,
        ignore_index: -100,
        deterministic: true,
        drop_oversized: true,
        ..PackerConfig::default()
    })
    .expect("packer");
    let samples = vec![
        TokenizedSample::new(vec![1, 2, 3, 4, 5]),
        TokenizedSample::new(vec![6, 7, 8]),
        TokenizedSample::new(vec![9, 10, 11, 12]),
    ];

    let output = packer.pack_output(&samples).expect("packed");

    assert_eq!(output.waste.packed_sequences, 2);
    assert_eq!(output.waste.packed_samples, 3);
    assert_eq!(output.waste.real_tokens, 12);
    assert_eq!(output.waste.padding_tokens, 4);
    assert_eq!(output.waste.samples_per_packed_sequence(), 1.5);
    assert_eq!(output.batch.token_ids.len(), 16);
    assert_eq!(
        output
            .layout
            .attention_mask
            .iter()
            .filter(|mask| **mask == 1)
            .count(),
        12
    );
    assert_eq!(output.layout.boundaries.len(), 3);
}

#[test]
fn packer_plugin_exposes_core_metadata() {
    let plugin = PackerPlugin::new();
    let metadata = plugin.metadata();

    assert_eq!(metadata.id.as_str(), PACKER_PLUGIN_NAME);
    assert_eq!(metadata.kind, tunex_core::plugin::PluginKind::Packer);
    assert!(plugin
        .capabilities()
        .iter()
        .any(|capability| capability.as_str() == "sequence-packing"));
}
