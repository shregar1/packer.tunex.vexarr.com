use tunex_core::{plugin::TunexPlugin, types::TokenizedSample};
use tunex_packer::{
    GreedyPacker, PackerConfig, PackerPlugin, TunexSequencePacker, PACKER_PLUGIN_NAME,
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

    let batch = packer.pack_samples(&samples).expect("packed");

    assert_eq!(batch.token_ids, vec![1, 2, 3, 4, 5, 0, 0, 0]);
    assert_eq!(batch.sequence_offsets, vec![0, 3]);
    assert_eq!(batch.labels, vec![1, 2, 3, 4, 5, -100, -100, -100]);
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
