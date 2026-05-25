use tunex_core::{plugin::TunexPlugin, types::TokenizedSample};
use tunex_packer::{
    PackerConfig, PackerContract, PackerPlugin, PackingStrategy, ASSISTANT_ONLY_LOSS,
    BEST_FIT_PACKING, DETERMINISTIC_PACKING, GPU_READY_LAYOUT, GREEDY_PACKING,
    LOSS_MASK_GENERATION, PACKER_PLUGIN_NAME, PARALLEL_PACKING, SEQUENCE_PACKING,
};

#[test]
fn packer_plugin_advertises_core_metadata_and_capabilities() {
    let plugin = PackerPlugin::new();
    let metadata = plugin.metadata();
    let capabilities = plugin
        .capabilities()
        .iter()
        .map(|capability| capability.as_str())
        .collect::<Vec<_>>();

    assert_eq!(metadata.id.as_str(), PACKER_PLUGIN_NAME);
    assert_eq!(metadata.name.as_str(), PACKER_PLUGIN_NAME);
    assert_eq!(metadata.kind, tunex_core::plugin::PluginKind::Packer);
    assert_eq!(plugin.health(), tunex_core::plugin::PluginStatus::Active);
    assert!(capabilities.contains(&SEQUENCE_PACKING));
    assert!(capabilities.contains(&GREEDY_PACKING));
    assert!(capabilities.contains(&BEST_FIT_PACKING));
    assert!(capabilities.contains(&GPU_READY_LAYOUT));
    assert!(capabilities.contains(&LOSS_MASK_GENERATION));
    assert!(capabilities.contains(&ASSISTANT_ONLY_LOSS));
    assert!(capabilities.contains(&PARALLEL_PACKING));
    assert!(capabilities.contains(&DETERMINISTIC_PACKING));
}

#[test]
fn packer_contract_builds_strategy_backed_sync_packer() {
    let contract = PackerContract::new(PackerConfig {
        strategy: PackingStrategy::BestFit,
        max_seq_len: 8,
        batch_size: 2,
        ..PackerConfig::default()
    })
    .expect("contract");
    let packer = contract.build_sync().expect("sync packer");
    let samples = vec![
        TokenizedSample::new(vec![1, 2, 3]),
        TokenizedSample::new(vec![4, 5]),
    ];

    let output = packer.pack_output(&samples).expect("packed");

    assert_eq!(output.waste.packed_samples, 2);
    assert_eq!(output.waste.real_tokens, 5);
}

#[test]
fn packer_contract_builds_core_sequence_packer_trait_object() {
    let contract = PackerContract::new(PackerConfig {
        strategy: PackingStrategy::Greedy,
        max_seq_len: 8,
        ..PackerConfig::default()
    })
    .expect("contract");

    assert!(contract.build_core().is_ok());
}
