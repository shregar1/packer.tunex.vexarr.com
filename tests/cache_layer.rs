use tunex_core::types::TokenizedSample;
use tunex_packer::{
    BatchFingerprint, CacheTarget, GreedyPacker, MemoryPackingCache, PackerConfig,
    PackingCacheEntry, PackingCacheKey, PackingCachePolicy, TunexSequencePacker,
};

#[test]
fn cache_key_includes_batch_fingerprint_and_config() {
    let samples = vec![TokenizedSample::new(vec![1, 2, 3])];
    let config = PackerConfig {
        max_seq_len: 8,
        pad_token_id: 0,
        ..PackerConfig::default()
    };

    let key =
        PackingCacheKey::from_samples(CacheTarget::PackedBatch, "tokenizer-a", &config, &samples);
    let same =
        PackingCacheKey::from_samples(CacheTarget::PackedBatch, "tokenizer-a", &config, &samples);
    let different = PackingCacheKey::from_samples(
        CacheTarget::PackedBatch,
        "tokenizer-a",
        &config,
        &[TokenizedSample::new(vec![9])],
    );

    assert_eq!(key, same);
    assert_ne!(key.batch_fingerprint, different.batch_fingerprint);
    assert_eq!(
        key.batch_fingerprint,
        BatchFingerprint::from_samples(&samples)
    );
}

#[test]
fn memory_cache_tracks_hits_misses_and_eviction() {
    let config = PackerConfig::default();
    let samples = vec![TokenizedSample::new(vec![1])];
    let key =
        PackingCacheKey::from_samples(CacheTarget::PackedBatch, "tokenizer-a", &config, &samples);
    let second_key = PackingCacheKey::from_samples(
        CacheTarget::PackedBatch,
        "tokenizer-a",
        &config,
        &[TokenizedSample::new(vec![2])],
    );
    let packer = GreedyPacker::new(config).expect("packer");
    let output = packer.pack_output(&samples).expect("packed");
    let mut cache = MemoryPackingCache::new(PackingCachePolicy::Lru { max_entries: 1 });

    assert!(cache.get(&key).is_none());
    cache.insert(
        key.clone(),
        PackingCacheEntry::PackedBatch(Box::new(output)),
    );
    assert!(matches!(
        cache.get(&key),
        Some(PackingCacheEntry::PackedBatch(_))
    ));
    cache.insert(
        second_key.clone(),
        PackingCacheEntry::AttentionMask(vec![1]),
    );

    assert_eq!(cache.len(), 1);
    assert!(cache.get(&key).is_none());
    assert!(cache.stats().hits >= 1);
    assert!(cache.stats().misses >= 2);
    assert_eq!(cache.stats().evictions, 1);
}

#[test]
fn disabled_cache_does_not_store_entries() {
    let samples = vec![TokenizedSample::new(vec![1])];
    let config = PackerConfig::default();
    let key =
        PackingCacheKey::from_samples(CacheTarget::AttentionMask, "tokenizer-a", &config, &samples);
    let mut cache = MemoryPackingCache::new(PackingCachePolicy::Disabled);

    cache.insert(key.clone(), PackingCacheEntry::AttentionMask(vec![1]));

    assert!(cache.is_empty());
    assert!(cache.get(&key).is_none());
}
