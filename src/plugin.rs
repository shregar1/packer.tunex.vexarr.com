//! TuneX plugin contract for the packer module.

use tunex_core::plugin::{
    PluginCapability, PluginId, PluginKind, PluginMetadata, PluginName, PluginStatus,
    PluginVersion, TunexPlugin,
};

use crate::capabilities::packer_capabilities;

/// Stable TuneX packer plugin name.
pub const PACKER_PLUGIN_NAME: &str = "tunex-packer";

/// Stable TuneX packer plugin version.
pub const PACKER_PLUGIN_VERSION: PluginVersion = PluginVersion::new(0, 1, 0);

/// Packer plugin metadata provider.
#[derive(Clone, Debug)]
pub struct PackerPlugin {
    capabilities: Vec<PluginCapability>,
}

impl PackerPlugin {
    /// Creates the default packer plugin descriptor.
    #[must_use]
    pub fn new() -> Self {
        Self {
            capabilities: packer_capabilities(),
        }
    }
}

impl Default for PackerPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl TunexPlugin for PackerPlugin {
    fn metadata(&self) -> PluginMetadata {
        PluginMetadata::new(
            PluginId::new(PACKER_PLUGIN_NAME),
            PluginName::new(PACKER_PLUGIN_NAME),
            PACKER_PLUGIN_VERSION,
            PluginKind::Packer,
            self.capabilities.clone(),
        )
    }

    fn capabilities(&self) -> &[PluginCapability] {
        &self.capabilities
    }

    fn health(&self) -> PluginStatus {
        PluginStatus::Active
    }
}
