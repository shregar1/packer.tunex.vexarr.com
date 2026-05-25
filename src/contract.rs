//! Config-backed TuneX packer contract factory.

use tunex_core::traits::SequencePacker;

use crate::{
    config::{PackerConfig, PackingStrategy},
    packer::{BestFitPacker, GreedyPacker, TunexSequencePacker},
    PackerResult,
};

/// Builds packer implementations from validated TuneX packer config.
#[derive(Clone, Debug)]
pub struct PackerContract {
    config: PackerConfig,
}

impl PackerContract {
    /// Creates a packer contract from config.
    ///
    /// # Errors
    ///
    /// Returns an error when config validation fails.
    pub fn new(config: PackerConfig) -> PackerResult<Self> {
        config.validate()?;
        Ok(Self { config })
    }

    /// Returns the validated config.
    #[must_use]
    pub const fn config(&self) -> &PackerConfig {
        &self.config
    }

    /// Builds a synchronous packer trait object.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured packer cannot be constructed.
    pub fn build_sync(&self) -> PackerResult<Box<dyn TunexSequencePacker>> {
        match self.config.strategy {
            PackingStrategy::BestFit => BestFitPacker::new(self.config.clone())
                .map(|packer| Box::new(packer) as Box<dyn TunexSequencePacker>),
            PackingStrategy::Greedy | PackingStrategy::PadOnly => {
                GreedyPacker::new(self.config.clone())
                    .map(|packer| Box::new(packer) as Box<dyn TunexSequencePacker>)
            }
        }
    }

    /// Builds a core sequence packer trait object.
    ///
    /// # Errors
    ///
    /// Returns an error when the configured packer cannot be constructed.
    pub fn build_core(&self) -> PackerResult<Box<dyn SequencePacker>> {
        match self.config.strategy {
            PackingStrategy::BestFit => BestFitPacker::new(self.config.clone())
                .map(|packer| Box::new(packer) as Box<dyn SequencePacker>),
            PackingStrategy::Greedy | PackingStrategy::PadOnly => {
                GreedyPacker::new(self.config.clone())
                    .map(|packer| Box::new(packer) as Box<dyn SequencePacker>)
            }
        }
    }
}

impl TryFrom<PackerConfig> for PackerContract {
    type Error = crate::PackerError;

    fn try_from(config: PackerConfig) -> Result<Self, Self::Error> {
        Self::new(config)
    }
}
