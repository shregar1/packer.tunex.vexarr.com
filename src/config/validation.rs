//! Packer config validation.

use super::PackerConfig;

/// One config validation error.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ConfigValidationError {
    /// Dot path of the invalid config value.
    pub path: String,
    /// Human-readable validation message.
    pub message: String,
}

impl ConfigValidationError {
    /// Creates a validation error.
    #[must_use]
    pub fn new(path: impl Into<String>, message: impl Into<String>) -> Self {
        Self {
            path: path.into(),
            message: message.into(),
        }
    }
}

/// Packer config validation report.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ConfigValidationReport {
    /// Validation errors.
    pub errors: Vec<ConfigValidationError>,
}

impl ConfigValidationReport {
    /// Returns true when no validation errors exist.
    #[must_use]
    pub fn is_valid(&self) -> bool {
        self.errors.is_empty()
    }

    fn push(&mut self, path: impl Into<String>, message: impl Into<String>) {
        self.errors.push(ConfigValidationError::new(path, message));
    }
}

impl std::fmt::Display for ConfigValidationReport {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if self.errors.is_empty() {
            return formatter.write_str("configuration is valid");
        }

        for (index, error) in self.errors.iter().enumerate() {
            if index > 0 {
                formatter.write_str("; ")?;
            }
            write!(formatter, "{}: {}", error.path, error.message)?;
        }

        Ok(())
    }
}

/// Validation contract for packer configs.
pub trait ValidatePackerConfig {
    /// Validates this config.
    fn validation_report(&self) -> ConfigValidationReport;
}

impl ValidatePackerConfig for PackerConfig {
    fn validation_report(&self) -> ConfigValidationReport {
        let mut report = ConfigValidationReport::default();

        if self.max_seq_len == 0 {
            report.push("max_seq_len", "max_seq_len must be greater than zero");
        }
        if self.batch_size == 0 {
            report.push("batch_size", "batch_size must be greater than zero");
        }
        if let Some(multiple) = self.pad_to_multiple_of {
            if multiple == 0 {
                report.push(
                    "pad_to_multiple_of",
                    "pad_to_multiple_of must be greater than zero",
                );
            } else if self.max_seq_len != 0 && !self.max_seq_len.is_multiple_of(multiple) {
                report.push(
                    "pad_to_multiple_of",
                    "max_seq_len must be divisible by pad_to_multiple_of",
                );
            }
        }
        if self.ignore_index >= 0 {
            report.push("ignore_index", "ignore_index should be negative");
        }
        if self.assistant_only_loss && !self.role_boundaries {
            report.push(
                "assistant_only_loss",
                "assistant-only loss requires role boundaries",
            );
        }
        if !self.drop_oversized && self.skip_long_samples {
            report.push(
                "drop_oversized",
                "oversized policy is ambiguous: disable skip_long_samples or enable drop_oversized",
            );
        }

        report
    }
}
