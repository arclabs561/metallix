use serde::Deserialize;

const DEEPSEEK_REGISTRY: &str = include_str!("../../../config/artifacts/deepseek.json");

#[derive(Clone, Debug, Deserialize)]
pub(crate) struct DeepSeekArtifactSource {
    pub schema_version: u32,
    pub model_repo: String,
    pub model_revision: String,
    pub template_repo: String,
    pub template_revision: String,
    pub weight_bytes: u64,
    pub metadata_files: Vec<String>,
    pub weight_pattern: String,
}

pub(crate) fn deepseek() -> Result<DeepSeekArtifactSource, String> {
    let registry: DeepSeekArtifactSource = serde_json::from_str(DEEPSEEK_REGISTRY)
        .map_err(|error| format!("embedded model registry is invalid: {error}"))?;
    if registry.schema_version != 1 {
        return Err(format!(
            "unsupported embedded model registry schema {}",
            registry.schema_version
        ));
    }
    Ok(registry)
}
