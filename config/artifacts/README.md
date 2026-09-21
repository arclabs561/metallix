# Artifact registry

Each supported downloadable model has one small manifest in this directory.
Manifests pin public repositories and revisions, describe the metadata needed
for bounded local inspection, and name the weight pattern. They contain no
credentials or model payloads. `weight_bytes` is a preflight floor derived from
the pinned Hub revision; `mx fetch` refuses a full download when the target
volume cannot hold it.

`mx fetch` consumes these manifests and stores ordinary Hugging Face-compatible
files in the destination directory supplied by the user.
