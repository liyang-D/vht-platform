# Default ASR Domain

This is the placeholder artifact layout for the default ASR domain.

Current runtime behavior does not require any files here. The default route in
`services/orchestrator/config/asr_domains.json` points to the base Riva ASR NIM
service and works without a custom language model.

When a custom language model is trained, place generated artifacts in the
subdirectories here and update the domain config if the deployed endpoint or
model archive changes.
