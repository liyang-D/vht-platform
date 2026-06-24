# ASR Runtime

This directory documents the local ASR runtime slot.

The active ASR service is the NVIDIA Speech NIM container configured in
`infra/compose.yml`, defaulting to `parakeet-1-1b-ctc-en-us`.
Model artifacts are stored in the Docker named volume `nim_asr_cache`.

ASR domain routing lives in
`services/orchestrator/config/asr_domains.json`. The checked-in `default`
domain is only a route/config template; custom language model artifacts can
stay unset until a domain LM has been trained and deployed.

Domain-specific training and deployment artifacts should be placed under
`services/runtime/asr/domains/<domain>/`. The `default` domain includes an empty
placeholder layout for corpus, normalized text, KenLM artifacts, decoder files,
evaluation reports, and Riva deployment output.

The default selector uses `NIM_TAGS_SELECTOR=name=parakeet-1-1b-ctc-en-us,mode=all`
so the ASR runtime can expose both offline and streaming modes. The current
upload-based app path sends complete audio files to the orchestrator and
defaults to offline transcription; a true streaming ASR client should use the
gRPC/WebSocket streaming transport rather than the multipart upload endpoint.
