# TTS Runtime

This directory documents the local TTS runtime slot.

The default TTS service is the local Chatterbox-Turbo runtime configured in
`infra/compose.yml`. It exposes `/v1/audio/synthesize` and returns WAV audio.

Put the default voice reference clip at `voices/default.wav`, or set
`CHATTERBOX_TTS_AUDIO_PROMPT_PATH` to another mounted path. Chatterbox-Turbo
expects a reference clip longer than five seconds when no built-in conditionals
are available.

The previous NVIDIA Magpie/Riva TTS runtime remains available as the
`tts-riva` Compose service behind the `riva-tts` profile. Its model artifacts
are stored in the Docker named volume `nim_tts_cache`.
