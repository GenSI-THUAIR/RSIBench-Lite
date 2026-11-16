# Trusted evaluation

`inference/runtime_infer_verifier.py` manages Modal smoke, calibration, and formal scoring.
`inference/infer_verifier.py` preserves the correctness and throughput protocol;
`delivery/` handles OCI layers, clean baselines, service startup, and input identity.
`grading/` applies the GRM rubric using explicitly supplied judge credentials.

`resources.py` publishes an immutable per-process bundle containing only the delivery,
inference, text, vision, and video helpers needed by the lite set. Training, speech,
action, and realtime-video evaluators are excluded. Input identities, workload sizes,
quality thresholds, and scoring semantics remain frozen.
