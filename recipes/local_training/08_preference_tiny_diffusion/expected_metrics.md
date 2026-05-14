# Expected signals

- Diffusion DPO and VRPO both print `Trainable: <path> (mdlm, <N> params,
  schedule=<schedule>)` followed by `Frozen reference: <path>`. The
  schedule string comes from `forward_process.json`; if it reads
  `schedule=None` something is wrong with the checkpoint copy.
- The dataset line is `hh-rlhf: <N> diffusion preference pairs` (or
  `ultrafeedback: ...`). The count reflects pairs left after filtering for
  sequences that fit `--seq-len`.
- VRPO is noticeably slower per step (it averages multiple diffusion
  estimates per pair). Expect step time roughly proportional to
  `--vrpo-num-samples`.
- The end-of-run block prints `--- After Diffusion DPO ---` (or `VRPO`)
  with three response generations through reverse diffusion.

Quality at this scale is bounded by the diffusion SFT base. The preference
step shifts the response distribution but cannot manufacture coherence
that the base does not already have.
