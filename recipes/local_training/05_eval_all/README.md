# 05 Eval All

Evaluate checkpoints produced by the laptop GPU track.

```bash
bash recipes/local_training/05_eval_all/run.sh
```

This recipe skips missing checkpoints, so it is safe to run after any partial
stage of the track.
