# Expected signals

- For reference-using algorithms (DPO, IPO, KTO) the script prints both
  `Trainable: <path>` and `Frozen reference: <path>` lines before training.
  If you do not see the frozen line, the algorithm is reference-free
  (SimPO, ORPO, CPO, RePO) and `estimate_vram.py` should report lower peak
  memory accordingly.
- Preference loss should stay finite. DPO loss in particular swings around
  much more than SFT loss, so step-to-step jitter is normal; what you do
  not want to see is monotonically increasing loss past warmup.
- The end-of-run samples answer three different fixed prompts than SFT
  (`What makes a good friend?`, `How do I learn to cook?`, `Tell me about
  dogs.`). Responses should be coherent at sentence level but are typically
  short and shallow on the laptop-scale base.

Tiny preference runs validate the loss path and the reference-model glue.
They are not a preference benchmark.
