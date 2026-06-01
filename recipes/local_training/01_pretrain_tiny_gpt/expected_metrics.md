# Expected signals

The estimator prints a rough VRAM figure before training begins. When
the GPU has less headroom than the reported peak, dropping
`BATCH_SIZE` or `SEQ_LEN` before the run is far better than pushing
through and triggering an out-of-memory during the first forward
pass.

Initial loss on a 4k-vocab model with uniform predictions is
`log(4096)`, which is roughly 8.3 nats. The default thousand-step run
usually lands in the 5 to 6 range, which means the easy entropy is
gone and the model is now climbing the long tail of bigram and
short-range context structure. If the loss is still above 6 at the end
of the run, the most likely cause is a vocabulary mismatch with the
loaded `tokenizer.json`, which leaves the model unable to attribute
mass to the actual training tokens.

Sample quality at a thousand steps looks like fluent tokens without
coherent narrative. The model has captured the unigram and short-range
bigram distribution but not the longer-range story templates. Story
coherence on TinyStories is roughly a function of parameters times
steps, and it appears around `gpt-25m` trained for three thousand steps.
Below that threshold the samples will read as text-shaped but
narratively flat.

The trainer logs every 100 steps and evaluates every 500. TinyStories
ships with a held-out split, so `Eval perplexity` prints at the end of
the run. OpenWebText is the one dataset that skips evaluation, because
it is streamed and has no fixed split to evaluate against. The
`run_metrics.json` file is written to the final `step_<N>` directory
and is also copied to the recipe save root, which is the place to look
for actual measured memory and timing numbers.
