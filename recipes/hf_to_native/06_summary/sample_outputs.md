# Sample Outputs

```text
run                   status     trainer      device  duration  peak_vram  steps  tok/s  params
--------------------  ---------  -----------  ------  --------  ---------  -----  -----  ------
smollm2-135m-sft      completed  SFTTrainer   cuda    12m03s    3.20G      100    5.4k/s 134.5M
smollm2-135m-simpo    completed  SimPOTrainer cuda    8m14s     3.80G      50     3.1k/s 134.5M
smollm2-135m-grpo     completed  GRPOTrainer  cuda    20m10s    5.60G      25     1.0k/s 134.5M
```
