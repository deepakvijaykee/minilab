"""Sample from a saved diffusion model checkpoint.

    python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000
    python scripts/sample_diffusion.py --tokenizer tokenizer.json --checkpoint checkpoints/diffusion/step_5000 --model sedd
"""

import argparse
from minilab.tokenizers import load_tokenizer
from minilab.models.mdlm import MDLM
from minilab.models.sedd import SEDD
from minilab.models.d3pm import D3PM
from minilab.diffusion import ForwardProcess
from minilab.generation import sample_d3pm, sample_diffusion, sample_sedd

MODELS = {"mdlm": MDLM, "sedd": SEDD, "d3pm": D3PM}
SAMPLERS = {"mdlm": sample_diffusion, "sedd": sample_sedd, "d3pm": sample_d3pm}

p = argparse.ArgumentParser()
p.add_argument("--tokenizer", required=True)
p.add_argument("--checkpoint", required=True)
p.add_argument("--model", default="mdlm", choices=list(MODELS))
p.add_argument("--num-samples", type=int, default=5)
p.add_argument("--seq-len", type=int, default=256)
p.add_argument("--num-steps", type=int, default=256)
p.add_argument("--temperature", type=float, default=1.0)
args = p.parse_args()

tok = load_tokenizer(args.tokenizer)

model = MODELS[args.model].load(args.checkpoint)
model.eval()
print(f"Loaded {args.checkpoint} ({model.num_parameters():,} params)\n")

# Load the exact forward process that was used at training time. Rebuilding from a
# CLI --schedule flag would silently sample the wrong chain for any non-default run.
fwd = ForwardProcess.load(f"{args.checkpoint}/forward_process.json")
print(f"Forward process: schedule={fwd.schedule} num_timesteps={fwd.num_timesteps}\n")
samples = SAMPLERS[args.model](model, fwd, batch_size=args.num_samples, seq_len=args.seq_len,
                               num_steps=args.num_steps, temperature=args.temperature)

for i in range(args.num_samples):
    s = [t for t in samples[i].tolist() if t < tok.vocab_size]
    print(f"--- Sample {i + 1} ---")
    print(tok.decode(s))
    print()
