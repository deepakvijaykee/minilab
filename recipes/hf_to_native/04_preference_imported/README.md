# 04 Preference Imported Model

Run native Minilab preference optimization on the imported-model SFT checkpoint.
The default is `simpo` because it avoids a second reference model.

```bash
bash recipes/hf_to_native/04_preference_imported/run.sh
```

Use DPO when you want a frozen-reference comparison:

```bash
ALGORITHM=dpo bash recipes/hf_to_native/04_preference_imported/run.sh
```
