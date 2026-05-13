# Backend Protocol

The .NET UI writes a JSON job spec and starts Python with:

```bash
python train_stub.py --job path/to/job_spec.json
```

Expected train backend behavior:

1. Read `job_spec.json`.
2. Load dataset.
3. Train tokenizer or load tokenizer.
4. Build GPT model.
5. Train.
6. Write logs as JSONL:

```json
{"step":0,"train_loss":4.2,"val_loss":4.3,"tokens_per_second":0,"message":"started"}
```

7. Save checkpoint manifest:

```json
{
  "formatVersion":"0.1",
  "modelWeightsPath":"model.pt",
  "tokenizerPath":"tokenizer.json",
  "step":1000,
  "trainLoss":1.8,
  "valLoss":2.0
}
```

Expected generation backend:

```bash
python generate_stub.py --checkpoint runs/default/checkpoint.json --prompt "The morning" --temperature 0.8 --top-k 40 --seed 42
```

Output JSON:

```json
{"text":"generated text here"}
```
