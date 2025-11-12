# WebArena Evaluation - Quick Start Guide

Choose your approach based on time and goals:

---

## Option 1: Simple Baseline (5 seconds) ⚡

**Best for**: Quick presentation results

```bash
# Get GPT-4 baseline
python3 eval/evaluate_webarena_baseline.py --mode quick

# Generate comparison
python3 compare_results.py

# View results
cat results/comparison_table.md
```

**Output**: WebShop (67%) vs WebArena GPT-4 (14.4%)

---

## Option 2: Adapted Transfer Learning (5 minutes) 🎯

**Best for**: Show your model generalizing to new domain

```bash
# 1. Create sample shopping tasks
python3 scripts/filter_shopping_tasks.py

# 2. Test adapter (no PyTorch needed)
python3 test_adapter.py

# 3. Run adapted evaluation (needs PyTorch)
# Install: pip3 install torch
python3 eval/evaluate_webarena_shopping.py --num_tasks 10 --verbose

# 4. Generate three-way comparison
python3 compare_adapted_results.py

# 5. View results
cat results/three_way_comparison.md
```

**Output**: WebShop (67%) → WebArena Shopping (adapted) (25%?) → WebArena Full (14.4%)

---

## What Each File Does

### Core Files Created

1. **`envs/webarena_adapter.py`** - Translates WebArena ↔ WebShop formats
   - Accessibility tree → text observations
   - WebShop actions → browser commands
   - Tested and working ✅

2. **`eval/evaluate_webarena_shopping.py`** - Runs your trained model on WebArena
   - Loads your WebShop model
   - Uses adapter to bridge format gap
   - Mock mode for testing without Docker

3. **`scripts/filter_shopping_tasks.py`** - Extracts shopping tasks
   - Creates 10 sample tasks for testing
   - Can filter real WebArena tasks if available

4. **`compare_adapted_results.py`** - Three-way comparison table
   - WebShop baseline
   - Adapted transfer
   - GPT-4 baseline

5. **`test_adapter.py`** - Quick adapter test
   - No PyTorch required
   - Verifies adapter works
   - Run first to check setup

### Documentation

- **`COMPLETE_WEBARENA_GUIDE.md`** - Full detailed guide
- **`WEBARENA_SETUP_INSTRUCTIONS.txt`** - Simple baseline approach
- **`QUICKSTART.md`** - This file

---

## Verification Checklist

Run these to verify everything works:

```bash
# ✓ Check adapter (no dependencies)
python3 test_adapter.py

# ✓ Check simple baseline
python3 eval/evaluate_webarena_baseline.py --mode quick
python3 compare_results.py

# ✓ Check sample tasks created
ls -la data/webarena_shopping_tasks.json
```

All green ✓? You're ready!

---

## Expected Results

### Simple Baseline

```
| Benchmark | Success Rate |
|-----------|--------------|
| WebShop   | 67.0%        |
| WebArena  | 14.4%        |
```

### Adapted Transfer

```
| Benchmark                   | Success Rate |
|-----------------------------|--------------|
| WebShop (trained)           | 67.0%        |
| WebArena Shopping (adapted) | 15-35%       |
| WebArena Full (GPT-4)       | 14.4%        |
```

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'torch'"

```bash
pip3 install torch
```

Or run without PyTorch:
```bash
# Just test adapter
python3 test_adapter.py

# Use simple baseline instead
python3 eval/evaluate_webarena_baseline.py --mode quick
```

### "Model not found"

```bash
# Check model exists
ls -la models/

# If missing, train first
python3 train_official_minimal.py --epochs 1
```

### "Tasks file not found"

```bash
python3 scripts/filter_shopping_tasks.py
```

---

## For Presentation

### Slide 1: RAGEN Performance
- Trained on WebShop
- 67% success rate
- Task-specific RL approach

### Slide 2: Transfer Learning (if you ran Option 2)
- Adapted to WebArena shopping
- Performance drop shows generalization challenge
- Adapter bridges format gap

### Slide 3: Baseline Comparison
- WebShop: Simple (67%)
- WebArena: Complex (14.4% GPT-4)
- Environment complexity matters

### Slide 4: Analysis
- Task-specific training wins on simple tasks
- Transfer learning reveals limitations
- Future work: End-to-end on complex observations

---

## Files Generated

After running evaluations:

```
results/
├── webarena_baseline.json              # GPT-4 baseline
├── webarena_adapted_results.json       # Your adapted model
├── comparison_table.csv                # Simple comparison
├── comparison_table.md
├── three_way_comparison.csv            # Full comparison
├── three_way_comparison.md
└── final_failure_analysis.md           # Updated analysis
```

---

## Time Required

| Approach | Setup | Run | Total |
|----------|-------|-----|-------|
| Simple Baseline | 0 min | 5 sec | < 1 min |
| Adapted (mock) | 2 min | 3 min | 5 min |
| Adapted (live Docker) | 30 min | 2 hours | 2.5 hours |

**Recommendation**: Start with Simple Baseline, then try Adapted with mock if you have time!

---

## Success Criteria

✅ Comparison tables generated
✅ Analysis explains performance differences
✅ Presentation slides ready
✅ Understanding of environment complexity impact

You don't need perfect results - good analysis of why performance differs is the key!

---

## Need Help?

1. Read `COMPLETE_WEBARENA_GUIDE.md` for detailed instructions
2. Run `python3 test_adapter.py` to verify setup
3. Check `results/final_failure_analysis.md` for analysis examples

Good luck! 🚀
