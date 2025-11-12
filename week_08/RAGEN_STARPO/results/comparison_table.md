# WebShop vs WebArena Comparison

## Performance Comparison

| Metric | WebShop (RAGEN) | WebArena (GPT-4) | Difference |
|--------|----------------|------------------|------------|
| **Success Rate** | 67.0% | 14.4% | +52.6% |
| **Avg Steps** | 9.3 | 3.9 | +5.4 |
| **Total Tasks** | 100 | 812 | - |

## Environment Comparison

| Aspect | WebShop | WebArena |
|--------|---------|----------|
| **Environment** | Text-based shopping | Real websites (4 domains) |
| **Complexity** | Simple (search/click/buy) | Complex (full browser) |
| **Observations** | Text descriptions | Accessibility tree + DOM |
| **Actions** | 18 predefined commands | Browser actions (click/type/scroll) |
| **Agent** | RAGEN + A*PO (trained) | GPT-4 with Chain-of-Thought |

## Key Insights

1. **WebShop shows 52.6 percentage points higher success rate**
   - Simpler action space aids learning
   - Text-based observations easier to process
   - Trained specifically on task distribution

2. **Environment Complexity**
   - WebShop: 18 predefined actions, focused domain
   - WebArena: Unlimited browser actions, 4 different websites

3. **Training Advantage**
   - RAGEN trained on WebShop → high performance
   - GPT-4 zero-shot on WebArena → lower baseline
   - Shows importance of task-specific training


*Generated on 2025-11-09 22:29:50*
