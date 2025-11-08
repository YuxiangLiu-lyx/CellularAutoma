# Scripts

## Generate Samples

Generate one sample per pattern for visualization and review.

### Usage

```bash
# Install dependencies first
pip install numpy matplotlib

# Run the script
cd /Users/liuyuxiang/Documents/USA/DeepLearningClass/project
python3 scripts/generate_samples.py
```

### Output

All files will be saved to `figures/samples/`:

- `{pattern}_initial.png` - Initial state
- `{pattern}_trajectory.png` - Evolution over 8 frames
- `{pattern}_animation.gif` - Animated GIF (50 steps)
- `all_patterns_overview.png` - Grid view of all patterns

### Patterns Generated

**Still Lifes**: block, beehive, boat, loaf  
**Oscillators (period 2)**: blinker, toad, beacon  
**Oscillators (period 15)**: pulsar  
**Spaceships**: glider, lwss

Total: 10 patterns

