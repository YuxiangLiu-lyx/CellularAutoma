"""
Analyze the label distribution in Game of Life 3x3 patterns.
Why are most patterns resulting in death?
"""
import numpy as np
from itertools import product


def game_of_life_rule(pattern):
    """Apply Game of Life rule to center cell."""
    center = pattern[1, 1]
    alive_neighbors = np.sum(pattern) - center
    
    if center == 1:  # Cell is alive
        return 1 if alive_neighbors in [2, 3] else 0
    else:  # Cell is dead
        return 1 if alive_neighbors == 3 else 0


def analyze_all_patterns():
    """Analyze all 512 patterns."""
    print("="*70)
    print("GAME OF LIFE LABEL DISTRIBUTION ANALYSIS")
    print("="*70)
    
    # Count by outcome
    outcomes = {
        'total': 0,
        'alive_next': 0,
        'dead_next': 0
    }
    
    # Count by current center state and outcome
    center_alive_outcomes = {'survive': 0, 'die': 0}
    center_dead_outcomes = {'born': 0, 'stay_dead': 0}
    
    # Count by neighbor count
    neighbor_stats = {}
    
    for bits in product([0, 1], repeat=9):
        pattern = np.array(bits).reshape(3, 3)
        center = pattern[1, 1]
        alive_neighbors = int(np.sum(pattern) - center)
        next_state = game_of_life_rule(pattern)
        
        outcomes['total'] += 1
        if next_state == 1:
            outcomes['alive_next'] += 1
        else:
            outcomes['dead_next'] += 1
        
        # Track by current state
        if center == 1:
            if next_state == 1:
                center_alive_outcomes['survive'] += 1
            else:
                center_alive_outcomes['die'] += 1
        else:
            if next_state == 1:
                center_dead_outcomes['born'] += 1
            else:
                center_dead_outcomes['stay_dead'] += 1
        
        # Track by neighbor count
        key = f"{int(center)}_{alive_neighbors}"
        if key not in neighbor_stats:
            neighbor_stats[key] = {'total': 0, 'alive': 0, 'dead': 0}
        neighbor_stats[key]['total'] += 1
        if next_state == 1:
            neighbor_stats[key]['alive'] += 1
        else:
            neighbor_stats[key]['dead'] += 1
    
    print(f"\nTotal patterns: {outcomes['total']}")
    print(f"Patterns resulting in ALIVE: {outcomes['alive_next']} ({outcomes['alive_next']/outcomes['total']*100:.1f}%)")
    print(f"Patterns resulting in DEAD:  {outcomes['dead_next']} ({outcomes['dead_next']/outcomes['total']*100:.1f}%)")
    
    print("\n" + "="*70)
    print("BASELINE STRATEGIES")
    print("="*70)
    print(f"Always predict DEAD:  {outcomes['dead_next']}/512 = {outcomes['dead_next']/512*100:.2f}% accuracy")
    print(f"Always predict ALIVE: {outcomes['alive_next']}/512 = {outcomes['alive_next']/512*100:.2f}% accuracy")
    print(f"Random guess (50/50): ~50% accuracy")
    
    print("\n" + "="*70)
    print("BREAKDOWN BY CURRENT CENTER STATE")
    print("="*70)
    
    total_center_alive = center_alive_outcomes['survive'] + center_alive_outcomes['die']
    total_center_dead = center_dead_outcomes['born'] + center_dead_outcomes['stay_dead']
    
    print(f"\nWhen center is ALIVE (256 patterns):")
    print(f"  Survive: {center_alive_outcomes['survive']} ({center_alive_outcomes['survive']/total_center_alive*100:.1f}%)")
    print(f"  Die:     {center_alive_outcomes['die']} ({center_alive_outcomes['die']/total_center_alive*100:.1f}%)")
    
    print(f"\nWhen center is DEAD (256 patterns):")
    print(f"  Born:       {center_dead_outcomes['born']} ({center_dead_outcomes['born']/total_center_dead*100:.1f}%)")
    print(f"  Stay dead:  {center_dead_outcomes['stay_dead']} ({center_dead_outcomes['stay_dead']/total_center_dead*100:.1f}%)")
    
    print("\n" + "="*70)
    print("DETAILED: NEIGHBOR COUNT ANALYSIS")
    print("="*70)
    print(f"{'Center':<8} {'Neighbors':<10} {'Total':<8} {'→Alive':<10} {'→Dead':<10} {'Rule':<30}")
    print("-"*70)
    
    for center in [0, 1]:
        for neighbors in range(9):
            key = f"{center}_{neighbors}"
            if key in neighbor_stats:
                stats = neighbor_stats[key]
                
                # Determine rule
                if center == 1:
                    if neighbors in [2, 3]:
                        rule = "Survive"
                    else:
                        rule = f"Die ({'underpop' if neighbors < 2 else 'overpop'})"
                else:
                    if neighbors == 3:
                        rule = "Born"
                    else:
                        rule = "Stay dead"
                
                print(f"{center:<8} {neighbors:<10} {stats['total']:<8} "
                      f"{stats['alive']:<10} {stats['dead']:<10} {rule:<30}")
    
    print("\n" + "="*70)
    print("KEY INSIGHTS")
    print("="*70)
    print("\n1. Why 'always predict dead' works so well (72.66%):")
    print("   - Only 2 out of 9 neighbor counts allow survival (2, 3)")
    print("   - Only 1 out of 9 neighbor counts allows birth (3)")
    print("   - Most random configurations don't satisfy these strict conditions")
    
    print("\n2. Game of Life is biased toward DEATH:")
    print("   - Alive cells: Only 22.2% (2/9) of neighbor counts allow survival")
    print("   - Dead cells: Only 11.1% (1/9) of neighbor counts allow birth")
    
    print("\n3. This is NOT a balanced classification problem!")
    print(f"   - Class imbalance: {outcomes['dead_next']/outcomes['alive_next']:.1f}:1 (dead:alive)")
    print("   - A model that learns anything about the rules")
    print("     will do MUCH better than 72.66%!")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    analyze_all_patterns()
