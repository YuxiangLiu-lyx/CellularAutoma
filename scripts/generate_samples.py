"""
Generate one sample per pattern for visualization and review
"""
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from src.utils.game_of_life import GameOfLife, place_pattern
from src.utils.patterns import PATTERN_CATEGORIES, get_pattern
from src.utils.visualization import (
    visualize_state, 
    visualize_trajectory, 
    create_animation,
    visualize_pattern_grid
)


def main():
    """Generate and visualize one sample per pattern."""
    
    # Use absolute path to project root
    project_root = Path(__file__).parent.parent
    output_dir = project_root / "figures" / "samples"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Generating samples for each pattern...")
    print("=" * 60)
    
    all_initial_states = {}
    
    for category_name, patterns in PATTERN_CATEGORIES.items():
        print(f"\nCategory: {category_name}")
        print("-" * 60)
        
        for pattern_name, pattern in patterns.items():
            print(f"  Processing {pattern_name}...")
            
            # Adjust grid size and steps based on pattern type
            if pattern_name == 'glider_gun':
                grid_size = (50, 80)
                num_steps = 150
            elif pattern_name == 'pulsar':
                grid_size = (30, 30)
                num_steps = 50
            else:
                grid_size = (20, 20)
                num_steps = 50
            
            gol = GameOfLife(grid_size)
            initial_state = place_pattern(grid_size, pattern, position=None)
            all_initial_states[pattern_name] = initial_state
            
            trajectory = gol.simulate(initial_state, num_steps)
            
            # Save initial state
            visualize_state(
                initial_state,
                title=f"{pattern_name.upper()} (t=0)",
                save_path=output_dir / f"{pattern_name}_initial.png",
                figsize=(10, 10) if pattern_name == 'glider_gun' else (8, 8),
                show_grid=True
            )
            
            # Save trajectory visualization
            visualize_trajectory(
                trajectory,
                pattern_name=pattern_name.upper(),
                save_path=output_dir / f"{pattern_name}_trajectory.png",
                num_frames_to_show=8,
                figsize=(18, 5) if pattern_name == 'glider_gun' else (16, 4),
                show_grid=True
            )
            
            # Save animation
            create_animation(
                trajectory,
                pattern_name=pattern_name.upper(),
                save_path=output_dir / f"{pattern_name}_animation.gif",
                fps=10,
                figsize=(10, 8) if pattern_name == 'glider_gun' else (8, 8),
                show_grid=True
            )
            
            print(f"    Generated: initial, trajectory, animation")
            if pattern_name == 'glider_gun':
                print(f"    Grid: {grid_size}, Steps: {num_steps} (larger to show glider emission)")
    
    # Create overview grid of all patterns
    print("\n" + "=" * 60)
    print("Creating pattern overview grid...")
    visualize_pattern_grid(
        all_initial_states,
        save_path=output_dir / "all_patterns_overview.png",
        figsize=(18, 12),
        show_grid=True
    )
    
    print("\n" + "=" * 60)
    print(f"All samples saved to: {output_dir.absolute()}")
    print("=" * 60)
    print("\nGenerated files:")
    print("  - *_initial.png: Initial state of each pattern")
    print("  - *_trajectory.png: Evolution over 8 timesteps")
    print("  - *_animation.gif: Animated evolution")
    print("  - all_patterns_overview.png: Grid of all patterns")


if __name__ == "__main__":
    main()
