"""
Predefined Game of Life patterns
"""
import numpy as np


# Still Lifes (period 1)
BLOCK = np.array([
    [1, 1],
    [1, 1]
], dtype=np.uint8)

BEEHIVE = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 1, 0]
], dtype=np.uint8)

BOAT = np.array([
    [1, 1, 0],
    [1, 0, 1],
    [0, 1, 0]
], dtype=np.uint8)

LOAF = np.array([
    [0, 1, 1, 0],
    [1, 0, 0, 1],
    [0, 1, 0, 1],
    [0, 0, 1, 0]
], dtype=np.uint8)


# Oscillators (period 2)
BLINKER = np.array([
    [1, 1, 1]
], dtype=np.uint8)

TOAD = np.array([
    [0, 1, 1, 1],
    [1, 1, 1, 0]
], dtype=np.uint8)

BEACON = np.array([
    [1, 1, 0, 0],
    [1, 1, 0, 0],
    [0, 0, 1, 1],
    [0, 0, 1, 1]
], dtype=np.uint8)


# Oscillators (period 15)
PULSAR = np.array([
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0]
], dtype=np.uint8)


# Spaceships (period 4)
GLIDER = np.array([
    [0, 1, 0],
    [0, 0, 1],
    [1, 1, 1]
], dtype=np.uint8)

LWSS = np.array([
    [0, 1, 0, 0, 1],
    [1, 0, 0, 0, 0],
    [1, 0, 0, 0, 1],
    [1, 1, 1, 1, 0]
], dtype=np.uint8)


# Glider Gun (period 30)
# Gosper's Glider Gun - emits one glider every 30 generations
GLIDER_GUN = np.array([
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1,1],
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [1,1,0,0,0,0,0,0,0,0,1,0,0,0,1,0,1,1,0,0,0,0,1,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
], dtype=np.uint8)


PATTERN_CATEGORIES = {
    'still_lifes': {
        'block': BLOCK,
        'beehive': BEEHIVE,
        'boat': BOAT,
        'loaf': LOAF
    },
    'oscillators_p2': {
        'blinker': BLINKER,
        'toad': TOAD,
        'beacon': BEACON
    },
    'oscillators_p15': {
        'pulsar': PULSAR
    },
    'spaceships': {
        'glider': GLIDER,
        'lwss': LWSS
    },
    'guns': {
        'glider_gun': GLIDER_GUN
    }
}


def get_pattern(name: str) -> np.ndarray:
    """
    Get a pattern by name.
    
    Args:
        name: Pattern name (e.g., 'glider', 'blinker')
        
    Returns:
        Pattern array
        
    Raises:
        ValueError: If pattern name not found
    """
    for category in PATTERN_CATEGORIES.values():
        if name in category:
            return category[name].copy()
    
    raise ValueError(f"Pattern '{name}' not found. Available patterns: "
                     f"{[p for cat in PATTERN_CATEGORIES.values() for p in cat.keys()]}")


def get_all_patterns():
    """
    Get all available patterns organized by category.
    
    Returns:
        Dictionary of categories and their patterns
    """
    return PATTERN_CATEGORIES
