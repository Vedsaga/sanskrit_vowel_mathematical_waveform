"""
Common configuration, imports, and fallbacks for formant analysis scripts.

This module centralizes:
- Optional import handling (seaborn, tqdm)
- Font configuration for Devanagari support
- Common constants
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# ============================================================================
# Optional Imports with Fallbacks
# ============================================================================

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    sns = None
    HAS_SEABORN = False

try:
    from tqdm import tqdm as _tqdm
    HAS_TQDM = True
    def tqdm(iterable, **kwargs):
        """Wrapper for tqdm that works whether tqdm is installed or not."""
        return _tqdm(iterable, **kwargs)
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, **kwargs):
        """Fallback when tqdm is not installed - just returns the iterable."""
        return iterable

# ============================================================================
# Font Configuration
# ============================================================================

DEVANAGARI_FONT_PATH = '/usr/share/fonts/noto/NotoSansDevanagari-Regular.ttf'

_matplotlib_configured = False

def configure_matplotlib():
    """
    Configure matplotlib for Devanagari font support and consistent styling.
    
    This function is idempotent - calling it multiple times has no effect
    after the first call.
    
    Usage:
        from common.config import configure_matplotlib
        configure_matplotlib()  # Call once at script start
    """
    global _matplotlib_configured
    
    if _matplotlib_configured:
        return
    
    if os.path.exists(DEVANAGARI_FONT_PATH):
        fm.fontManager.addfont(DEVANAGARI_FONT_PATH)
        plt.rcParams['font.family'] = ['Noto Sans Devanagari', 'DejaVu Sans', 'sans-serif']
    
    _matplotlib_configured = True


# ============================================================================
# Analysis Constants
# ============================================================================

# Default parameters for formant extraction
DEFAULT_TIME_STEP = 0.01  # seconds
DEFAULT_MAX_FORMANTS = 5
DEFAULT_MAX_FORMANT_FREQ = 5500.0  # Hz
DEFAULT_WINDOW_LENGTH = 0.025  # seconds

# Default parameters for weighting
DEFAULT_NOISE_FLOOR = 50.0  # dB
DEFAULT_SOFT_GATE_THRESHOLD = 30.0  # dB
DEFAULT_INTENSITY_CLIP_DB = 30.0  # dB (prevents burst dominance)
DEFAULT_INTENSITY_EXPONENT = 2.0
DEFAULT_STABILITY_SMOOTHING = 0.1
