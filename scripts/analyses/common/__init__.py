"""
Common utilities for formant analysis scripts.

This package provides shared functions to reduce code duplication across
the formant-based-invariant and temporal-hypotheses analysis scripts.

Usage:
    from common import (
        configure_matplotlib,
        extract_formants_with_weights,
        compute_joint_weights,
        apply_dark_theme,
        COLORS,
        tqdm, HAS_SEABORN, HAS_TQDM
    )
"""

from .config import (
    configure_matplotlib,
    HAS_SEABORN,
    HAS_TQDM,
    tqdm,
    DEVANAGARI_FONT_PATH
)

from .weighting import (
    compute_joint_weights,
    compute_weight_entropy
)

from .formant_extraction import (
    extract_formants_with_weights,
    extract_raw_formant_trajectory
)

from .plotting import (
    apply_dark_theme,
    create_styled_figure,
    style_legend,
    COLORS
)

__all__ = [
    # Config
    'configure_matplotlib',
    'HAS_SEABORN',
    'HAS_TQDM', 
    'tqdm',
    'DEVANAGARI_FONT_PATH',
    # Weighting
    'compute_joint_weights',
    'compute_weight_entropy',
    # Formant extraction
    'extract_formants_with_weights',
    'extract_raw_formant_trajectory',
    # Plotting
    'apply_dark_theme',
    'create_styled_figure',
    'style_legend',
    'COLORS',
]
