import os
import matplotlib.pyplot as plt
from abc import ABC, abstractmethod

from modern_nlp.core.utils import get_logger

logger = get_logger(__name__)

class BaseVisualizer(ABC):
    """
    Abstract Base Class for all visualization generators in modern_nlp.
    Provides standard rendering configs and multi-format save mechanisms.
    """
    def __init__(self, output_dir: str) -> None:
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self._configure_matplotlib()
        
    def _configure_matplotlib(self) -> None:
        """Applies generic styling to matplotlib."""
        plt.rcParams['figure.figsize'] = (10, 8)
        plt.rcParams['font.size'] = 12
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['ytick.labelsize'] = 10
        plt.rcParams['legend.fontsize'] = 10
        
    def save_figure(self, filename_prefix: str) -> None:
        """
        Saves the current matplotlib figure as PNG, SVG, and PDF.
        """
        formats = ["png", "svg", "pdf"]
        for fmt in formats:
            path = os.path.join(self.output_dir, f"{filename_prefix}.{fmt}")
            try:
                plt.savefig(path, format=fmt, dpi=300, bbox_inches="tight")
                logger.debug(f"Saved figure to {path}")
            except Exception as e:
                logger.error(f"Failed to save {fmt} figure {path}: {e}")
        plt.close()

    @abstractmethod
    def generate_all(self, *args, **kwargs) -> None:
        """
        Main entrypoint for generating the suite of visualizations.
        """
        pass
