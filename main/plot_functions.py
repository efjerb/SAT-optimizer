import cycler
import matplotlib.pyplot as plt

# Define the golden ratio constant
GOLDEN_RATIO = (5 ** 0.5 - 1) / 2  # ≈ 0.618
CM_TO_INCH = 1 / 2.54  # Conversion factor: 1 cm = 0.3937 inches

# Define standard paper sizes in cm
A4_WIDTH = 21.0  # A4 width in cm
COLUMN_WIDTHS = {"single": 8.6, "double": 17.9, "full": A4_WIDTH, "thesis": 12.5}  # Common column widths in cm

# Define a color cycle for consistency
COLOR_CYCLE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

def get_figure_size(width="double", height=None, golden=False):
    """
    Compute figure size in inches for scientific papers.

    Parameters:
    - width: "single" (8.8 cm), "double" (18.0 cm), "full" (A4 width, 21.0 cm), or "thesis" (12.5 cm)
    - height: Custom height in cm (default: golden ratio scaling)
    - golden: Whether to apply golden ratio scaling to height (default: True)

    Returns:
    - Tuple (width, height) in inches for figsize in Matplotlib.
    """
    w_cm = COLUMN_WIDTHS.get(width, COLUMN_WIDTHS["single"])  # Get width in cm
    h_cm = height if height else (w_cm * GOLDEN_RATIO if golden else 6)  # Default to 3:4 ratio if not golden

    return (w_cm * CM_TO_INCH, h_cm * CM_TO_INCH)  # Convert cm to inches

def set_plot_style(width="single", height=None, golden=True, fontsize=8, color_cycle="tab10"):
    """
    Set global matplotlib rcParams for consistent figure sizes and styling.

    Parameters:
    - width: "single", "double", "full" (A4-based widths), or "thesis" (12.5 cm)
    - height: Custom height in cm (default: golden ratio scaling)
    - golden: Whether to apply golden ratio scaling to height (default: True)
    - fontsize: Base font size for labels and ticks (default: 10)
    - color_cycle: Built-in Matplotlib color cycle (e.g., "tab10", "Set1", "viridis", etc.)
    """
    figsize = get_figure_size(width, height, golden)  # Compute figure size


    # Set color cycle based on built-in cycles
    if color_cycle == "tab10":
        cycle = cycler.cycler(color=plt.cm.tab10.colors)  # Default Tableau 10 colors
    elif color_cycle == "Set1":
        cycle = cycler.cycler(color=plt.cm.Set1.colors)
    elif color_cycle == "viridis":
        cycle = cycler.cycler(color=plt.cm.viridis.colors)
    elif color_cycle == "pastel":
        cycle = cycler.cycler(color=plt.cm.Paired.colors)  # A pastel-like set
    else:
        cycle = cycler.cycler(color=plt.cm.tab10.colors)  # Default to 'tab10'


    
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "figure.figsize": figsize,
        "font.size": fontsize,
        "axes.labelsize": fontsize + 2,
        "axes.titlesize": fontsize + 4,
        "xtick.labelsize": fontsize,
        "ytick.labelsize": fontsize,
        "legend.fontsize": fontsize,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "axes.prop_cycle": cycle,  # Set color cycle
        "savefig.dpi": 300,  # High resolution for publications
        "savefig.bbox": "tight",
        "pdf.fonttype": 42,  # Ensures vectorized text in PDF
    })

if __name__ == "__main__":
    

    # Example usage of global settings
    set_plot_style("double", fontsize=12)

    # Example of overriding figure size for a single plot
    fig, ax = plt.subplots(figsize=get_figure_size("single", height=5))  # Custom height
    ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_xlabel("X-axis")
    ax.set_ylabel("Y-axis")

    plt.savefig("custom_size_plot.pdf")
    plt.show()
