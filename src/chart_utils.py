"""Shared chart saving utility for HTML and PNG export."""

from pathlib import Path

OUTPUT_DIR = Path("outputs/figures")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def save_chart(fig, name, width=None, height=None):
    """
    Save a Plotly figure as HTML (always) and PNG (if kaleido is installed).

    Args:
        fig: Plotly figure object
        name: filename without extension (e.g. 'elasticity_heatmap')
        width/height: optional overrides for PNG export
    """
    html_path = OUTPUT_DIR / f"{name}.html"
    png_path = OUTPUT_DIR / f"{name}.png"

    fig.write_html(str(html_path))

    try:
        kwargs = {"scale": 2}
        if width:
            kwargs["width"] = width
        if height:
            kwargs["height"] = height
        fig.write_image(str(png_path), **kwargs)
        print(f"  Saved: {png_path}")
    except Exception:
        print(f"  Saved: {html_path} (PNG skipped — install kaleido)")

    return html_path, png_path
