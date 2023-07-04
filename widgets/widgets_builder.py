__all__ = ["build_widget_outputs"]

import ipywidgets as widgets


def build_widget_outputs(names):
    return {name: widgets.Output() for name in names}
