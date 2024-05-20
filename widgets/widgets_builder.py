__all__ = ["build_widget_outputs"]

import ipywidgets as widgets


def build_widget_outputs(names, **args):
    return {name: widgets.Output(**args) for name in names}
