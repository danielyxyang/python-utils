import ipywidgets as widgets


def build_widget_outputs(names, **args):
    return {name: widgets.Output(**args) for name in names}
