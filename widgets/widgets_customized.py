import os

import ipywidgets as widgets
from traitlets import Any


class CheckboxList(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected checkboxes")

    def __init__(self, options, value=[], colors={}, description=None, **kwargs):
        # create checkbox widgets
        checkboxes = {}
        checkboxes_list = []
        for option in options:
            option_path = option.split(".")
            parent_all = ".".join(option_path[:-1] + ["*"])
            # create checkbox
            checkbox = widgets.Checkbox(
                value=option in value or parent_all in value,
                indent=False,
                layout=dict(width="auto", margin="0px"),
            )
            checkbox_label = widgets.Label(
                value=option_path[-1],
                layout=dict(margin="0px"),
                style=dict(text_color=colors.get(option, colors.get(parent_all, None))),
            )
            checkbox_widget = widgets.HBox(
                [checkbox, checkbox_label],
                layout=dict(overflow="visible", padding="0 10px 0 10px", margin=f"0 0 0 {len(option_path[:-1])*30}px"),
            )
            # register handler to update widget value
            checkbox.observe(self.__update_value, names="value")

            checkboxes[option] = checkbox
            checkboxes_list.append(checkbox_widget)

        # create label and container widgets
        children = []
        if description is not None:
            children.append(widgets.Label(value=description))
        children.append(widgets.VBox(checkboxes_list, **kwargs))
        super().__init__(children=children)

        self.checkboxes = checkboxes
        self.__update_value()

    # PRIVATE METHODS

    def __update_value(self, *args):
        value = {}
        for option, checkbox in self.checkboxes.items():
            option_path = option.split(".")
            if len(option_path) == 1:
                value[option] = checkbox.value
            else:
                parent = ".".join(option_path[:-1])
                value[option] = value[parent] and checkbox.value
                checkbox.disabled = not value[parent]
        self.value = value


class FileExplorerWidget(widgets.VBox, widgets.widget_description.DescriptionWidget, widgets.ValueWidget):
    value = Any(help="Selected path")

    def __init__(self, path_names, default=None):
        # create widgets
        path_widgets = []
        path_handlers = []
        for i, path_name in enumerate(path_names):
            # create widget
            if i == 0:
                path_widget = widgets.Text(description=path_name)
            else:
                path_widget = widgets.Dropdown(description=path_name)
            # register event handler
            path_handler = self.__build_on_change_handler(i)
            path_widget.observe(path_handler, names="value")

            path_widgets.append(path_widget)
            path_handlers.append(path_handler)

        # initialize widget
        super().__init__(children=path_widgets)
        self.path_widgets = path_widgets
        self.path_handlers = path_handlers

        # set default value
        if default is not None:
            path_widgets[0].value = default
        # update widget value
        self.__update_value()

    # PRIVATE METHODS

    def __get_path(self, i=None):
        path_widgets = self.path_widgets if i is None else self.path_widgets[:i+1]
        path_components = [path_widget.value for path_widget in path_widgets]
        if all(path_component is not None for path_component in path_components):
            return os.path.join(*path_components)
        else:
            return None

    def __build_on_change_handler(self, i):
        def on_change(*args):
            path = self.__get_path(i)

            if i < len(self.path_widgets) - 1:
                # update next path widget
                next_dropdown = self.path_widgets[i+1]
                next_handler = self.path_handlers[i+1]

                next_dropdown.unobserve(next_handler, names="value")
                if path is not None and os.path.isdir(path):
                    previous_value = next_dropdown.value
                    if i < len(self.path_widgets) - 2:
                        entries = sorted(e.name for e in os.scandir(path) if e.is_dir())
                    else:
                        entries = sorted(e.name for e in os.scandir(path) if e.is_file())
                    next_dropdown.options = entries
                    next_dropdown.value = previous_value if previous_value in entries else None
                else:
                    next_dropdown.options = []
                next_handler()
                next_dropdown.observe(next_handler, names="value")
            else:
                # update widget value
                self.__update_value()

        return on_change

    def __update_value(self, *args):
        self.value = self.__get_path()
