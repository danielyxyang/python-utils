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
