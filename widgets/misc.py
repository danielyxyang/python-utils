import logging

import pandas as pd
from IPython.display import HTML, display

logger = logging.getLogger(__name__)

def display_table(table, sort=None, index=None, columns=None, mode="html", html_align=None, html_height=None):
    df = pd.DataFrame(table)
    if sort is not None:
        df = df.sort_values(sort)
    if index is not None:
        df = df.set_index(index)
    if columns is not None:
        df = df.reindex(columns, axis="columns")
    if mode == "html":
        df_style = df.style
    if html_align is not None:
        if mode == "html":
            df_style = df_style.set_properties(**{"text-align": html_align}).set_table_styles([dict(selector="th", props=f"text-align: {html_align}")])
        else:
            logger.warning("Aligning columns is only supported in mode \"html\".")
    if html_height is not None:
        if mode != "html":
            logger.warning("Limiting height of table is only supported in mode \"html\".")
    with pd.option_context(
        "display.expand_frame_repr", False,
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
    ):
        if mode == "html":
            if html_height is not None:
                display(HTML(f"<div style='height: {html_height}; overflow: auto;'>{df_style.to_html()}</div>"))
            else:
                display(df_style)
        elif mode == "csv":
            print(df.to_csv(index=False, sep="\t"))
        else:
            print(df)
