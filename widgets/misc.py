import logging

import pandas as pd
from IPython.display import display

logger = logging.getLogger(__name__)

def display_table(table, sort=None, index=None, columns=None, mode="html", align=None):
    df = pd.DataFrame(table)
    if sort is not None:
        df = df.sort_values(sort)
    if index is not None:
        df = df.set_index(index)
    if columns is not None:
        df = df.reindex(columns, axis="columns")
    if mode == "html":
        df_style = df.style
    if align is not None:
        if mode == "html":
            df_style = df_style.set_properties(**{"text-align": align}).set_table_styles([dict(selector="th", props=[("text-align", align)])])
        else:
            logger.warning("Aligning columns is only supported in mode \"html\".")
    with pd.option_context(
        "display.expand_frame_repr", False,
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
    ):
        if mode == "html":
            display(df_style)
        elif mode == "csv":
            print(df.to_csv(index=False, sep="\t"))
        else:
            print(df)
