import logging

import pandas as pd
from IPython.display import display

logger = logging.getLogger(__name__)

def display_table(table, columns=None, sort=None, html=True, align=None):
    df = pd.DataFrame(table)
    if columns is not None:
        df = df[columns]
    if sort is not None:
        df = df.sort_values(sort)
    if html:
        df_style = df.style
    if align is not None:
        if html:
            df_style = df_style.set_properties(**{"text-align": align}).set_table_styles([dict(selector="th", props=[("text-align", align)])])
        else:
            logger.warning("Aligning columns is not supported in text-mode.")
    with pd.option_context(
        "display.expand_frame_repr", False,
        "display.max_rows", None,
        "display.max_columns", None,
        "display.max_colwidth", None,
    ):
        if html:
            display(df_style)
        else:
            print(df)
