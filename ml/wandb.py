import pandas as pd


def load_wandb_summary(path, metrics=None):
    """Load the summary of a W&B run from a CSV file."""
    # load data
    df = pd.read_csv(path)
    df = df.set_index("Name")

    # select desired columns and rename
    if metrics is not None:
        df = df[metrics.keys()].rename(columns=metrics)
    return df


def load_wandb_history(path, metrics=None):
    """Load the history of a W&B run from a CSV file."""
    # load data
    df = pd.read_csv(path)

    # parse experiment and metric names from column names
    columns_general = []
    experiment_names = set()
    metric_names = set()
    for col in df.columns:
        if " - " in col:
            experiment_name, metric_name = col.split(" - ", 1)
            experiment_names.add(experiment_name)
            metric_names.add(metric_name)
        else:
            columns_general.append(col)
    metric_names = sorted(metric_names)

    # extract history per experiment
    dfs_by_experiment = {}
    for experiment_name in experiment_names:
        # extract columns specific to the current experiment
        columns_specific = {f"{experiment_name} - {metric_name}": metric_name for metric_name in metric_names}
        df_experiment = (
            df[columns_general + list(columns_specific.keys())]
            .rename(columns=columns_specific)
            .dropna(how="all", subset=columns_specific.values())
        )
        # select desired columns and rename
        if metrics is not None:
            df_experiment = df_experiment[metrics.keys()].rename(columns=metrics)
        dfs_by_experiment[experiment_name] = df_experiment

    return dfs_by_experiment
