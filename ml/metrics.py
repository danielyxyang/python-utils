import numpy as np

from ..math import safe_div


def calibration_curve(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 20,
    return_error: bool = True,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute calibration curve for the predictions."""
    n_samples = len(y_pred)

    # define bins
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    binids = np.digitize(y_pred, bins[1:-1])

    # compute accuracy and average confidence per bin
    bin_count_true = np.bincount(binids, weights=y_true, minlength=n_bins)
    bin_count_pred = np.bincount(binids, weights=y_pred, minlength=n_bins)
    bin_count = np.bincount(binids, minlength=n_bins)
    bin_prob_true = safe_div(bin_count_true, bin_count, default=np.nan)
    bin_prob_pred = safe_div(bin_count_pred, bin_count, default=np.nan)
    output = {
        "curve": (bin_prob_true, bin_prob_pred, bins, bin_count),
    }

    if return_error:
        # compute calibration error
        calibration_error = np.abs(bin_prob_true - bin_prob_pred)
        output["ece"] = np.nansum(bin_count / n_samples * calibration_error).item()
        output["mce"] = np.nanmax(calibration_error).item()

    return output
