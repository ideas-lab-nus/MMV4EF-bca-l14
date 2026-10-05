"""Explicit Set2 encodings and honest focused temperature limits."""
import math
import numpy as np

SET2 = ("#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f", "#e5c494", "#b3b3b3")
FORECAST_COLORS = {("no_pv", "observed"): SET2[2], ("no_pv", "lstm64"): SET2[1],
                   ("onsite_pv", "observed"): SET2[0], ("onsite_pv", "lstm64"): SET2[3]}
GRID_COLORS = {"observed": SET2[4], "lstm64": SET2[6]}
SELF_COLORS = {"observed": SET2[5], "lstm64": SET2[7]}
WEIGHT_STYLES = {100.0: "-", 1000.0: (0, (6, 2)), 10000.0: (0, (5, 2, 1, 2))}
WEIGHT_OFFSETS = {100.0: -.025, 1000.0: 0.0, 10000.0: .025}
AXIS_AUDIT = []


def focus_temperature_axes(axes, *, label):
    """Use one data-covering range for directly compared temperature panels."""
    values=[]
    for axis in axes:
        for line in axis.lines:
            arr=np.asarray(line.get_ydata(),dtype=float)
            values.extend(arr[np.isfinite(arr)].tolist())
    if not values:
        raise ValueError("Temperature axis has no finite plotted values")
    minimum,maximum=min(values),max(values)
    padding=max(.15,.06*(maximum-minimum))
    low=math.floor((minimum-padding)*4)/4
    high=math.ceil((maximum+padding)*4)/4
    if high-low<1:
        mid=(low+high)/2
        low,high=mid-.5,mid+.5
    assert low<=minimum and high>=maximum
    for axis in axes:
        axis.set_ylim(low,high)
    AXIS_AUDIT.append({"label":label,"plotted_min_c":minimum,"plotted_max_c":maximum,
                       "limits_c":[low,high],"compared_axes":len(axes),"all_values_covered":True})
    return low,high
