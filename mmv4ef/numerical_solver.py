"""Validate numerical solves and competing first-mode branches in-place.

No objective or physical constraint is changed. An alternative first-mode
constraint is used only on a disposable copy; a better feasible candidate is
returned to the unrestricted model as a MIP start before controls are read.
"""
import time

import gurobipy as gp
from gurobipy import GRB


def quality(model):
    return {"status": int(model.Status), "gap": float(model.MIPGap) if model.SolCount else None,
            "objective": float(model.ObjVal) if model.SolCount else None,
            "constraint_violation": float(model.ConstrVio) if model.SolCount else None,
            "bound_violation": float(model.BoundVio) if model.SolCount else None,
            "integer_violation": float(model.IntVio) if model.SolCount else None,
            "runtime_s": float(model.Runtime), "numeric_focus": int(model.Params.NumericFocus),
            "presolve": int(model.Params.Presolve)}


def accepted(model):
    return (model.Status == GRB.OPTIMAL and model.SolCount and model.MIPGap <= .00010001
            and model.ConstrVio <= 2e-6 and model.BoundVio <= 2e-6 and model.IntVio <= 2e-5)


def solve_checked(model, *, allow_infeasible=False):
    attempts = []
    for trial in range(3):
        if trial:
            model.reset()
            model.Params.NumericFocus = 3
            model.Params.Presolve = 0
            model.Params.DualReductions = 0
            model.Params.FeasibilityTol = 1e-8
            model.Params.OptimalityTol = 1e-8
            if trial == 2:
                model.Params.MIPGap = 1e-6
                model.Params.MIPFocus = 1
        model.optimize()
        attempts.append(quality(model))
        if accepted(model):
            return attempts
        if allow_infeasible and trial and model.Status == GRB.INFEASIBLE:
            return attempts
    raise RuntimeError(f"Numerical validation rejected the solve after three attempts: {attempts}")


def validated_optimize(model, *, check_first_mode):
    started = time.perf_counter()
    attempts = solve_checked(model)
    initial = quality(model)
    initial_mode = int(round(model.getVarByName("z_mode[0]").X))
    report = {"revision": "numericfocus2_redundant_dwell_removed_first_branch_check_v1",
              "first_mode_checked": bool(check_first_mode), "initial": initial,
              "initial_mode": initial_mode, "initial_attempts": attempts,
              "alternative": None, "recovered_from_better_branch": False}
    if check_first_mode:
        alternative = model.copy()
        try:
            alternative.addConstr(alternative.getVarByName("z_mode[0]") == 1-initial_mode,
                                  name="numerical_first_mode_challenger")
            branch_attempts = solve_checked(alternative, allow_infeasible=True)
            report["alternative"] = {**quality(alternative), "forced_first_mode": 1-initial_mode,
                                     "attempts": branch_attempts}
            if alternative.SolCount and alternative.ObjVal < model.ObjVal-1e-6:
                target = alternative.ObjVal
                point = {v.VarName: v.X for v in alternative.getVars()}
                fingerprint = model.Fingerprint
                model.reset()
                for var in model.getVars():
                    var.Start = point[var.VarName]
                model.Params.NumericFocus = 3
                model.Params.Presolve = 0
                recovery_attempts = solve_checked(model)
                assert model.ObjVal <= target+1e-6, ("Unrestricted model lost feasible challenger", target, quality(model))
                report["recovered_from_better_branch"] = True
                report["recovery_attempts"] = recovery_attempts
                report["before_start_fingerprint"] = fingerprint
                report["challenger_objective"] = target
        finally:
            alternative.dispose()
    report["final"] = quality(model)
    report["final_mode"] = int(round(model.getVarByName("z_mode[0]").X))
    report["validation_wall_s"] = time.perf_counter()-started
    assert accepted(model)
    return report
