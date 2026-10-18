import pandas as pd

from src.data.cleaning import clean


def _row(**kw):
    base = dict(IsCanceled=0, LeadTime=10, Adults=2, Children=0.0, Babies=0, StaysInWeekendNights=1,
                StaysInWeekNights=2, ADR=100.0, Country="PRT", Agent="9", Company="NULL")
    base.update(kw)
    return base


def test_clean_removes_duplicates_and_invalid_rows():
    df = pd.DataFrame([
        _row(), _row(),                       # exact duplicate
        _row(Adults=0),                       # zero guests
        _row(StaysInWeekendNights=0, StaysInWeekNights=0),  # zero nights
        _row(ADR=5400.0),                     # outlier
        _row(ADR=-5.0),                       # negative ADR
        _row(Children=float("nan"), Country=None, Agent="NULL", Company="40", LeadTime=11),
    ])
    out, rep = clean(df)
    assert len(out) == 2
    assert rep["duplicates_removed"] == 1
    assert out["Children"].isna().sum() == 0
    assert (out["Country"] == "Unknown").sum() == 1


def test_agent_company_null_become_flags():
    df = pd.DataFrame([_row(Agent="   NULL", Company="NULL"), _row(Agent="9", Company="40", LeadTime=3)])
    out, _ = clean(df)
    flags = out.set_index("LeadTime")[["HasAgent", "HasCompany"]]
    assert flags.loc[10].tolist() == [0, 0]
    assert flags.loc[3].tolist() == [1, 1]
