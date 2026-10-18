"""Prediction points, feature sets per task and the leakage map (31 variables x 3 tasks).

Legend: "ok" = known at the prediction point | "doubt" = decide + justify + ablation | "leak" = never.
Default rule: "doubt" columns are EXCLUDED from the main model; one ablation reports the gap.
"""
import pandas as pd

PREDICTION_POINT = {
    "regression": "ADR is predicted when a booking is quoted / created.",
    "classification": "IsCanceled is predicted when the booking enters the PMS.",
    "clustering": "Booking profile known at booking time (no outcome).",
}

# variable: (ADR, IsCanceled, clustering, why)
LEAKAGE_MAP = {
    "IsCanceled": ("leak", "TARGET", "leak", "future outcome / target of task B"),
    "LeadTime": ("ok", "ok", "ok", "known at booking"),
    "ArrivalDateYear": ("doubt", "doubt", "leak", "drift 2015-17; dropped in baselines (question for instructor)"),
    "ArrivalDateMonth": ("ok", "ok", "ok", "season"),
    "ArrivalDateWeekNumber": ("ok", "ok", "doubt", "redundant with month"),
    "ArrivalDateDayOfMonth": ("ok", "ok", "doubt", "weak signal"),
    "StaysInWeekendNights": ("ok", "ok", "ok", "as booked (can be amended)"),
    "StaysInWeekNights": ("ok", "ok", "ok", "as booked"),
    "Adults": ("ok", "ok", "ok", ""),
    "Children": ("ok", "ok", "ok", "few NaN -> 0"),
    "Babies": ("ok", "ok", "ok", ""),
    "Meal": ("ok", "ok", "ok", ""),
    "Country": ("ok", "ok", "doubt", "high cardinality -> rare levels grouped"),
    "MarketSegment": ("ok", "ok", "ok", ""),
    "DistributionChannel": ("ok", "ok", "ok", ""),
    "IsRepeatedGuest": ("ok", "ok", "ok", ""),
    "PreviousCancellations": ("ok", "ok", "ok", "history BEFORE this booking"),
    "PreviousBookingsNotCanceled": ("ok", "ok", "ok", "history BEFORE this booking"),
    "ReservedRoomType": ("ok", "ok", "ok", "what was requested"),
    "AssignedRoomType": ("leak", "leak", "leak", "set at check-in -> indirect leak"),
    "BookingChanges": ("doubt", "leak", "doubt", "grows until check-in / cancellation"),
    "DepositType": ("ok", "doubt", "ok", "'Non Refund' behaves like a cancellation artefact"),
    "Agent": ("ok", "ok", "doubt", "ID; NULL = no agent -> HasAgent flag"),
    "Company": ("ok", "ok", "doubt", "~95% NULL -> HasCompany flag"),
    "DaysInWaitingList": ("doubt", "doubt", "doubt", "known only after confirmation"),
    "CustomerType": ("ok", "ok", "ok", ""),
    "ADR": ("TARGET", "doubt", "doubt", "target of task A; price known later?"),
    "RequiredCarParkingSpaces": ("doubt", "doubt", "doubt", "may be added after booking"),
    "TotalOfSpecialRequests": ("doubt", "doubt", "doubt", "accumulates over time"),
    "ReservationStatus": ("leak", "leak", "leak", "Canceled / No-Show / Check-Out = the answer"),
    "ReservationStatusDate": ("leak", "leak", "leak", "date of that final status"),
}

FEATURES = {
    "regression": {
        "numeric": ["LeadTime", "TotalNights", "TotalGuests", "IsRepeatedGuest",
                    "PreviousCancellations", "PreviousBookingsNotCanceled",
                    "HasAgent", "HasCompany"],
        "categorical": ["ArrivalDateMonth", "Meal", "Country", "MarketSegment",
                        "DistributionChannel", "DepositType", "CustomerType",
                        "ReservedRoomType"],
    },
    "classification": {
        "numeric": ["LeadTime", "StaysInWeekendNights", "StaysInWeekNights",
                    "Adults", "Children", "Babies", "IsRepeatedGuest",
                    "PreviousCancellations", "PreviousBookingsNotCanceled",
                    "HasAgent", "HasCompany"],
        "categorical": ["ArrivalDateMonth", "Meal", "Country", "MarketSegment",
                        "DistributionChannel", "ReservedRoomType", "CustomerType"],
    },
    "clustering": {
        "numeric": ["LeadTime", "TotalNights", "TotalGuests", "ADR", "IsRepeatedGuest"],
        "categorical": ["MarketSegment", "DistributionChannel", "CustomerType",
                        "DepositType", "Meal", "ArrivalDateMonth"],
    },
}


def leakage_table():
    rows = [(v, *vals) for v, vals in LEAKAGE_MAP.items()]
    return pd.DataFrame(rows, columns=["variable", "ADR_regression", "IsCanceled_classification",
                                       "clustering", "why"])
