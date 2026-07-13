"""
Core/Pre_Compliance_Check/standards.py

-------------------------------------------------------
Regulatory database for the Sound Quality Framework.

This module stores all compliance standards in a
common format so that new standards (AIS-145,
Hospital Alarm, IEC 60601-1-8, EV Motor, etc.)
can be added without changing the compliance engine.


"""

from dataclasses import dataclass
from typing import Dict, List


# -----------------------------------------------------
# Standard Data Structure
# -----------------------------------------------------

@dataclass(frozen=True)
class AcousticStandard:
    """
    Generic acoustic standard.
    """

    name: str
    version: str

    octave_bands: List[int]

    minimum_octave_10: List[float]
    minimum_octave_20: List[float]

    overall_forward_limits: Dict[int, tuple]
    overall_reverse_limits: Dict[int, tuple]

    background_margin_db: float

    calibration_required: bool

    requires_frequency_shift: bool

    minimum_frequency_shift: float


# -----------------------------------------------------
# UN Regulation 138
# Amendment 4 (2025)
# -----------------------------------------------------

UN_R138 = AcousticStandard(

    name="UN Regulation No.138",

    version="Revision 1 Amendment 4",

    octave_bands=[

        160,
        200,
        250,
        315,
        400,
        500,
        630,
        800,
        1000,
        1250,
        1600,
        2000,
        2500,
        3150,
        4000,
        5000

    ],

    minimum_octave_10=[

        45,
        44,
        43,
        44,
        45,
        45,
        46,
        46,
        46,
        46,
        44,
        42,
        39,
        36,
        34,
        31

    ],

    minimum_octave_20=[

        50,
        49,
        48,
        49,
        50,
        50,
        51,
        51,
        51,
        51,
        49,
        47,
        44,
        41,
        39,
        36

    ],

    # (Minimum, Maximum)

    overall_forward_limits={

        0: (None,69),

        10:(50,75),

        20:(56,75)

    },

    overall_reverse_limits={

        0:(47,69),

        6:(47,75),

        20:(47,75)

    },

    background_margin_db=6,

    calibration_required=True,

    requires_frequency_shift=True,

    minimum_frequency_shift=0.8

)


# -----------------------------------------------------
# AIS-145
#
# Currently follows UN R138.
# Easily replace later if revised.
# -----------------------------------------------------

AIS145 = AcousticStandard(

    **{

        **UN_R138.__dict__,

        "name":"AIS-145"

    }

)


# -----------------------------------------------------
# Placeholder
#
# Example of future extension
# -----------------------------------------------------

HOSPITAL_ALARM = AcousticStandard(

    name="IEC 60601-1-8",

    version="Future",

    octave_bands=[],

    minimum_octave_10=[],

    minimum_octave_20=[],

    overall_forward_limits={},

    overall_reverse_limits={},

    background_margin_db=6,

    calibration_required=False,

    requires_frequency_shift=False,

    minimum_frequency_shift=0

)


# -----------------------------------------------------
# Database
# -----------------------------------------------------

SUPPORTED_STANDARDS = {

    "UN R138": UN_R138,

    "AIS-145": AIS145,

    "Hospital Alarm": HOSPITAL_ALARM

}


# -----------------------------------------------------
# API
# -----------------------------------------------------

class Standards:

    """
    Regulatory database.
    """

    @staticmethod
    def get(name: str) -> AcousticStandard:

        if name not in SUPPORTED_STANDARDS:

            raise ValueError(

                f"Unknown standard: {name}"

            )

        return SUPPORTED_STANDARDS[name]

    @staticmethod
    def available():

        return list(

            SUPPORTED_STANDARDS.keys()

        )