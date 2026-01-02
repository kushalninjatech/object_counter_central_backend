"""
Pydantic schemas for LLM structured output.
"""
from typing import Optional
from pydantic import BaseModel, Field
from enum import Enum


class NumberplateColor(str, Enum):
    """Indian numberplate color options."""
    WHITE = "white"
    YELLOW = "yellow"
    BLACK = "black"
    BLUE = "blue"
    RED = "red"
    GREEN = "green"
    UNKNOWN = "unknown"


class VehicleSide(str, Enum):
    """Vehicle side captured in image."""
    FRONT = "front"
    BACK = "back"
    SIDE = "side"
    UNKNOWN = "unknown"


class NumberplateExtractionResult(BaseModel):
    """
    Structured output from LLM for numberplate extraction.

    This schema defines exactly what we expect the LLM to return
    when analyzing a vehicle image.
    """

    numberplate_available: bool = Field(
        description="Whether a numberplate is visible and readable in the image"
    )

    numberplate_text: Optional[str] = Field(
        default=None,
        description=(
            "The text on the numberplate if available. "
            "Should be in uppercase without spaces. "
            "Example: MH12AB1234"
        )
    )

    numberplate_color: NumberplateColor = Field(
        default=NumberplateColor.UNKNOWN,
        description=(
            "Color of the numberplate background. "
            "Options: white (private vehicles), yellow (commercial taxis/autos), "
            "black (rental), blue (diplomatic), red (temporary), green (electric)"
        )
    )

    vehicle_side: VehicleSide = Field(
        default=VehicleSide.UNKNOWN,
        description="Which side of the vehicle is visible in the image"
    )

    confidence_score: float = Field(
        default=0.0,
        ge=0.0,
        le=1.0,
        description="Confidence score for the numberplate text extraction (0.0 to 1.0)"
    )

    reasoning: Optional[str] = Field(
        default=None,
        description=(
            "Brief explanation of the extraction result. "
            "Why the numberplate was or wasn't readable."
        )
    )

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "numberplate_available": True,
                    "numberplate_text": "MH12AB1234",
                    "numberplate_color": "white",
                    "vehicle_side": "front",
                    "confidence_score": 0.95,
                    "reasoning": "Clear front view with well-lit numberplate"
                }
            ]
        }
    }
