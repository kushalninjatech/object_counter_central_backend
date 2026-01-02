"""
LLM Service for numberplate extraction using Google Gemini.
"""
import base64
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from PIL import Image

from app.core.config import settings
from app.core.prompts import NUMBERPLATE_EXTRACTION_PROMPT
from app.core.logging import app_logger as logger
from app.schemas.llm_schemas import NumberplateExtractionResult
from app.services.storage_service import get_storage_service


class LLMService:
    """
    Service for extracting numberplate information from vehicle images using LLM.

    Uses Google Gemini Flash model with structured output via LangChain.
    """

    def __init__(self):
        """Initialize LLM service with settings from config."""
        if not settings.GOOGLE_API_KEY:
            raise ValueError("GOOGLE_API_KEY not set in environment variables")

        # Initialize Gemini Flash model with structured output
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-3-flash-preview",  # Latest flash model
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0,  # Deterministic output
            max_tokens=None,  # No limit on output
            timeout=None,
            max_retries=2,
        )

        # Create structured output parser
        self.structured_llm = self.llm.with_structured_output(
            NumberplateExtractionResult
        )

        logger.info("LLM Service initialized with Gemini Flash")

    def extract_numberplate(
        self,
        image_path: str,
        image_data: Optional[bytes] = None
    ) -> NumberplateExtractionResult:
        """
        Extract numberplate information from a vehicle image.

        Args:
            image_path: Path to the image file
            image_data: Optional raw image bytes (if already loaded)

        Returns:
            NumberplateExtractionResult with extracted information

        Raises:
            Exception: If LLM call fails or image cannot be processed
        """
        try:
            logger.info(f"Processing image for numberplate extraction: {image_path}")

            # Load image if not provided
            if image_data is None:
                storage_service = get_storage_service()
                image_data = storage_service.get_file(image_path)

            # Encode image to base64
            image_base64 = base64.b64encode(image_data).decode("utf-8")

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": NUMBERPLATE_EXTRACTION_PROMPT},
                    {
                        "type": "image_url",
                        "image_url": f"data:image/jpeg;base64,{image_base64}"
                    }
                ]
            )

            # Call LLM with structured output
            logger.info("Sending image to Gemini Flash LLM...")
            result: NumberplateExtractionResult = self.structured_llm.invoke([message])

            # Handle None response from LLM (schema mismatch or parsing failure)
            if result is None:
                logger.warning("LLM returned None - schema mismatch or parsing failure")
                from app.schemas.llm_schemas import NumberplateColor, VehicleSide
                result = NumberplateExtractionResult(
                    numberplate_available=False,
                    numberplate_text=None,
                    numberplate_color=NumberplateColor.UNKNOWN,
                    vehicle_side=VehicleSide.UNKNOWN,
                    confidence_score=0.0,
                    reasoning="LLM failed to generate structured output - schema mismatch"
                )

            logger.info(
                f"LLM extraction complete: "
                f"available={result.numberplate_available}, "
                f"text={result.numberplate_text}, "
                f"confidence={result.confidence_score}"
            )

            return result

        except Exception as exc:
            logger.error(f"Error in LLM numberplate extraction: {exc}")
            raise

    def validate_image(self, image_path: str) -> bool:
        """
        Validate if image file is readable.

        Args:
            image_path: Path to image file

        Returns:
            True if image is valid, False otherwise
        """
        try:
            with Image.open(image_path) as img:
                img.verify()
            return True
        except Exception as exc:
            logger.error(f"Image validation failed for {image_path}: {exc}")
            return False


# Singleton instance
_llm_service_instance: Optional[LLMService] = None


def get_llm_service() -> LLMService:
    """
    Get singleton instance of LLM service.

    Returns:
        LLMService instance
    """
    global _llm_service_instance
    if _llm_service_instance is None:
        _llm_service_instance = LLMService()
    return _llm_service_instance
