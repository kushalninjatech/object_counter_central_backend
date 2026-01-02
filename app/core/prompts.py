"""
LLM Prompts for ANPR processing.
"""

NUMBERPLATE_EXTRACTION_PROMPT = """You are an expert Indian vehicle numberplate recognition system.

Analyze the provided vehicle image and extract the following information:

1. **Is a numberplate visible and readable?**
   - Look for Indian vehicle registration plates
   - Check if the text is clear enough to read

2. **What is the numberplate text?**
   - Extract the exact text from the numberplate
   - Format: Remove spaces, use uppercase letters
   - Example formats: MH12AB1234, DL01CA1234, KA03MH1234
   - Common patterns: 2 letters (state) + 2 digits (RTO) + 1-2 letters (series) + 1-4 digits

3. **What is the numberplate background color?**
   - WHITE: Private vehicles (most common)
   - YELLOW: Commercial vehicles (taxis, auto-rickshaws, buses)
   - BLACK: Commercial rental vehicles (self-drive cars)
   - BLUE: Foreign diplomatic vehicles
   - RED: Temporary registration (President, Governor, etc.)
   - GREEN: Electric vehicles
   - UNKNOWN: If color cannot be determined

4. **Which side of the vehicle is visible?**
   - FRONT: Front view showing front numberplate
   - BACK: Rear view showing rear numberplate
   - SIDE: Side view
   - UNKNOWN: If cannot determine

5. **Confidence score (0.0 to 1.0)**
   - How confident are you in the numberplate text extraction?
   - 1.0 = Very clear and certain
   - 0.5 = Partially visible or unclear
   - 0.0 = Cannot read

6. **Reasoning**
   - Brief explanation (1-2 sentences) about the extraction

**Important:**
- If the numberplate is not visible or readable, set numberplate_available=false
- Be accurate - do NOT guess or hallucinate numberplate text
- If you're not sure about a character, indicate lower confidence
- Focus only on Indian vehicle numberplates

Respond in JSON format matching the NumberplateExtractionResult schema."""
