"""
Assignment 2: AI Food Safety Inspector
Zero-Shot Prompting with Structured Outputs

Your mission: Analyze restaurant reviews and complaints to detect health violations
using only clear instructions — no training examples needed!
"""

import os
import json
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field
from enum import Enum
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser, JsonOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field

# --- Enums (Already provided, no change) ---

class ViolationCategory(Enum):
    TEMPERATURE_CONTROL = "Food Temperature Control"
    PERSONAL_HYGIENE = "Personal Hygiene"
    PEST_CONTROL = "Pest Control"
    CROSS_CONTAMINATION = "Cross Contamination"
    FACILITY_MAINTENANCE = "Facility Maintenance"
    UNKNOWN = "Unknown"


class SeverityLevel(Enum):
    CRITICAL = "Critical" # Highest severity, immediate health risk (e.g., pests, no refrigeration)
    HIGH = "High"        # Significant risk (e.g., cross-contamination, no handwashing)
    MEDIUM = "Medium"      # General risk (e.g., facility issues, minor personal hygiene)
    LOW = "Low"          # Minor risk (e.g., dirty floor, chipped plate)


class InspectionPriority(Enum):
    URGENT = "URGENT"    # Risk Score > 80, critical violations present
    HIGH = "HIGH"        # Risk Score 60-80, high violations present
    ROUTINE = "ROUTINE"    # Risk Score 20-59, medium/low violations only
    LOW = "LOW"          # Risk Score < 20, no violations or very minor concerns


# --- Pydantic Schemas for Structured Output ---
# Using Pydantic/BaseModel is the standard way to enforce output structure in LangChain.

class ViolationSchema(BaseModel):
    """Structured violation data schema for LLM output"""
    category: ViolationCategory = Field(
        description="The specific violation category from the Enum: TEMPERATURE_CONTROL, PERSONAL_HYGIENE, PEST_CONTROL, CROSS_CONTAMINATION, FACILITY_MAINTENANCE, or UNKNOWN."
    )
    description: str = Field(
        description="A clear, concise summary of the violation based on the text."
    )
    severity: SeverityLevel = Field(
        description="The level of risk based on the violation: CRITICAL, HIGH, MEDIUM, or LOW."
    )
    evidence: str = Field(
        description="The exact quote from the review text that provides evidence for the violation. Must be a direct quote."
    )
    confidence: float = Field(
        description="The confidence score (0.0 to 1.0) that this is a genuine violation."
    )

class AnalysisResult(BaseModel):
    """The full list of detected violations."""
    violations: List[ViolationSchema]


# --- Dataclasses for Internal Python Use (Already provided, but used for mapping) ---
# We keep these separate from Pydantic schema for clear internal data structure.

@dataclass
class Violation:
    """Structured violation data"""
    category: str
    description: str
    severity: str
    evidence: str
    confidence: float


@dataclass
class InspectionReport:
    """Complete inspection analysis"""
    restaurant_name: str
    overall_risk_score: int
    violations: List[Violation]
    inspection_priority: str
    recommended_actions: List[str]
    follow_up_required: bool
    source_count: int = 1 # Added for batch analysis


class FoodSafetyInspector:
    """
    AI-powered food safety analyzer using zero-shot structured prompting.
    """

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.1):
        """Initialize with LLM for consistent violation detection."""
        # TODO: Initialize an LLM for consistent JSON-style outputs
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        
        self.analysis_chain = None
        self.risk_chain = None
        self._setup_chains()

    def _setup_chains(self):
        """
        TODO #1: Create zero-shot prompts for violation detection and risk assessment.
        
        Create ONE chain for analysis using `with_structured_output` for robust JSON.
        The risk assessment will be handled by a simpler, rule-based chain for this assignment.
        """

        # --- Violation Detection Chain ---
        # The prompt guides the LLM on its role, the task, and the required format (implicitly handled by with_structured_output).
        analysis_template_str = (
            "You are a food safety inspector AI. Analyze the following text for potential health code violations.\n"
            "Your response MUST adhere strictly to the provided JSON schema for a list of violations.\n"
            "If no violations are found, return an empty list: {{\"violations\": []}}.\n\n"
            "Instructions:\n"
            "1. **Category & Severity**: Assign the most accurate `ViolationCategory` and `SeverityLevel` based on the context.\n"
            "2. **Evidence**: The `evidence` field MUST be an exact, direct quote from the `Text to analyze`.\n"
            "3. **Confidence**: Provide a confidence score (0.0 to 1.0) indicating how likely the evidence suggests a real violation.\n"
            "4. **Ambiguity**: If a concern is too vague or sarcastic, use a low confidence score (e.g., < 0.3) or the UNKNOWN category.\n\n"
            "Text to analyze: {review_text}\n"
        )
        analysis_prompt = PromptTemplate.from_template(analysis_template_str)

        # Set up the analysis chain using with_structured_output for JSON guarantee
        self.analysis_chain = analysis_prompt | self.llm.with_structured_output(AnalysisResult)

        # --- Risk Assessment Chain (Simpler Rule-based for this assignment) ---
        # We will use Python logic for risk calculation to ensure deterministic scoring,
        # but keep a simple LLM chain for generating a final summary/action list.
        risk_template_str = (
            "Based on the following violations, generate a list of 3-5 specific, recommended actions for the restaurant to address the issues. "
            "Also, determine if a follow-up inspection is absolutely required (return 'Yes' or 'No').\n\n"
            "Violations:\n{violations_summary}\n\n"
            "Format your output as a single JSON object:\n"
            "{{\n"
            "  \"recommended_actions\": [\n"
            "    \"Action 1\",\n"
            "    \"Action 2\"\n"
            "  ],\n"
            "  \"follow_up_required\": \"Yes\" or \"No\"\n"
            "}}"
        )
        risk_prompt = PromptTemplate.from_template(risk_template_str)
        self.risk_chain = risk_prompt | self.llm | JsonOutputParser()


    def detect_violations(self, text: str) -> List[Violation]:
        """
        TODO #2: Detect health violations from text input.
        """

        try:
            # Invoke the structured chain
            structured_result: AnalysisResult = self.analysis_chain.invoke({"review_text": text})

            # Map Pydantic/BaseModel objects to our internal Violation dataclasses
            violations: List[Violation] = []
            for v_schema in structured_result.violations:
                # The asdict() function is available for Pydantic v1 BaseModels (which we're using with langchain_core.pydantic_v1)
                # We need to manually convert the Enum values to strings for our dataclass
                violations.append(Violation(
                    category=v_schema.category.value,
                    description=v_schema.description,
                    severity=v_schema.severity.value,
                    evidence=v_schema.evidence,
                    confidence=v_schema.confidence
                ))
            
            # Apply confidence filter
            return self.filter_false_positives(violations)
            
        except Exception as e:
            # print(f"Error detecting violations/parsing JSON: {e}")
            # print(f"Raw analysis result might be: {raw_response}")
            return []


    def calculate_risk_score(self, violations: List[Violation]) -> Tuple[int, str]:
        """
        TODO #3: Calculate overall risk score and determine inspection priority.
        
        Implement a deterministic, rule-based scoring system.
        """

        score = 0
        critical_count = 0
        high_count = 0

        # Define points per severity level
        severity_points = {
            SeverityLevel.CRITICAL.value: 30,
            SeverityLevel.HIGH.value: 15,
            SeverityLevel.MEDIUM.value: 5,
            SeverityLevel.LOW.value: 1,
        }

        for v in violations:
            # Tally scores and counts based on confidence and severity
            points = severity_points.get(v.severity, 0)
            score += round(points * v.confidence)
            
            if v.severity == SeverityLevel.CRITICAL.value:
                critical_count += 1
            elif v.severity == SeverityLevel.HIGH.value:
                high_count += 1

        # Cap the score at 100
        risk_score = min(score, 100)

        # Determine Inspection Priority
        if critical_count > 0 or risk_score > 80:
            priority = InspectionPriority.URGENT.value
        elif high_count > 0 or risk_score >= 60:
            priority = InspectionPriority.HIGH.value
        elif risk_score >= 20:
            priority = InspectionPriority.ROUTINE.value
        else:
            priority = InspectionPriority.LOW.value
            
        return risk_score, priority


    def analyze_review(
        self, text: str, restaurant_name: str = "Unknown"
    ) -> InspectionReport:
        """
        TODO #4: Complete analysis pipeline for a single review.
        """

        # 1. Detect violations
        violations = self.detect_violations(text)
        
        # 2. Calculate risk score
        risk_score, priority = self.calculate_risk_score(violations)
        
        recommended_actions = []
        follow_up_required = False
        
        if violations:
            # Format violations for the recommendation chain
            violations_summary = "\n".join([
                f"- [{v.severity}] {v.category}: {v.description} (Evidence: '{v.evidence}')"
                for v in violations
            ])
            
            # 3. Generate recommendations and follow-up status using the risk chain
            try:
                risk_data = self.risk_chain.invoke({"violations_summary": violations_summary})
                recommended_actions = risk_data.get("recommended_actions", [])
                follow_up_required = risk_data.get("follow_up_required", "No").lower() == "yes"
            except Exception as e:
                # print(f"Error generating recommendations: {e}")
                # Fallback recommendations
                recommended_actions = ["Review LLM output parser settings."]
                follow_up_required = priority in [InspectionPriority.URGENT.value, InspectionPriority.HIGH.value]
        
        # 4. Create InspectionReport
        return InspectionReport(
            restaurant_name=restaurant_name,
            overall_risk_score=risk_score,
            violations=violations,
            inspection_priority=priority,
            recommended_actions=recommended_actions,
            follow_up_required=follow_up_required,
            source_count=1
        )


    def batch_analyze(self, reviews: List[Dict[str, str]]) -> InspectionReport:
        """
        TODO #5: Analyze multiple reviews for the same restaurant.
        """
        
        all_violations: List[Violation] = []
        
        # 1. Process all reviews individually
        for review in reviews:
            # We assume all reviews are for the same, unnamed restaurant for simplicity in this batch function
            violations = self.detect_violations(review["text"])
            
            # Add a source tag to evidence to help in deduplication
            for v in violations:
                v.evidence = f"[{review.get('source', 'Review')}] {v.evidence}"
                
            all_violations.extend(violations)

        # 2. Deduplication and Aggregation
        # Simple deduplication based on category and description/severity
        unique_violations: Dict[Tuple[str, str], Violation] = {}
        for v in all_violations:
            key = (v.category, v.severity)
            
            if key not in unique_violations:
                unique_violations[key] = v
            else:
                # If a similar violation is found, update the confidence to the highest reported value
                # and concatenate the evidence to show multiple sources.
                existing_v = unique_violations[key]
                existing_v.confidence = max(existing_v.confidence, v.confidence)
                if v.evidence not in existing_v.evidence: # Prevent adding identical evidence
                    existing_v.evidence += f"; {v.evidence}"
        
        final_violations = list(unique_violations.values())
        
        # 3. Calculate aggregate risk score
        aggregate_risk_score, aggregate_priority = self.calculate_risk_score(final_violations)

        # 4. Generate aggregated recommendations (similar to single review analysis)
        recommended_actions = []
        follow_up_required = False
        
        if final_violations:
            violations_summary = "\n".join([
                f"- [{v.severity}] {v.category}: {v.description} (Evidence: {v.evidence})"
                for v in final_violations
            ])
            try:
                risk_data = self.risk_chain.invoke({"violations_summary": violations_summary})
                recommended_actions = risk_data.get("recommended_actions", [])
                follow_up_required = risk_data.get("follow_up_required", "No").lower() == "yes"
            except Exception:
                recommended_actions = ["Review aggregated LLM output."]
                follow_up_required = aggregate_priority in [InspectionPriority.URGENT.value, InspectionPriority.HIGH.value]


        # 5. Create the Aggregated InspectionReport
        return InspectionReport(
            restaurant_name="Batch Analysis Target",
            overall_risk_score=aggregate_risk_score,
            violations=final_violations,
            inspection_priority=aggregate_priority,
            recommended_actions=recommended_actions,
            follow_up_required=follow_up_required,
            source_count=len(reviews)
        )


    def filter_false_positives(self, violations: List[Violation]) -> List[Violation]:
        """
        TODO #6 (Bonus): Filter out likely false positives.
        
        We will implement filtering based on a confidence threshold.
        """

        # Filter: Only keep violations with a confidence of 0.5 or higher
        # LLMs are instructed to use low confidence for vague/sarcastic claims
        CONFIDENCE_THRESHOLD = 0.5
        
        filtered = [v for v in violations if v.confidence >= CONFIDENCE_THRESHOLD]
        
        return filtered


def test_inspector():
    """Test the food safety inspector with various scenarios."""

    inspector = FoodSafetyInspector()

    # Test cases with varying violation types
    test_reviews = [
        {
            "restaurant": "Bob's Burgers",
            "text": "Great food but saw a mouse run across the dining room! Also, the chef wasn't wearing gloves while handling raw chicken.",
        },
        {
            "restaurant": "Pizza Palace",
            "text": "Just left and the bathroom had no soap, and I'm pretty sure that meat sitting on the counter wasn't refrigerated 😷",
        },
        {
            "restaurant": "Sushi Express",
            "text": "Love this place! Though it's weird they keep the raw fish next to the vegetables #sushitime #questionable",
        },
        {
            "restaurant": "Taco Town",
            "text": "Best tacos in town! Super clean kitchen, staff always wears hairnets, everything looks fresh!",
        },
        {
            "restaurant": "Burger Barn",
            "text": "The cockroach in my salad added extra protein! Just kidding, but seriously the place needs cleaning.",
        },
    ]

    print("🍽️ FOOD SAFETY INSPECTION SYSTEM 🍽️\n")
    print("=" * 70)

    for review_data in test_reviews:
        print(f"\n🏪 Restaurant: {review_data['restaurant']}")
        print(f"📝 Review: \"{review_data['text'][:100]}...\"")

        # Analyze the review
        report = inspector.analyze_review(
            review_data["text"], review_data["restaurant"]
        )

        # Display results
        print(f"\n📊 Inspection Report:")
        print(f"  Risk Score: {report.overall_risk_score}/100")
        print(f"  Priority: {report.inspection_priority}")
        print(f"  Violations Found: {len(report.violations)}")

        if report.violations:
            print("\n  Detected Violations:")
            for v in report.violations:
                print(f"    • [{v.severity}] {v.category}: {v.description}")
                # Ensure evidence doesn't break the print layout
                evidence_to_print = v.evidence.replace('\n', ' ') 
                print(f'      Evidence: "{evidence_to_print[:50]}..."')
                print(f"      Confidence: {v.confidence:.0%}")

        if report.recommended_actions:
            print("\n  Recommended Actions:")
            for action in report.recommended_actions:
                print(f"    ✓ {action}")

        print(f"\n  Follow-up Required: {'Yes' if report.follow_up_required else 'No'}")
        print("-" * 70)

    # Test batch analysis
    print("\n🔬 BATCH ANALYSIS TEST:")
    print("=" * 70)

    # Multiple reviews for same restaurant
    batch_reviews = [
        {"text": "Saw bugs in the kitchen!", "source": "Yelp"},
        {"text": "Food was cold and undercooked and I'm pretty sure I saw the cook touch his hair.", "source": "Google"},
        {"text": "Staff not wearing hairnets, and the floors are super greasy.", "source": "Twitter"},
        {"text": "The temp on my soup was lukewarm, and the bathroom soap dispenser was empty.", "source": "Yelp"},
    ]

    # TODO: Uncomment when batch_analyze is implemented
    batch_report = inspector.batch_analyze(batch_reviews)
    
    print(f"Aggregated from {batch_report.source_count} reviews.")
    print(f"Aggregate Risk Score: {batch_report.overall_risk_score}/100")
    print(f"Aggregate Priority: {batch_report.inspection_priority}")
    print(f"Total Unique Violations: {len(batch_report.violations)}")
    
    if batch_report.violations:
        print("\n  Aggregated Violations:")
        for v in batch_report.violations:
            print(f"    • [{v.severity}] {v.category}: {v.description}")
            evidence_to_print = v.evidence.replace('\n', ' ') 
            print(f'      Evidence: "{evidence_to_print[:50]}..."')
            print(f"      Confidence: {v.confidence:.0%}")

    if batch_report.recommended_actions:
        print("\n  Recommended Actions:")
        for action in batch_report.recommended_actions:
            print(f"    ✓ {action}")

    print(f"\n  Follow-up Required: {'Yes' if batch_report.follow_up_required else 'No'}")
    print("-" * 70)


if __name__ == "__main__":
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running. Example: 'export OPENAI_API_KEY=...'")
    else:
        test_inspector()