"""
Assignment 3: Transit Delay Explainer

Focus: Prompt templates, model configs (temperature/top_p), and LCEL chaining

Scenario: Convert terse transit operations bulletins into a rider-facing advisory
with two bullet points: cause + action.
"""

import os
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class TransitExplainer:
    def __init__(self):
        # Create two LLMs: calm and creative
        self.calm_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2
        )

        self.creative_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.8,
            top_p=0.9
        )

        # Build role-aware prompt
        system_prompt = (
            "You rewrite internal operations notes into concise rider guidance "
            "with exactly two bullets:\n"
            "• Bullet 1: Plain-language cause\n"
            "• Bullet 2: What riders should do now\n"
            "Keep it friendly, calm, and clear."
        )

        user_prompt = (
            "Line: {line_name}\n"
            "Status: {status_text}\n"
            "Return only 2 bullet points."
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("user", user_prompt),
            ]
        )

        # Create chains
        self.calm_chain = self.prompt | self.calm_llm | StrOutputParser()
        self.creative_chain = self.prompt | self.creative_llm | StrOutputParser()

    def explain(self, line_name: str, status_text: str) -> str:
        """
        Invoke both chains and return the calm version.
        The creative version is generated for comparison (not returned).
        """

        calm = self.calm_chain.invoke(
            {"line_name": line_name, "status_text": status_text}
        )

        # Optional: generate creative version (not returned)
        _ = self.creative_chain.invoke(
            {"line_name": line_name, "status_text": status_text}
        )

        return calm


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    explainer = TransitExplainer()

    samples = [
        ("Green Line", "Signal failure near Station X causing cascading delays."),
        (
            "Red Line",
            "Unplanned track inspection between A and B, single-tracking in effect.",
        ),
    ]

    print("\n🚌 Transit Delay Explainer — demo\n" + "-" * 48)
    for line, status in samples:
        print(f"\nLine: {line}\nStatus: {status}")
        print(explainer.explain(line, status))


if __name__ == "__main__":
    _demo()
