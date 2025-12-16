"""
Assignment 10: Synthesis Orchestrator (Two-Stage Pipeline)
"""

import os
from typing import List

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class SynthesisOrchestrator:
    """Two-stage pipeline: extractor (batch) → synthesizer (single)."""

    def __init__(self):
        # -------- Prompt strings --------
        self.extractor_system = (
            "You extract 1–2 key factual claims from a short note. "
            "Be neutral, concise, and do not add opinions."
        )
        self.extractor_user = (
            "Note:\n{note}\n\n"
            "Return bullet points of the key claims only."
        )

        self.synth_system = (
            "You synthesize claims from multiple sources into a balanced summary. "
            "Highlight areas of agreement and conflict clearly."
        )
        self.synth_user = (
            "Claims from multiple notes:\n{claims}\n\n"
            "Return sections:\n"
            "OVERALL SUMMARY:\n"
            "AGREEMENTS:\n"
            "CONFLICTS:\n"
            "Keep the response concise."
        )

        # -------- Build prompts --------
        self.extract_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.extractor_system),
                ("user", self.extractor_user),
            ]
        )

        self.synth_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.synth_system),
                ("user", self.synth_user),
            ]
        )

        # -------- LLM --------
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
        )

        # -------- Chains --------
        self.extract_chain = self.extract_prompt | self.llm | StrOutputParser()
        self.synth_chain = self.synth_prompt | self.llm | StrOutputParser()

    def extract_claims(self, notes: List[str]) -> List[str]:
        """Return extracted claims (as strings), one per note."""
        inputs = [{"note": n} for n in notes]
        return self.extract_chain.batch(inputs)

    def synthesize(self, claims: List[str]) -> str:
        """Return a synthesis from already-extracted claims."""
        joined_claims = "\n".join(f"- {c}" for c in claims)
        return self.synth_chain.invoke({"claims": joined_claims})

    def run(self, notes: List[str]) -> str:
        """End-to-end: extract claims (batch) then synthesize."""
        extracted = self.extract_claims(notes)
        return self.synthesize(extracted)


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    orch = SynthesisOrchestrator()

    notes = [
        "Team A reduced latency by 20% after switching cache strategy.",
        "Users report fewer timeouts; however, spikes still occur on Mondays.",
        "Data suggests cache hit rate improved but cold-starts remain high.",
    ]

    print("\n🧪 Synthesis Orchestrator — demo\n" + "-" * 42)
    print(orch.run(notes))


if __name__ == "__main__":
    _demo()
