"""
Assignment 5: Flashcard Maker — Structured Outputs with Pydantic

Focus: Use `with_structured_output` to coerce JSON into a Pydantic model.
"""

import os
from typing import List
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI


class Flashcard(BaseModel):
    """Structured flashcard model."""

    term: str = Field(..., description="Short technical term")
    definition: str = Field(..., description="One-sentence machine learning definition")


class FlashcardMaker:
    def __init__(self):
        # Create LLM
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )

        # Wrap LLM with structured output
        self.structured = self.llm.with_structured_output(Flashcard)

    def make_cards(self, topics: List[str]) -> List[Flashcard]:
        """Generate one flashcard per topic."""

        cards: List[Flashcard] = []

        for topic in topics:
            card = self.structured.invoke(
                f"Create a beginner-friendly MACHINE LEARNING flashcard about '{topic}'. "
                "Return a short technical term and a one-sentence definition."
            )
            cards.append(card)

        return cards


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    maker = FlashcardMaker()
    topics = ["positional encoding", "dropout", "precision vs recall"]

    print("\n🧠 Flashcard Maker — demo\n" + "-" * 40)
    for c in maker.make_cards(topics):
        print(f"• {c.term}: {c.definition}")


if __name__ == "__main__":
    _demo()
