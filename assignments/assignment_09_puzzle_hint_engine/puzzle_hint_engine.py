"""
Assignment 9: Puzzle Hint Engine (Difficulty Controls)
"""

import os
from typing import List

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate


class Hint(BaseModel):
    level: int = Field(..., description="1=light nudge, higher=more direct")
    text: str


class HintList(BaseModel):
    hints: List[Hint]


class PuzzleHintEngine:
    def __init__(self):
        self.system_prompt = (
            "You provide puzzle hints in progressive layers.\n"
            "Higher difficulty → more subtle hints.\n"
            "Lower difficulty → more direct hints.\n"
            "Never explicitly reveal the answer unless difficulty is 1."
        )

        # IMPORTANT: JSON braces escaped with double {{ }}
        self.user_prompt = (
            "Puzzle: {puzzle}\n"
            "User attempt: {attempt}\n"
            "Difficulty: {difficulty}\n\n"
            "Return 2–3 hints in the following JSON format:\n"
            "{{\n"
            '  "hints": [\n'
            '    {{ "level": 1, "text": "..." }}\n'
            "  ]\n"
            "}}\n"
        )

        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        ).with_structured_output(HintList)

        self.chain = self.prompt | self.llm

    def get_hints(self, puzzle: str, attempt: str, difficulty: int = 3) -> List[Hint]:
        result: HintList = self.chain.invoke(
            {
                "puzzle": puzzle,
                "attempt": attempt,
                "difficulty": difficulty,
            }
        )
        return result.hints


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    engine = PuzzleHintEngine()

    print("\n🧩 Puzzle Hint Engine — demo\n" + "-" * 40)

    hints = engine.get_hints(
        puzzle="I speak without a mouth and hear without ears.",
        attempt="Is it wind?",
        difficulty=2,
    )

    for h in hints:
        print(f"[{h.level}] {h.text}")


if __name__ == "__main__":
    _demo()
