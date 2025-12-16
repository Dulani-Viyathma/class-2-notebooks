"""
Assignment 7: Minutes & Action Items Batcher

Goal: Convert meeting transcripts into concise minutes and action items,
with support for batch processing many transcripts at once.
"""

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MinutesBatcher:
    """Summarize transcripts into minutes and action items."""

    def __init__(self):
        # Prompt strings
        self.system_prompt = (
            "You produce crisp meeting minutes and clear action items. "
            "Minutes should summarize key discussion points. "
            "Actions should list tasks with an owner and a due date."
        )

        self.user_prompt = (
            "Title: {title}\n\n"
            "Transcript:\n{transcript}\n\n"
            "Return sections exactly as:\n"
            "MINUTES:\n- ...\n\n"
            "ACTIONS:\n- Task (Owner; Due date)"
        )

        # Build ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # Low-temperature LLM for consistency
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.2,
        )

        # Build chain
        self.chain = self.prompt | self.llm | StrOutputParser()

    def summarize_one(self, title: str, transcript: str) -> str:
        """Return minutes+actions for a single transcript."""

        result = self.chain.invoke(
            {
                "title": title,
                "transcript": transcript,
            }
        )
        return result

    def summarize_batch(self, items: List[Dict[str, str]]) -> List[str]:
        """Return minutes+actions for a batch of transcripts."""

        inputs = [
            {
                "title": item["title"],
                "transcript": item["transcript"],
            }
            for item in items
        ]

        results = self.chain.batch(inputs)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    mb = MinutesBatcher()

    print("\n📝 Minutes & Actions — demo\n" + "-" * 40)

    print(
        mb.summarize_one(
            "Sprint Planning",
            "Discussed backlog grooming, two blockers, and deployment window next Tuesday.",
        )
    )

    # Optional batch demo
    batch_items = [
        {
            "title": "Design Review",
            "transcript": "Reviewed new UI mockups, agreed on color changes, action to update designs.",
        },
        {
            "title": "Retrospective",
            "transcript": "Talked about sprint wins, deployment delay, and improving test coverage.",
        },
    ]

    results = mb.summarize_batch(batch_items)

    print("\n🔁 Batch Results\n" + "-" * 40)
    for r in results:
        print(r)
        print("-" * 20)


if __name__ == "__main__":
    _demo()
