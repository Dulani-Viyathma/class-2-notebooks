"""
Assignment 6: Reply Macro Composer — Runtime Configs
"""

import os
from typing import List, Dict

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser


class MacroComposer:
    """Compose reply macros with configurable tone and length."""

    def __init__(self):
        # Prompt strings
        self.system_prompt = (
            "You craft helpful, concise support macros that sound friendly, "
            "polite, and professional. Avoid unnecessary details."
        )

        self.user_prompt = (
            "Customer message:\n{message}\n\n"
            "Context:\n{context}\n\n"
            "Style hint: {style_hint}\n\n"
            "Return a ready-to-send macro with a greeting and a sign-off."
        )

        # Build ChatPromptTemplate
        self.prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        # Base LLM (low temperature for consistency)
        self.llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )

    def _build_chain(self, style_hint: str):
        """
        Small helper to build a chain with runtime configuration.
        """
        tuned_llm = self.llm.bind(
            temperature=0.2 if style_hint == "neutral" else 0.4,
            max_tokens=150,
        )

        return self.prompt | tuned_llm | StrOutputParser()

    def compose_macro(
        self, message: str, context: str, style_hint: str = "neutral"
    ) -> str:
        """Return a polished macro."""

        chain = self._build_chain(style_hint)

        result = chain.invoke(
            {
                "message": message,
                "context": context,
                "style_hint": style_hint,
            }
        )

        return result

    def compose_bulk(
        self, items: List[Dict[str, str]], style_hint: str = "neutral"
    ) -> List[str]:
        """Batch-compose macros for many items."""

        chain = self._build_chain(style_hint)

        inputs = [
            {
                "message": item["message"],
                "context": item["context"],
                "style_hint": style_hint,
            }
            for item in items
        ]

        results = chain.batch(inputs)
        return results


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    mc = MacroComposer()

    print("\n✉️ Macro Composer — demo\n" + "-" * 40)

    print(
        mc.compose_macro(
            "My package arrived damaged. What can I do?",
            context="Order #123, policy: refund or replacement within 30 days.",
            style_hint="warm",
        )
    )


if __name__ == "__main__":
    _demo()
