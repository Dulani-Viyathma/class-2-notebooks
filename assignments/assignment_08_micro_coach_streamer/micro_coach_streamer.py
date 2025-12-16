"""
Assignment 8: Micro-Coach (On-Demand Streaming)

Goal: Provide a short plan non-streamed, and when `stream=True` deliver
encouraging guidance token-by-token via a callback.
"""

import os
from typing import Any

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.callbacks import BaseCallbackHandler


class PrintTokens(BaseCallbackHandler):
    """Print tokens to stdout as they arrive."""

    def on_llm_new_token(self, token: str, **kwargs: Any) -> None:
        print(token, end="", flush=True)


class MicroCoach:
    def __init__(self):
        # Prompt strings
        self.system_prompt = (
            "You are a supportive micro-coach. Keep plans realistic, motivating, "
            "and very brief."
        )
        self.user_prompt = (
            "Goal: {goal}\n"
            "Time available: {time_available}\n"
            "Return a simple 3-step plan."
        )

        # Build prompts
        self.plain_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", self.system_prompt),
                ("user", self.user_prompt),
            ]
        )

        self.stream_prompt = self.plain_prompt  # same prompt, different execution

        # Non-streaming LLM
        self.llm_plain = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.3,
        )

        # Streaming LLM with callback
        self.llm_streaming = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.5,
            streaming=True,
            callbacks=[PrintTokens()],
        )

        # Chains
        self.plain_chain = self.plain_prompt | self.llm_plain | StrOutputParser()
        self.stream_chain = self.stream_prompt | self.llm_streaming | StrOutputParser()

    def coach(self, goal: str, time_available: str, stream: bool = False) -> str:
        """Return guidance using streaming or non-streaming path."""

        inputs = {
            "goal": goal,
            "time_available": time_available,
        }

        if stream:
            # Stream tokens live to console
            _ = self.stream_chain.invoke(inputs)
            return ""

        # Non-streamed compact response
        result = self.plain_chain.invoke(inputs)
        return result


def _demo():
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Set OPENAI_API_KEY before running.")

    coach = MicroCoach()

    print("\n🏃 Micro-Coach — demo\n" + "-" * 40)

    # Non-streaming example
    print(coach.coach("resume drafting", "25 minutes", stream=False))

    print("\nStreaming example:")
    coach.coach("push-ups habit", "10 minutes", stream=True)
    print()


if __name__ == "__main__":
    _demo()
