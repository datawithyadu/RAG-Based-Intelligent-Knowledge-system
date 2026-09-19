"""LLM service: turn a question plus retrieved chunks into an answer."""

import json
import re

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say so plainly - do not invent one.
If the question has several parts, answer each part.

Context:
{context}

Question: {question}

Answer:"""

DECOMPOSE_PROMPT = """Split the user's message into the separate questions it contains.

Rules:
- One entry per distinct question. Do not merge two questions into one.
- Rewrite each so it stands alone, replacing pronouns with what they refer to.
- If the message contains only one question, return it unchanged as a single entry.
- Do not answer anything. Do not add questions the user did not ask.

Return a JSON array of strings and nothing else.

Message: {question}
"""


class LLMService:
    """Holds one LLM client, reused for every question."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.llm = ChatOpenAI(
            model=model or Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE if temperature is None else temperature,
            api_key=Config.OPENAI_API_KEY,
        )

    def decompose(self, question: str) -> list[str]:
        """Split a multi-part message into standalone sub-questions.

        Costs one extra LLM call per request. Worth it when a message asks
        two things: a single embedding of "what is X and what is Y" sits
        between both topics and retrieves well for neither.

        Every failure path returns [question], so the system falls back to
        its previous single-query behaviour rather than breaking.
        """
        try:
            reply = self.llm.invoke(DECOMPOSE_PROMPT.format(question=question)).content
        except Exception as e:
            logger.error("Decompose call failed, using the question as-is: %s", e)
            return [question]

        cleaned = re.sub(r"^```(?:json)?|```$", "", reply.strip(), flags=re.MULTILINE).strip()

        try:
            parts = json.loads(cleaned)
        except json.JSONDecodeError:
            logger.warning("Decompose reply was not JSON, using the question as-is")
            return [question]

        if not isinstance(parts, list):
            return [question]

        subs = [str(p).strip() for p in parts if str(p).strip()]

        if not subs:
            return [question]

        if len(subs) > 1:
            logger.info("Split into %s sub-questions: %s", len(subs), subs)

        return subs

    def answer(self, question: str, chunks: list[Document]) -> str:
        """Build a prompt from the retrieved chunks and ask the model."""
        if not chunks:
            logger.warning("No chunks retrieved for: %s", question)
            return "I could not find anything relevant in the documents."

        context = "\n\n---\n\n".join(
            f"[{c.metadata.get('source', 'unknown')}]\n{c.page_content}" for c in chunks
        )
        prompt = PROMPT.format(context=context, question=question)

        logger.info("Asking %s with %s chunks (%s chars of context)",
                    self.llm.model_name, len(chunks), len(context))

        return self.llm.invoke(prompt).content
