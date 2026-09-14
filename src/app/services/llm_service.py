"""LLM service: turn a question plus retrieved chunks into an answer."""

from langchain_core.documents import Document
from langchain_openai import ChatOpenAI

from app.config import Config
from app.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT = """Answer the question using only the context below.
If the context does not contain the answer, say so plainly - do not invent one.

Context:
{context}

Question: {question}

Answer:"""


class LLMService:
    """Holds one LLM client, reused for every question."""

    def __init__(self, model: str | None = None, temperature: float | None = None):
        self.llm = ChatOpenAI(
            model=model or Config.CHAT_MODEL,
            temperature=Config.TEMPERATURE if temperature is None else temperature,
            api_key=Config.OPENAI_API_KEY,
        )

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
