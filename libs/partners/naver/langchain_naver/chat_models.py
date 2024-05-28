"""Naver chat models."""

from typing import Any, AsyncIterator, Iterator, List, Optional

from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult


class ChatNaver(BaseChatModel):
    """ChatNaver chat model.

    Example:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            from langchain_naver import ChatNaver

            model = ChatNaver()
            model.invoke([HumanMessage(content="Come up with 10 names for a song about parrots.")])
    """  # noqa: E501

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-naver"

    # TODO: This method must be implemented to generate chat responses.
    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError

    # TODO: Implement if ChatNaver supports streaming. Otherwise delete method.
    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        raise NotImplementedError

    # TODO: Implement if ChatNaver supports async streaming. Otherwise delete
    # method.
    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        raise NotImplementedError

    # TODO: Implement if ChatNaver supports async generation. Otherwise delete
    # method.
    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        raise NotImplementedError
