"""Naver chat models."""
import os
from typing import Any, AsyncIterator, Iterator, List, Optional, Dict

import openai
from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.callbacks import (
    AsyncCallbackManagerForLLMRun,
    CallbackManagerForLLMRun,
)
# from langchain_core.language_models.chat_models import BaseChatModel, LangSmithParams
from langchain_core.language_models.chat_models import LangSmithParams
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env
from langchain_openai.chat_models.base import BaseChatOpenAI

DEFAULT_BASE_URL = "https://clovastudio.stream.ntruss.com/testapp/v1/chat-completions"


class ChatNaver(BaseChatOpenAI):
    """`NCP ClovaStudio` Chat Completion API.

    following environment variables set or passed in constructor in lower case:
    - ``NCP_CLOVASTUDIO_API_KEY``
    - ``NCP_APIGW_API_KEY``
    
    Example:
        .. code-block:: python

            from langchain_core.messages import HumanMessage

            from langchain_naver import ChatNaver

            model = ChatNaver()
            model.invoke([HumanMessage(content="Come up with 10 names for a song about parrots.")])
    """  # noqa: E501

    model_name: str = Field(default="HCX-003", alias="model")

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="clovastudio_api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    ncp_clovastudio_api_base: Optional[str] = Field(default=DEFAULT_BASE_URL, alias="base_url")

    class Config:
        """Configuration for this pydantic object."""

        allow_population_by_field_name = True

    @property
    def lc_secrets(self) -> Dict[str, str]:
        return {
            "ncp_clovastudio_api_key": "NCP_CLOVASTUDIO_API_KEY",
            "ncp_apigw_api_key": "NCP_APIGW_API_KEY",
        }

    @property
    def _llm_type(self) -> str:
        """Return type of chat model."""
        return "chat-naver"

    def _get_ls_params(
        self, stop: Optional[List[str]] = None, **kwargs: Any
    ) -> LangSmithParams:
        """Get the parameters used to invoke the model."""
        params = super()._get_ls_params(stop=stop, **kwargs)
        params["ls_provider"] = "naver"
        return params

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        values["ncp_clovastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_clovastudio_api_key", "NCP_CLOVASTUDIO_API_KEY")
        )
        values["ncp_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_apigw_api_key", "NCP_APIGW_API_KEY")
        )
        values["ncp_clovastudio_api_base"] = values["ncp_clovastudio_api_base"] or os.getenv(
            "NCP_CLOVASTUDIO_API_BASE"
        )

        client_params = {
            "api_key": (
                values["ncp_clovastudio_api_key"].get_secret_value()
                if values["ncp_clovastudio_api_key"]
                else None
            ),
            "base_url": values["ncp_clovastudio_api_base"],
            # "timeout": values["request_timeout"],
            # "max_retries": values["max_retries"],
            # "default_headers": values["default_headers"],
            # "default_query": values["default_query"],
        }

        if not values.get("client"):
            sync_specific = {"http_client": values["http_client"]}
            values["client"] = openai.OpenAI(
                **client_params, **sync_specific
            ).chat.completions
        if not values.get("async_client"):
            async_specific = {"http_client": values["http_async_client"]}
            values["async_client"] = openai.AsyncOpenAI(
                **client_params, **async_specific
            ).chat.completions
        return values

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
