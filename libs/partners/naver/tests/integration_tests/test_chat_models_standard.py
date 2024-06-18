"""Standard LangChain interface tests"""

from typing import Type

from langchain_core.language_models import BaseChatModel
from langchain_naver import ChatNaver
from langchain_standard_tests.integration_tests import ChatModelIntegrationTests


class TestNaverStandard(ChatModelIntegrationTests):
    @property
    def chat_model_class(self) -> Type[BaseChatModel]:
        return ChatNaver

    @property
    def chat_model_params(self) -> dict:
        return {"model": "HCX-003"}
