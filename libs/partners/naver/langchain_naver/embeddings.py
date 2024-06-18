from typing import List, Optional, Dict

from langchain_core.pydantic_v1 import Field, SecretStr, root_validator
from langchain_core.embeddings import Embeddings
from langchain_core.utils import convert_to_secret_str, get_from_dict_or_env

DEFAULT_BASE_URL = "https://clovastudio.apigw.ntruss.com/testapp/v1/api-tools/embedding"


class NaverEmbeddings(Embeddings):
    """`NCP ClovaStudio` embedding API.

    following environment variables set or passed in constructor in lower case:
    - ``NCP_CLOVASTUDIO_API_KEY``
    - ``NCP_APIGW_API_KEY``

    Example:
        .. code-block:: python

            from langchain_naver import NaverEmbeddings

            model = NaverEmbeddings()
    """

    model_name: str = Field(default="clir-emb-dolphin", alias="model")

    ncp_clovastudio_api_key: Optional[SecretStr] = Field(default=None, alias="clovastudio_api_key")
    """Automatically inferred from env are `NCP_CLOVASTUDIO_API_KEY` if not provided."""

    ncp_apigw_api_key: Optional[SecretStr] = Field(default=None, alias="apigw_api_key")
    """Automatically inferred from env are `NCP_APIGW_API_KEY` if not provided."""

    ncp_clovastudio_api_base: Optional[str] = Field(default=DEFAULT_BASE_URL, alias="base_url")

    @root_validator(pre=True, allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate api key exists in environment."""
        values["ncp_clovastudio_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_clovastudio_api_key", "NCP_CLOVASTUDIO_API_KEY")
        )
        values["ncp_apigw_api_key"] = convert_to_secret_str(
            get_from_dict_or_env(values, "ncp_apigw_api_key", "NCP_APIGW_API_KEY")
        )
        return values

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embed search docs."""
        raise NotImplementedError

    def embed_query(self, text: str) -> List[float]:
        """Embed query text."""
        raise NotImplementedError

    # only keep aembed_documents and aembed_query if they're implemented!
    # delete them otherwise to use the base class' default
    # implementation, which calls the sync version in an executor
    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        """Asynchronous Embed search docs."""
        raise NotImplementedError

    async def aembed_query(self, text: str) -> List[float]:
        """Asynchronous Embed query text."""
        raise NotImplementedError
