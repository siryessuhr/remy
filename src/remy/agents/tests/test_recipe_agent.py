"""Tests for the recipe_agent module."""

import asyncio
import json

import pytest
from langchain_core.prompts import ChatPromptTemplate
from pydantic import ValidationError

from remy.agents.recipe_agent import RECIPE_EXTRACTION_PROMPT, RecipeAgent
from remy.models.recipe import BaseRecipeModel
from remy.settings import settings
from remy.tools.vector_search import vector_similarity_search
from remy.utils import get_embeddings_client, get_llm, parse_structured_response, use_ollama


class MockLLMResponse:
    """Simple mock response class to avoid MagicMock attribute assignment issues.

    When you assign `mock.content = "value"` on a MagicMock, it creates another MagicMock
    instead of storing the actual value. This class properly stores the content as a string.
    """

    def __init__(self, content):
        self.content = content


class TestRecipeAgentFromEnv:
    """Tests for the RecipeAgent.from_env() classmethod."""

    def test_from_env_creates_instance(self, mocker):
        """Test that from_env returns a RecipeAgent instance."""
        mock_llm = mocker.patch("remy.utils.get_llm", autospec=True)
        mock_llm.return_value = mocker.MagicMock()

        agent = RecipeAgent.from_env()

        assert isinstance(agent, RecipeAgent)
        assert agent.llm is not None

    def test_from_env_defaults_to_openai(self, mocker):
        """Test that from_env uses the OpenAI factory unless Ollama is configured."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", None)
        mocker.patch.object(settings, "OLLAMA_MODEL", None)
        mocker.patch.object(settings, "OPENAI_API_KEY", "test-key")
        mock_chat_openai = mocker.patch("remy.utils.ChatOpenAI", autospec=True)
        mock_chat_openai.return_value = mocker.MagicMock()

        RecipeAgent.from_env()

        mock_chat_openai.assert_called_once()

    def test_from_env_uses_ollama_when_configured(self, mocker):
        """Test that from_env switches to Ollama when Ollama env vars are set."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        mocker.patch.object(settings, "OLLAMA_MODEL", "gemma3:4b")
        mock_chat_ollama = mocker.patch("remy.utils.ChatOllama", autospec=True)
        mock_chat_ollama.return_value = mocker.MagicMock()

        RecipeAgent.from_env()

        mock_chat_ollama.assert_called_once_with(
            base_url="http://localhost:11434",
            model="gemma3:4b",
            format="json",
            temperature=0,
        )


class TestRecipeAgentExtractRecipe:
    """Tests for the RecipeAgent.extract_recipe() method."""

    def test_extract_recipe_returns_valid_model(self, mocker):
        """Test that extract_recipe returns a valid BaseRecipeModel."""
        mock_llm = mocker.MagicMock()
        recipe_data = {
            "url": "https://example.com/recipe",
            "title": "Test Recipe",
            # ingredients and instructions MUST be strings, not lists!
            "ingredients": "flour, sugar, eggs",
            "instructions": "Mix and bake",
        }

        mock_chain = mocker.MagicMock()
        mock_invoke = mocker.MagicMock(return_value=MockLLMResponse(json.dumps(recipe_data)))
        # Configure __or__ to return a chain that has our pre-configured invoke
        mock_result = mocker.MagicMock()
        mock_result.invoke = mock_invoke
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch(
            "remy.agents.recipe_agent.ChatPromptTemplate.from_template",
            return_value=mock_chain,
        )
        agent = RecipeAgent(llm=mock_llm)
        # pyrefly: ignore [missing-attribute]
        result = agent.extract_recipe("Some recipe text")

        assert isinstance(result, BaseRecipeModel)
        assert str(result.url) == "https://example.com/recipe"
        assert result.title == "Test Recipe"
        assert result.ingredients == "flour, sugar, eggs"
        assert result.instructions == "Mix and bake"

    def test_extract_recipe_calls_llm_with_prompt(self, mocker):
        """Test that extract_recipe invokes the chain with correct prompt."""
        mock_llm = mocker.MagicMock()
        response_content = json.dumps(
            {
                "url": "",
                "title": "Test",
                "ingredients": "item1",
                "instructions": "step1",
            }
        )

        mock_invoke = mocker.MagicMock(return_value=MockLLMResponse(response_content))
        mock_result = mocker.MagicMock()
        mock_result.invoke = mock_invoke

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch(
            "remy.agents.recipe_agent.ChatPromptTemplate.from_template",
            return_value=mock_chain,
        )
        agent = RecipeAgent(llm=mock_llm)
        test_input = "How to make a cake?"
        # pyrefly: ignore [missing-attribute]
        agent.extract_recipe(test_input)

        # Verify the invoke was called with the text parameter
        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[0][0]
        assert isinstance(call_kwargs, dict)
        assert "text" in call_kwargs
        assert call_kwargs["text"] == test_input

    def test_extract_recipe_handles_whitespace_in_response(self, mocker):
        """Test that extract_recipe strips whitespace from LLM response."""
        mock_llm = mocker.MagicMock()
        recipe_data = {
            "url": "http://example.com/recipe",
            "title": "Recipe",
            "ingredients": "item",
            "instructions": "step",
        }
        raw_response = "\n  \n" + json.dumps(recipe_data) + "\n  \n"

        mock_invoke = mocker.MagicMock(return_value=MockLLMResponse(raw_response))
        mock_result = mocker.MagicMock()
        mock_result.invoke = mock_invoke

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch(
            "remy.agents.recipe_agent.ChatPromptTemplate.from_template",
            return_value=mock_chain,
        )
        agent = RecipeAgent(llm=mock_llm)
        # pyrefly: ignore [missing-attribute]
        result = agent.extract_recipe("test input")

        assert isinstance(result, BaseRecipeModel)
        assert str(result.url) == "http://example.com/recipe"
        assert result.title == "Recipe"

    def test_extract_recipe_raises_on_invalid_json(self, mocker):
        """Test that extract_recipe raises on malformed JSON response."""
        mock_llm = mocker.MagicMock()
        mock_invoke = mocker.MagicMock(return_value=MockLLMResponse("{invalid json content"))
        mock_result = mocker.MagicMock()
        mock_result.invoke = mock_invoke

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch(
            "remy.agents.recipe_agent.ChatPromptTemplate.from_template",
            return_value=mock_chain,
        )
        agent = RecipeAgent(llm=mock_llm)

        with pytest.raises(ValidationError):
            # pyrefly: ignore [missing-attribute]
            agent.extract_recipe("test input")

    def test_generate_labels_accepts_list_response(self, mocker):
        """Tests that list-based JSON label responses are normalized to the expected schema."""
        mock_llm = mocker.MagicMock()
        mock_response = MockLLMResponse('["chicken", "grilled", "high-protein"]')
        mock_result = mocker.MagicMock()
        mock_result.ainvoke = mocker.AsyncMock(return_value=mock_response)

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch("remy.agents.recipe_agent.ChatPromptTemplate.from_template", return_value=mock_chain)
        agent = RecipeAgent(llm=mock_llm)
        state = type(
            "State",
            (),
            {
                "processed_recipe": type(
                    "Recipe",
                    (),
                    {"ingredients": "chicken, yogurt", "instructions": "grill until done"},
                )()
            },
        )()

        result = asyncio.run(agent._generate_labels(state))

        assert result["labels"] == ["chicken", "grilled", "high-protein"]


class TestRecipeAgentIntegration:
    """Integration-style tests for the recipe_agent module."""

    def test_prompt_template_creation(self):
        """Test that ChatPromptTemplate is created correctly."""
        prompt = ChatPromptTemplate.from_template(RECIPE_EXTRACTION_PROMPT)
        assert prompt is not None
        assert hasattr(prompt, "messages")

    def test_base_recipe_model_validation(self):
        """Test that BaseRecipeModel validates correctly."""
        data = {
            "url": "https://example.com/recipe",
            "title": "Valid Recipe",
            "ingredients": "flour, sugar",
            "instructions": "Mix ingredients.",
        }
        model = BaseRecipeModel.model_validate(data)

        assert str(model.url) == "https://example.com/recipe"
        assert model.title == "Valid Recipe"
        assert model.ingredients == "flour, sugar"
        assert model.instructions == "Mix ingredients."

    def test_recipe_agent_instance_with_mocked_llm(self, mocker):
        """Test creating a RecipeAgent instance with a mocked LLM."""
        mock_llm = mocker.MagicMock()
        agent = RecipeAgent(llm=mock_llm)

        assert agent.llm is mock_llm


class TestLLMProviderFactories:
    """Tests for LLM and embedding provider selection."""

    @pytest.mark.asyncio
    async def test_vector_similarity_search_uses_recipe_model_schema(self, mocker):
        """Ensure the SQLAlchemy-backed recipe schema is used for semantic search."""
        recipe = mocker.Mock(
            id=7,
            url="https://example.com/pasta",
            title="Pasta",
            ingredients="pasta, olive oil",
            labels="dinner",
            instructions="Cook pasta",
        )
        recipe.ingred_embedding = [0.1, 0.2, 0.3]

        session = mocker.AsyncMock()
        session.execute = mocker.AsyncMock(return_value=mocker.Mock(all=lambda: [(recipe, 0.25)]))

        session_cm = mocker.MagicMock()
        session_cm.__aenter__ = mocker.AsyncMock(return_value=session)
        session_cm.__aexit__ = mocker.AsyncMock(return_value=None)

        engine = mocker.Mock()
        engine.dispose = mocker.AsyncMock()
        mocker.patch("remy.tools.vector_search.create_engine", return_value=engine)
        mocker.patch("remy.tools.vector_search.create_session_factory", return_value=lambda: session_cm)

        results = await vector_similarity_search([0.1, 0.2, 0.3], top_k=5, min_score=0.0)

        assert len(results) == 1
        assert results[0]["id"] == 7
        assert results[0]["title"] == "Pasta"
        assert results[0]["score"] > 0.0

    def test_parse_structured_response_supports_ollama_json_fallback(self):
        """Test that plain JSON content can be parsed when the model doesn't support structured output."""
        response = MockLLMResponse('{"user_intent": "search_recipe_in_db", "url": ""}')

        result = parse_structured_response(response, dict[str, str])

        assert result["user_intent"] == "search_recipe_in_db"
        assert result["url"] == ""

    def test_use_ollama_is_false_by_default(self, mocker):
        """Test that OpenAI is the default provider when Ollama is unset."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", None)
        mocker.patch.object(settings, "OLLAMA_MODEL", None)

        assert use_ollama() is False

    def test_use_ollama_true_when_ollama_settings_present(self, mocker):
        """Test that the factory prefers Ollama when its settings are configured."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        mocker.patch.object(settings, "OLLAMA_MODEL", "gemma3:4b")

        assert use_ollama() is True

    def test_get_embeddings_client_defaults_to_openai(self, mocker):
        """Test that embeddings default to OpenAI unless Ollama is configured."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", None)
        mocker.patch.object(settings, "OLLAMA_MODEL", None)
        mocker.patch.object(settings, "OPENAI_API_KEY", "test-key")
        mock_openai_embeddings = mocker.patch("remy.utils.OpenAIEmbeddings", autospec=True)
        mock_openai_embeddings.return_value = mocker.MagicMock()

        client = get_embeddings_client()

        assert client is not None
        mock_openai_embeddings.assert_called_once()

    def test_get_llm_returns_ollama_when_configured(self, mocker):
        """Test that the LLM factory picks Ollama when configured."""
        mocker.patch.object(settings, "OLLAMA_BASE_URL", "http://localhost:11434")
        mocker.patch.object(settings, "OLLAMA_MODEL", "gemma3:4b")
        mock_chat_ollama = mocker.patch("remy.utils.ChatOllama", autospec=True)
        mock_chat_ollama.return_value = mocker.MagicMock()

        llm = get_llm()

        assert llm is not None
        mock_chat_ollama.assert_called_once_with(
            base_url="http://localhost:11434",
            model="gemma3:4b",
            format="json",
            temperature=0,
        )


class TestRecipeAgentFromEnvConfig:
    """Additional tests for Ollama configuration."""

    @pytest.mark.asyncio
    async def test_user_intent_prompt_routes_summary_requests_to_search(self, mocker):
        """A summary request should not be treated as a recipe-extraction task."""
        mock_llm = mocker.MagicMock()
        mock_response = MockLLMResponse(json.dumps({"user_intent": "search_recipe_in_db", "url": ""}))
        mock_result = mocker.MagicMock()
        mock_result.ainvoke = mocker.AsyncMock(return_value=mock_response)

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)
        mocker.patch("remy.agents.recipe_agent.ChatPromptTemplate.from_template", return_value=mock_chain)

        agent = RecipeAgent(llm=mock_llm)
        state = agent._understand_user_intent(
            type("State", (), {"user_request": "Summarize a chicken dinner with a quick ingredient list"})()
        )

        intent = await state
        assert intent["user_intent"] == "search_recipe_in_db"

    def test_extract_recipe_passes_text_to_chain(self, mocker):
        """Test that text parameter is correctly passed to LLM chain."""
        mock_llm = mocker.MagicMock()
        response_content = json.dumps(
            {
                "url": "http://test.com",
                "title": "Test",
                "ingredients": "item",
                "instructions": "step",
            }
        )

        mock_invoke = mocker.MagicMock(return_value=MockLLMResponse(response_content))
        mock_result = mocker.MagicMock()
        mock_result.invoke = mock_invoke

        mock_chain = mocker.MagicMock()
        mock_chain.__or__ = mocker.MagicMock(return_value=mock_result)

        mocker.patch(
            "remy.agents.recipe_agent.ChatPromptTemplate.from_template",
            return_value=mock_chain,
        )
        agent = RecipeAgent(llm=mock_llm)
        test_text = "Test recipe input"
        # pyrefly: ignore [missing-attribute]
        agent.extract_recipe(test_text)

        mock_invoke.assert_called_once()
        call_kwargs = mock_invoke.call_args[0][0]
        assert call_kwargs == {"text": test_text}
