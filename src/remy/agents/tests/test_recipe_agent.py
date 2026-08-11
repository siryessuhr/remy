"""Tests for the recipe_agent module."""

import json

import pytest
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import ChatOllama
from pydantic import ValidationError

from remy.agents.recipe_agent import RECIPE_EXTRACTION_PROMPT, RecipeAgent
from remy.models.recipe import BaseRecipeModel
from remy.settings import settings


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
        mock_chat_ollama = mocker.patch("remy.agents.recipe_agent.ChatOllama", autospec=True)
        mock_chat_ollama.return_value = mocker.MagicMock()

        agent = RecipeAgent.from_env()

        assert isinstance(agent, RecipeAgent)
        assert agent.llm is not None

    def test_from_env_configures_correct_base_url(self, mocker):
        """Test that from_env configures Ollama with the correct base URL."""
        expected_base_url = "http://localhost:11434"

        mock_chat_ollama = mocker.patch("remy.agents.recipe_agent.ChatOllama", autospec=True)
        mock_chat_ollama.return_value = mocker.MagicMock()

        RecipeAgent.from_env()

        mock_chat_ollama.assert_called_once_with(
            base_url=expected_base_url,
            model=settings.OLLAMA_MODEL,
            format="json",
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
        mock_llm = mocker.MagicMock(spec=ChatOllama)
        agent = RecipeAgent(llm=mock_llm)

        assert agent.llm is mock_llm


class TestRecipeAgentFromEnvConfig:
    """Additional tests for Ollama configuration."""

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
