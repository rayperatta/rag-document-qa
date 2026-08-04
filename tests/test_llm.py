"""Tests for LLM generator — metadata and usage tracking."""
from rag.llm import LLMGenerator


class TestLLMGeneratorUsage:
    def test_last_usage_init_empty(self):
        llm = LLMGenerator(api_key="sk-test")
        assert llm.last_usage == {}

    def test_last_usage_init_empty_disabled(self):
        llm = LLMGenerator(api_key="")
        assert llm.last_usage == {}
