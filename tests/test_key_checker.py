import os
import pytest
from unittest.mock import patch
from key_checker import load_keys, PROVIDERS, check_all


def test_load_keys_from_env():
    env = {"ANTHROPIC_API_KEY": "sk-ant-test123"}
    with patch.dict(os.environ, env, clear=False):
        keys = load_keys(cli_args={})
    assert keys["anthropic"] == "sk-ant-test123"


def test_load_keys_cli_overrides_env():
    env = {"ANTHROPIC_API_KEY": "from-env"}
    cli = {"anthropic": "from-cli"}
    with patch.dict(os.environ, env, clear=False):
        keys = load_keys(cli_args=cli)
    assert keys["anthropic"] == "from-cli"


def test_load_keys_skips_missing():
    with patch.dict(os.environ, {}, clear=True):
        keys = load_keys(cli_args={})
    assert keys == {}


def test_load_keys_from_dotenv(tmp_path):
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("OPENAI_API_KEY=sk-dotenv-test\n")
    with patch.dict(os.environ, {}, clear=True):
        keys = load_keys(cli_args={}, dotenv_path=str(dotenv_file))
    assert keys["openai"] == "sk-dotenv-test"


def test_cli_overrides_dotenv(tmp_path):
    dotenv_file = tmp_path / ".env"
    dotenv_file.write_text("OPENAI_API_KEY=sk-dotenv\n")
    cli = {"openai": "sk-cli"}
    with patch.dict(os.environ, {}, clear=True):
        keys = load_keys(cli_args=cli, dotenv_path=str(dotenv_file))
    assert keys["openai"] == "sk-cli"


import httpx
import respx
from key_checker import check_anthropic, check_openai, check_gemini, check_nvidia, check_openrouter, CheckResult


@pytest.mark.asyncio
async def test_check_openai_valid():
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "gpt-4o"}]})
        )
        result = await check_openai("sk-test")
    assert result.valid is True
    assert result.status == "Valid"


@pytest.mark.asyncio
async def test_check_openai_invalid():
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )
        result = await check_openai("sk-bad")
    assert result.valid is False
    assert result.status == "Invalid"


@pytest.mark.asyncio
async def test_check_anthropic_valid():
    with respx.mock:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(200, json={
                "content": [{"type": "text", "text": "hi"}],
                "model": "claude-sonnet-4-20250514",
            })
        )
        result = await check_anthropic("sk-ant-test")
    assert result.valid is True
    assert result.status == "Valid"


@pytest.mark.asyncio
async def test_check_anthropic_invalid():
    with respx.mock:
        respx.post("https://api.anthropic.com/v1/messages").mock(
            return_value=httpx.Response(401, json={"error": {"message": "Invalid API key"}})
        )
        result = await check_anthropic("sk-ant-bad")
    assert result.valid is False
    assert result.status == "Invalid"


@pytest.mark.asyncio
async def test_check_gemini_valid():
    with respx.mock:
        respx.get("https://generativelanguage.googleapis.com/v1beta/models").mock(
            return_value=httpx.Response(200, json={"models": [{"name": "models/gemini-2.0-flash"}]})
        )
        result = await check_gemini("AIza-test")
    assert result.valid is True
    assert result.status == "Valid"


@pytest.mark.asyncio
async def test_check_gemini_invalid():
    with respx.mock:
        respx.get("https://generativelanguage.googleapis.com/v1beta/models").mock(
            return_value=httpx.Response(403, json={"error": {"message": "API key not valid"}})
        )
        result = await check_gemini("AIza-bad")
    assert result.valid is False
    assert result.status == "Invalid"


@pytest.mark.asyncio
async def test_check_nvidia_valid():
    with respx.mock:
        respx.get("https://integrate.api.nvidia.com/v1/models").mock(
            return_value=httpx.Response(200, json={"data": [{"id": "meta/llama-3.1-8b-instruct"}]})
        )
        result = await check_nvidia("nvapi-test")
    assert result.valid is True
    assert result.status == "Valid"


@pytest.mark.asyncio
async def test_check_nvidia_invalid():
    with respx.mock:
        respx.get("https://integrate.api.nvidia.com/v1/models").mock(
            return_value=httpx.Response(401, json={"error": "Unauthorized"})
        )
        result = await check_nvidia("nvapi-bad")
    assert result.valid is False
    assert result.status == "Invalid"


@pytest.mark.asyncio
async def test_check_openrouter_valid():
    with respx.mock:
        respx.get("https://openrouter.ai/api/v1/auth/key").mock(
            return_value=httpx.Response(200, json={
                "data": {"label": "test", "limit": 10.0, "usage": 2.5}
            })
        )
        result = await check_openrouter("sk-or-test")
    assert result.valid is True
    assert "$7.50 credits remaining" in result.detail


@pytest.mark.asyncio
async def test_check_openrouter_invalid():
    with respx.mock:
        respx.get("https://openrouter.ai/api/v1/auth/key").mock(
            return_value=httpx.Response(401, json={"error": "Invalid key"})
        )
        result = await check_openrouter("sk-or-bad")
    assert result.valid is False
    assert result.status == "Invalid"


@pytest.mark.asyncio
async def test_check_network_error():
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").mock(side_effect=httpx.ConnectError("Connection refused"))
        result = await check_openai("sk-test")
    assert result.valid is False
    assert result.status == "Unreachable"


@pytest.mark.asyncio
async def test_check_rate_limited():
    with respx.mock:
        respx.get("https://api.openai.com/v1/models").mock(
            return_value=httpx.Response(429, json={"error": {"message": "Rate limited"}})
        )
        result = await check_openai("sk-test")
    assert result.valid is True
    assert result.status == "Valid (rate limited)"


import json
from key_checker import format_json, format_table


def test_format_json():
    results = [
        CheckResult(provider="OpenAI", valid=True, status="Valid", detail="gpt-4o"),
        CheckResult(provider="Gemini", valid=False, status="Invalid", detail="403"),
    ]
    output = format_json(results)
    parsed = json.loads(output)
    assert len(parsed) == 2
    assert parsed[0]["provider"] == "OpenAI"
    assert parsed[0]["valid"] is True
    assert parsed[1]["valid"] is False


def test_format_table():
    results = [
        CheckResult(provider="OpenAI", valid=True, status="Valid", detail="gpt-4o"),
    ]
    output = format_table(results)
    assert "OpenAI" in output
    assert "Valid" in output
    assert "gpt-4o" in output


from key_checker import build_parser


def test_build_parser_provider_args():
    parser = build_parser()
    args = parser.parse_args(["--anthropic", "sk-test", "--json"])
    assert args.anthropic == "sk-test"
    assert args.json is True


def test_build_parser_no_args():
    parser = build_parser()
    args = parser.parse_args([])
    assert args.anthropic is None
    assert args.json is False


@pytest.mark.asyncio
async def test_check_all_dispatches_all_providers():
    async def fake_check(key):
        return CheckResult(provider="fake", valid=True, status="Valid", detail=key)

    with patch.dict("key_checker.CHECKERS", {"openai": fake_check, "gemini": fake_check}):
        results = await check_all({"openai": "sk-test", "gemini": "AIza-test"})
    assert len(results) == 2
    assert all(r.valid for r in results)
    assert {r.detail for r in results} == {"sk-test", "AIza-test"}
