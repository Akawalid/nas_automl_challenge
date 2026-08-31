"""Tests for :class:`tools.datasets.NpyWebDataset` download validation helpers."""

from __future__ import annotations

import pytest

from tools.datasets import NpyWebDataset


def test_validate_zip_bytes_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        NpyWebDataset._validate_zip_bytes(b"", "https://example.com/a.zip")


def test_validate_zip_bytes_rejects_non_zip() -> None:
    with pytest.raises(ValueError, match="PK"):
        NpyWebDataset._validate_zip_bytes(b"<!DOCTYPE html><html>", "https://x/y")


def test_validate_http_response_for_zip_waf_header() -> None:
    class _Resp:
        headers = {"x-amzn-waf-action": "challenge"}
        status_code = 200
        content = b""

    with pytest.raises(RuntimeError, match="WAF"):
        NpyWebDataset._validate_http_response_for_zip(_Resp(), "https://x")  # type: ignore[arg-type]


def test_figshare_article_zip_url_default() -> None:
    from tools.datasets import _figshare_article_zip_url

    assert (
        _figshare_article_zip_url("MultNIST")
        == "https://ndownloader.figshare.com/articles/24574678/versions/1"
    )


def test_figshare_article_zip_url_json_override(monkeypatch: pytest.MonkeyPatch) -> None:
    from tools.datasets import _figshare_article_zip_url

    monkeypatch.setenv(
        "NPY_WEBDATASET_URL_OVERRIDES",
        '{"MultNIST": "https://mirror.example/MultNIST.zip"}',
    )
    assert _figshare_article_zip_url("MultNIST") == "https://mirror.example/MultNIST.zip"
    assert (
        _figshare_article_zip_url("AddNIST")
        == "https://ndownloader.figshare.com/articles/24574354/versions/1"
    )


def test_figshare_article_zip_url_unknown_raises() -> None:
    from tools.datasets import _figshare_article_zip_url

    with pytest.raises(KeyError, match="No Figshare article registered"):
        _figshare_article_zip_url("NotARegisteredDataset")
