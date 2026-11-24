import pytest

from notebook.video_hosts import kinescope_embed_url, kinescope_url, parse_kinescope_url


@pytest.mark.parametrize(
    "url",
    [
        "https://kinescope.io/abc123",
        "http://kinescope.io/embed/abc123?x=1",
        "https://www.kinescope.io/abc123/",
    ],
)
def test_accepted_forms_return_the_id(url):
    assert parse_kinescope_url(url) == "abc123"


@pytest.mark.parametrize(
    "url",
    [
        "https://youtu.be/abc123",
        "https://kinescope.io/",
        "https://evil.example/kinescope.io/abc123",
        "kinescope.io/abc123",
        "https://kinescope.io/a/b/c",
    ],
)
def test_other_urls_are_rejected(url):
    with pytest.raises(ValueError):
        parse_kinescope_url(url)


def test_canonical_url():
    assert kinescope_url("abc123") == "https://kinescope.io/abc123"


def test_embed_url_uses_t_and_whole_seconds():
    assert (
        kinescope_embed_url("https://kinescope.io/abc123", 125.9)
        == "https://kinescope.io/embed/abc123?t=125"
    )
    assert kinescope_embed_url("https://youtu.be/x", 5) is None
