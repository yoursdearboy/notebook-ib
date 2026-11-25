from pathlib import Path

import pytest

from notebook.transcripts import Cue, chunk_cues, format_time, parse_vtt


def write_vtt(path: Path, body: str) -> Path:
    path.write_text("WEBVTT\n\n" + body, encoding="utf-8")
    return path


def cue(index: int, start: float, text: str = "aaaaaaaaaa") -> Cue:
    return Cue(index=index, start=start, end=start + 5, text=text)


def test_parse_vtt_strips_markup_and_entities(tmp_path):
    path = write_vtt(
        tmp_path / "a.vtt",
        "00:00:01.000 --> 00:00:04.000\n<v Anna>Hello &amp; welcome.</v>\n\n"
        "00:00:04.000 --> 00:00:09.500\nToday: mitosis.\n",
    )

    cues = parse_vtt(path)

    assert [c.text for c in cues] == ["Hello & welcome.", "Today: mitosis."]
    assert (cues[0].start, cues[0].end, cues[0].index) == (1.0, 4.0, 0)
    assert (cues[1].start, cues[1].end, cues[1].index) == (4.0, 9.5, 1)


def test_parse_vtt_accepts_short_timestamps_and_skips_empty_cues(tmp_path):
    path = write_vtt(
        tmp_path / "a.vtt",
        "01:02.500 --> 01:05.000\nfirst\n\n"
        "01:05.000 --> 01:06.000\n<b></b>\n\n"
        "01:06.000 --> 01:07.000\nsecond\nline\n",
    )

    cues = parse_vtt(path)

    assert [(c.index, c.start, c.end, c.text) for c in cues] == [
        (0, 62.5, 65.0, "first"),
        (1, 66.0, 67.0, "second line"),
    ]


def test_parse_vtt_without_usable_cues_is_rejected(tmp_path):
    path = write_vtt(tmp_path / "a.vtt", "00:00:01.000 --> 00:00:02.000\n<b></b>\n")

    with pytest.raises(ValueError):
        parse_vtt(path)


def test_format_time():
    assert format_time(5) == "0:05"
    assert format_time(760.9) == "12:40"
    assert format_time(3760) == "1:02:40"


def test_chunks_are_cue_aligned_with_cue_overlap():
    cues = [
        cue(0, 0, "aaaaaaaaaa"),
        cue(1, 10, "bbbbbbbbbb"),
        cue(2, 20, "cccccccccc"),
        cue(3, 30, "dddddddddd"),
    ]

    chunks = chunk_cues(cues, chunk_size=25, chunk_overlap=10)

    assert [tuple(c.index for c in ch.cues) for ch in chunks] == [
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert chunks[0].text == "aaaaaaaaaa bbbbbbbbbb"


def test_cue_longer_than_chunk_size_is_its_own_chunk():
    cues = [cue(0, 0), cue(1, 10, "x" * 50), cue(2, 20)]

    chunks = chunk_cues(cues, chunk_size=25, chunk_overlap=0)

    assert any(len(ch.cues) == 1 and len(ch.text) == 50 for ch in chunks)
    assert {c.index for ch in chunks for c in ch.cues} == {0, 1, 2}


def test_lead_in_is_the_cues_in_the_60_seconds_before_the_chunk():
    cues = [cue(i, start) for i, start in enumerate([0, 30, 60, 90, 120])]

    chunks = chunk_cues(cues, chunk_size=10, chunk_overlap=0)

    assert [tuple(c.index for c in ch.lead_in) for ch in chunks] == [
        (),
        (0,),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
