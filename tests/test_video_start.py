import pytest
from langchain_core.documents import Document

from notebook.video_start import VideoStart, evidence_text, select_starts


def video_doc(cue_start=5, start=100.0, lead_in=None, text="cited words") -> Document:
    return Document(
        page_content=text,
        metadata={
            "source": "lec.vtt",
            "video_file": "lec.mp4",
            "start": start,
            "cue_start": cue_start,
            "lead_in": [{"cue": 3, "t": 70.0, "text": "earlier words"}]
            if lead_in is None
            else lead_in,
        },
    )


def pdf_doc() -> Document:
    return Document(page_content="pdf text", metadata={"source": "a.pdf", "page": 1})


def test_valid_cue_in_lead_in_is_used():
    assert select_starts([video_doc()], [VideoStart(citation=1, cue=3)]) == {1: 70.0}


def test_the_chunks_own_first_cue_is_valid():
    assert select_starts([video_doc()], [VideoStart(citation=1, cue=5)]) == {1: 100.0}


@pytest.mark.parametrize(
    "start",
    [
        VideoStart(citation=2, cue=3),  # citation is a PDF chunk
        VideoStart(citation=9, cue=3),  # unknown citation
        VideoStart(citation=0, cue=3),  # citations are 1-based
        VideoStart(citation=1, cue=6),  # after the cited passage
        VideoStart(citation=1, cue=0),  # outside the lookback window
    ],
)
def test_invalid_starts_are_ignored(start):
    assert select_starts([video_doc(), pdf_doc()], [start]) == {}


def test_first_valid_start_for_a_citation_wins():
    starts = [
        VideoStart(citation=1, cue=99),
        VideoStart(citation=1, cue=3),
        VideoStart(citation=1, cue=5),
    ]

    assert select_starts([video_doc()], starts) == {1: 70.0}


def test_evidence_shows_preceding_cues_then_the_labelled_first_cue():
    text = evidence_text(video_doc())

    assert text == (
        "Preceding transcript:\n"
        "c3 [1:10] earlier words\n"
        "Evidence text:\n"
        "c5 [1:40] cited words"
    )


def test_chunk_at_video_start_offers_only_its_own_cue():
    text = evidence_text(video_doc(cue_start=0, start=0.0, lead_in=[]))

    assert "Preceding transcript" not in text
    assert text == "Evidence text:\nc0 [0:00] cited words"


def test_non_video_evidence_is_the_chunk_text():
    assert evidence_text(pdf_doc()) == "Evidence text:\npdf text"
