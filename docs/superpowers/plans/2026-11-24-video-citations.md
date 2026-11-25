# Video Indexing and Citations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Index WebVTT transcripts of videos and cite them, so a citation opens the video in Open WebUI's sidebar at a start time the model picks.

**Architecture:** A `.vtt` transcript is the indexed source. Cue-aligned chunks carry their time range and the 60 seconds of preceding cues. At answer time a structured-output model call names the cue to start from, validated against that stored list, with the chunk's first cue as the fallback. Local videos are served by signed links and a new `/citation-assets/view-video` page. Kinescope videos are embedded directly with `?t=`.

**Tech Stack:** Python 3.14, LangChain / LangGraph, FastAPI/Starlette 0.52, `webvtt-py`, pytest, Open WebUI.

**Spec:** `docs/superpowers/specs/2026-11-24-video-citations-design.md`

## Global Constraints

- Transcript format is WebVTT (`.vtt`), parsed with `webvtt-py`; no hand-written parser.
- Local video extensions: `.mp4` and `.webm`, a sibling of the transcript with the same stem.
- Third-party host: Kinescope only. Accepted `--video-url` forms: `kinescope.io/VIDEO_ID` and `kinescope.io/embed/VIDEO_ID`. Canonical stored form: `https://kinescope.io/VIDEO_ID`.
- Kinescope embed URL: `https://kinescope.io/embed/VIDEO_ID?t=SECONDS`. Use `t`, never `seek` (`seek` cuts the video before the start).
- The lookback window is a fixed 60 seconds, recorded in the split signature. No new settings and no `VIDEO_LEAD_IN_SECONDS`.
- Citation label is `video 12:40`, or `1:02:40` past an hour. It is the time of the cited passage; the playback start is a separate `start` field.
- Fallback playback start is the cited chunk's first cue.
- The PDF viewer moves from `/citation-assets/view` to `/citation-assets/view-pdf`. No alias for the old route.
- When no retrieved chunk is a video, the plain answer path is unchanged.
- Signatures cover the file path only; start time and title are not signed.

## Review Focus

1. A cue longer than `chunk_size` must become its own chunk, not be split or dropped (Task 2).
2. Cue text with WebVTT markup or HTML entities (`<v Anna>…</v>`, `&amp;`) must be stored as plain text (Task 2).
3. A chunk at the very start of the video, with no preceding cues, must still get a start (Task 6).
4. A model start that names a PDF chunk, an unknown citation, a cue after the cited passage, or a duplicate citation must be ignored and never fail the answer (Task 6).
5. A cited video whose local file was deleted or moved must produce a citation with no `embed_url`, not an error (Task 7).

---

### Task 1: Check the Kinescope embed in Open WebUI (manual, throwaway)

This is the spec's first "verify early" item. Nothing here is committed.

**Files:**
- Modify (temporarily, reverted at the end): `notebook/api.py:86`

- [ ] **Step 1: Ask the user for one Kinescope video URL** they can embed. Do not continue without it.

- [ ] **Step 2: Temporarily force the embed.** In `_source_event`, set `source["embed_url"] = "https://kinescope.io/embed/<VIDEO_ID>?t=30"` for every source, ignoring the PDF logic.

- [ ] **Step 3: Run the app** (README Setup: `uv run bin/frontend .open-webui/frontend`, `uv run bin/serve`), ask a question that returns a citation, and click it.
Expected: the Kinescope player appears in the right sidebar and starts near 0:30. Click the same citation again: the player reloads at 0:30.

- [ ] **Step 4: Record the outcome** in the spec's "Verify early" list: items 1, 2 and 5 pass or fail, with what the browser console shows on failure. Items 3 and 4 are settled by existing tests: `test_range_request_returns_partial_content` shows range requests work, and `bin/frontend` already gives each click a unique `source_id`.

- [ ] **Step 5: Revert and commit the spec note.**

```bash
git checkout notebook/api.py
git add docs/superpowers/specs/2026-11-24-video-citations-design.md
git commit -m "docs: record Kinescope embed check"
```

If the embed is blocked, stop and take it back to the user; the spec's Kinescope decision needs revisiting.

---

### Task 2: Transcript parsing and cue-aligned chunking

**Files:**
- Create: `notebook/transcripts.py`
- Create: `tests/test_transcripts.py`
- Modify: `pyproject.toml` (add `webvtt-py`), `uv.lock`

**Interfaces:**
- Produces:
  - `LEAD_IN_SECONDS: int = 60`
  - `Cue(index: int, start: float, end: float, text: str)`, a frozen dataclass; `index` is the 0-based position among kept cues.
  - `CueChunk(text: str, cues: tuple[Cue, ...], lead_in: tuple[Cue, ...])`, a frozen dataclass.
  - `parse_vtt(path: Path) -> list[Cue]`
  - `chunk_cues(cues: list[Cue], chunk_size: int, chunk_overlap: int) -> list[CueChunk]`
  - `format_time(seconds: float) -> str`

- [ ] **Step 1: Add the dependency.** `uv add webvtt-py`. Confirm `uv run python -c "import webvtt"` works on Python 3.14. If it does not install, stop and report.

- [ ] **Step 2: Write the failing tests** in `tests/test_transcripts.py`:

```python
def test_parse_vtt_strips_markup_and_entities(tmp_path):
    # file: "00:00:01.000 --> 00:00:04.000\n<v Anna>Hello &amp; welcome.</v>"
    assert [c.text for c in cues] == ["Hello & welcome.", "Today: mitosis."]
    assert (cues[0].start, cues[0].end, cues[0].index) == (1.0, 4.0, 0)

def test_parse_vtt_accepts_short_timestamps_and_skips_empty_cues(tmp_path):
    # "01:02.500 --> 01:05.000" (no hours) parses as 62.5 -> 65.0; a cue with only a tag is dropped
    # and the following cue gets index 1, not 2.

def test_format_time():
    assert format_time(5) == "0:05"
    assert format_time(760.9) == "12:40"
    assert format_time(3760) == "1:02:40"

def test_chunks_are_cue_aligned_with_cue_overlap():
    # four 10-character cues, chunk_size=25, chunk_overlap=10
    assert [tuple(c.index for c in ch.cues) for ch in chunks] == [(0, 1), (1, 2), (2, 3)]
    assert chunks[0].text == "aaaaaaaaaa bbbbbbbbbb"

def test_cue_longer_than_chunk_size_is_its_own_chunk():
    # one 50-character cue between two short ones, chunk_size=25
    assert any(len(ch.cues) == 1 and len(ch.text) == 50 for ch in chunks)

def test_lead_in_is_the_cues_in_the_60_seconds_before_the_chunk():
    # cues start at 0, 30, 60, 90, 120; chunk_size=10, chunk_overlap=0 (one cue per chunk)
    assert [tuple(c.index for c in ch.lead_in) for ch in chunks] == [(), (0,), (0, 1), (1, 2), (2, 3)]
```

- [ ] **Step 3: Run them to verify they fail.** `uv run pytest tests/test_transcripts.py -v`. Expected: FAIL, `notebook.transcripts` not found.

- [ ] **Step 4: Implement the module.**
  - `parse_vtt` uses `webvtt.read`, takes the caption text without tags, applies `html.unescape`, collapses whitespace, skips empty cues, and raises `ValueError` if no cue is left.
  - `chunk_cues` accumulates cues until the space-joined text would exceed `chunk_size`; a cue alone over the limit is its own chunk. The next chunk starts at the earliest cue whose trailing run fits in `chunk_overlap`, but always after the previous chunk's first cue. It stops once the last cue is included.
  - `lead_in` is the cues before the chunk's first cue whose `start >= first.start - LEAD_IN_SECONDS`.

- [ ] **Step 5: Run the tests to verify they pass.** `uv run pytest tests/test_transcripts.py -v`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add notebook/transcripts.py tests/test_transcripts.py pyproject.toml uv.lock
git commit -m "feat: parse and chunk WebVTT transcripts"
```

---

### Task 3: Kinescope URL helpers

**Files:**
- Create: `notebook/video_hosts.py`
- Create: `tests/test_video_hosts.py`

**Interfaces:**
- Produces:
  - `parse_kinescope_url(url: str) -> str`: the video ID; raises `ValueError` for any other URL.
  - `kinescope_url(video_id: str) -> str`: `https://kinescope.io/{video_id}`.
  - `kinescope_embed_url(video_url: str, start: float) -> str | None`: `https://kinescope.io/embed/{id}?t={int(start)}`, or `None` if `video_url` is not Kinescope.

- [ ] **Step 1: Write the failing tests:**

```python
@pytest.mark.parametrize("url", ["https://kinescope.io/abc123", "http://kinescope.io/embed/abc123?x=1", "https://www.kinescope.io/abc123/"])
def test_accepted_forms_return_the_id(url):
    assert parse_kinescope_url(url) == "abc123"

@pytest.mark.parametrize("url", ["https://youtu.be/abc123", "https://kinescope.io/", "https://evil.example/kinescope.io/abc123", "kinescope.io/abc123", "https://kinescope.io/a/b/c"])
def test_other_urls_are_rejected(url):
    with pytest.raises(ValueError): parse_kinescope_url(url)

def test_embed_url_uses_t_and_whole_seconds():
    assert kinescope_embed_url("https://kinescope.io/abc123", 125.9) == "https://kinescope.io/embed/abc123?t=125"
    assert kinescope_embed_url("https://youtu.be/x", 5) is None
```

- [ ] **Step 2: Run to verify they fail.** `uv run pytest tests/test_video_hosts.py -v`. Expected: FAIL, module not found.

- [ ] **Step 3: Implement** the three functions with `urllib.parse.urlsplit`: scheme `http`/`https`, hostname `kinescope.io` or `www.kinescope.io`, path `/ID` or `/embed/ID`, ID matching `[A-Za-z0-9_-]+`.

- [ ] **Step 4: Run to verify they pass**, then commit.

```bash
git add notebook/video_hosts.py tests/test_video_hosts.py
git commit -m "feat: add Kinescope URL helpers"
```

---

### Task 4: Index transcripts and pair videos

**Files:**
- Modify: `notebook/documents.py` (supported suffixes at lines 28 and 36, `index_path`, `_split_signature`, `prepare_chunks`)
- Modify: `notebook/cli.py` (`_create_parser`, `main`)
- Test: `tests/test_documents.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `parse_vtt`, `chunk_cues`, `CueChunk`, `LEAD_IN_SECONDS` (Task 2); `parse_kinescope_url`, `kinescope_url` (Task 3).
- Produces:
  - `prepare_chunks(path: Path, config: Config, title: str | None = None, video_url: str | None = None) -> list[Document]`
  - `index_path(path, config, store, embeddings, title=None, video_url=None) -> int`
  - Chunk metadata for a transcript, besides `source`, `source_hash`, `split_signature`, `chunk_index` and optional `title`:
    - `start: float`, `end: float`, `cue_start: int`, `cue_end: int`
    - `lead_in: list[{"cue": int, "t": float, "text": str}]`
    - `video_file: str` (stored with `stored_source`) or `video_url: str` (canonical form)
  - CLI: `--video-url URL` on `index`.

- [ ] **Step 1: Write the failing tests** in `tests/test_documents.py`:

```python
def test_vtt_with_sibling_video_stores_times_cues_lead_in_and_video_file(tmp_path):
    chunks = prepare_chunks(vtt, config(CHUNK_SIZE="30", CHUNK_OVERLAP="0"))
    m = chunks[1].metadata
    assert (m["start"], m["cue_start"]) == (30.0, 1)
    assert m["lead_in"] == [{"cue": 0, "t": 0.0, "text": "first cue."}]
    assert m["video_file"] == "lec.mp4" and "video_url" not in m

def test_vtt_with_video_url_stores_the_canonical_url(tmp_path):
    chunks = prepare_chunks(vtt, config(), video_url="https://kinescope.io/embed/abc123")
    assert chunks[0].metadata["video_url"] == "https://kinescope.io/abc123"

@pytest.mark.parametrize("case", ["no_video", "both", "not_kinescope", "ambiguous_siblings", "video_url_on_pdf", "no_cues"])
def test_invalid_video_pairing_is_rejected(tmp_path, case):
    with pytest.raises(ValueError): ...

def test_split_signature_differs_for_transcripts(tmp_path):
    assert vtt_chunks[0].metadata["split_signature"] != txt_chunks[0].metadata["split_signature"]
```

In `tests/test_cli.py`, add:
- `test_video_url_with_a_directory_fails_before_indexing`: the exit status is 1 and `store.replace_source` is not called.
- `test_video_url_is_passed_to_index_path`.

Use the existing `FakeStore` and `FakeEmbeddings` there. The "relative to assets root" test pattern is in `test_source_is_stored_relative_to_assets_root`.

- [ ] **Step 2: Run them to verify they fail.** `uv run pytest tests/test_documents.py tests/test_cli.py -v`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement.**
  - Replace both literal suffix sets with one `SUPPORTED_SUFFIXES` that adds `.vtt`.
  - `find_sibling_video(transcript: Path) -> Path | None` checks `.mp4` and `.webm`, case-insensitively. It raises `ValueError` if both exist.
  - `prepare_chunks` handles `.vtt`: it raises `ValueError` for each invalid pairing in the test above, including `video_url` on a non-`.vtt` file. It builds `Document`s with the metadata above and the same chunk id scheme as other sources.
  - `_split_signature` takes a `kind: str | None` instead of `pdf: bool` and adds `"transcript": "webvtt-v1", "lead_in_seconds": LEAD_IN_SECONDS` for transcripts. The PDF signature value must not change.
  - `index_path` fails with `logger.error` and returns 1 when `video_url` is given and `len(files) != 1`.
  - CLI: `--video-url` option; update the `path` help text to name VTT.

- [ ] **Step 4: Run to verify they pass.** `uv run pytest tests/test_documents.py tests/test_cli.py -q`. Expected: PASS, including every earlier test.

- [ ] **Step 5: Commit.**

```bash
git add notebook/documents.py notebook/cli.py tests/test_documents.py tests/test_cli.py
git commit -m "feat: index WebVTT transcripts with paired video"
```

---

### Task 5: Video locations and reference fields

**Files:**
- Modify: `notebook/rag.py` (`Reference`, `build_references` at line ~255, location formatting in `generate` at line ~207)
- Test: `tests/test_rag.py`, `tests/test_cli.py` (render tests)

**Interfaces:**
- Consumes: `format_time` (Task 2); chunk metadata from Task 4.
- Produces:
  - `location_label(metadata: Mapping[str, object]) -> str | None` in `notebook/rag.py`: `page N`, `lines A-B`, `video 12:40` or `None`.
  - `Reference` gains `start: float | None`, `video_file: str | None`, `video_url: str | None` (all `None` for non-video sources).
  - `build_references(documents: list[Document], starts: Mapping[int, float] | None = None) -> list[Reference]`: for a video chunk, `start` is `starts[index]` if present, else `metadata["start"]`.

- [ ] **Step 1: Write the failing tests:**

```python
def test_video_reference_has_label_start_and_video_fields():
    refs = build_references([doc(src, "x", start=760.0, end=790.0, cue_start=3, cue_end=4, video_file="lec.mp4")])
    assert refs[0]["location"] == "video 12:40"
    assert (refs[0]["start"], refs[0]["video_file"], refs[0]["video_url"]) == (760.0, "lec.mp4", None)

def test_model_start_overrides_the_chunk_start_for_that_citation_only():
    refs = build_references([video_doc(start=760.0), video_doc(start=900.0)], {1: 700.0})
    assert [r["start"] for r in refs] == [700.0, 900.0]
    assert refs[0]["location"] == "video 12:40"

def test_label_past_one_hour():
    assert build_references([video_doc(start=3760.0)])[0]["location"] == "video 1:02:40"

def test_pdf_reference_has_no_video_fields():
    assert refs[0]["start"] is None

def test_missing_location_metadata_still_raises():   # existing behavior, now including start
```

- [ ] **Step 2: Run to verify they fail.** `uv run pytest tests/test_rag.py tests/test_cli.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement** `location_label` and use it in both `generate` and `build_references`, replacing the two duplicated page/lines blocks. Keep the existing `ValueError` text when no location can be formed.

- [ ] **Step 4: Run to verify they pass**, then commit.

```bash
git add notebook/rag.py tests/test_rag.py tests/test_cli.py
git commit -m "feat: add video citation labels and reference fields"
```

---

### Task 6: Model-chosen start via structured output

**Files:**
- Create: `notebook/video_start.py`
- Modify: `notebook/rag.py` (`RagState`, `generate`)
- Modify: `notebook/api.py:~165` (pass `state.get("starts")` to `build_references`)
- Test: `tests/test_video_start.py`, `tests/test_rag.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `location_label` (Task 5); chunk metadata `start`, `cue_start`, `lead_in` (Task 4); `format_time` (Task 2).
- Produces:
  - `VideoStart(citation: int, cue: int)` and `StructuredAnswer(text: str, starts: list[VideoStart] = [])`, both pydantic models.
  - `evidence_text(document: Document) -> str`: for a video chunk, the "Preceding transcript:" block of `c{cue} [m:ss] text` lines, then the "Evidence text:" block with the chunk's first cue labelled `c{cue_start} [m:ss]`. Other sources: the chunk text.
  - `select_starts(documents: list[Document], starts: list[VideoStart]) -> dict[int, float]`: citation number (1-based) to start seconds, valid entries only.
  - `RagState.starts: dict[int, float]`.

- [ ] **Step 1: Write the failing tests** in `tests/test_video_start.py`:

```python
def test_valid_cue_in_lead_in_is_used():
    assert select_starts([video_doc(cue_start=5, start=100.0, lead_in=[{"cue": 3, "t": 70.0, "text": "a"}])], [VideoStart(citation=1, cue=3)]) == {1: 70.0}

def test_the_chunks_own_first_cue_is_valid():
    assert select_starts([doc], [VideoStart(citation=1, cue=5)]) == {1: 100.0}

@pytest.mark.parametrize("start", [VideoStart(citation=2, cue=3),  # citation is a PDF chunk
    VideoStart(citation=9, cue=3),   # unknown citation
    VideoStart(citation=1, cue=6),   # after the cited passage
    VideoStart(citation=1, cue=0)])  # outside the lookback window
def test_invalid_starts_are_ignored(start):
    assert select_starts([video_doc, pdf_doc], [start]) == {}

def test_first_valid_start_for_a_citation_wins():
    assert select_starts([doc], [VideoStart(citation=1, cue=3), VideoStart(citation=1, cue=5)]) == {1: 70.0}

def test_chunk_at_video_start_offers_only_its_own_cue():
    assert "Preceding transcript" not in evidence_text(first_chunk) and "c0 [0:00]" in evidence_text(first_chunk)
```

In `tests/test_rag.py`, with a `StructuredFakeModel` whose `with_structured_output(schema)` returns an object whose `invoke` returns a `StructuredAnswer`:
- `test_video_chunks_use_structured_output_and_return_starts`: `state["answer"]` is the structured text and `state["starts"] == {1: 70.0}`.
- `test_model_without_structured_output_falls_back_to_plain_answer`: the model raises `NotImplementedError` from `with_structured_output`; `answer` is still produced and `starts == {}`.
- `test_pdf_only_retrieval_never_asks_for_structured_output`: `with_structured_output` is not called.
- `test_citations_in_structured_text_are_still_validated`: an unknown `[7]` raises `ValueError`.
- `test_empty_structured_text_fails_clearly`.

In `tests/test_api.py`: `test_model_start_reaches_the_reference` uses a fake graph state with `starts={1: 70.0}`; the reference passed on has `start == 70.0`.

- [ ] **Step 2: Run to verify they fail.** `uv run pytest tests/test_video_start.py tests/test_rag.py tests/test_api.py -q`. Expected: the new tests FAIL.

- [ ] **Step 3: Implement `notebook/video_start.py`** with the signatures above.

- [ ] **Step 4: Wire `generate`.**
  - Use `evidence_text` for each document.
  - If any document has a `video_file` or `video_url`, add a system-prompt sentence asking for the cue where a student should start watching to have the context for the cited passage, and call `model.with_structured_output(StructuredAnswer)`.
  - Catch only `NotImplementedError` from `with_structured_output`; provider errors from `invoke` still propagate, as today.
  - Normalize the result to `StructuredAnswer`, run `_validate_citations` on its text, and return `{"answer": text, "starts": select_starts(documents, result.starts)}`.
  - Set `starts` to `{}` on the plain path.
  - Verify the structured path against the real provider in `.env` if it has a key; otherwise note it was not run.

- [ ] **Step 5: Run to verify they pass.** `uv run pytest tests/test_video_start.py tests/test_rag.py tests/test_api.py tests/test_cli.py -q`. Expected: PASS.

- [ ] **Step 6: Commit.**

```bash
git add notebook/video_start.py notebook/rag.py notebook/api.py tests
git commit -m "feat: let the model choose the video start"
```

---

### Task 7: Serving local video and building embed URLs

**Files:**
- Modify: `notebook/asset_citations.py`
- Create: `notebook/video_viewer.html`
- Modify: `notebook/api.py:86` (`_source_event`)
- Test: `tests/test_asset_citations.py`, `tests/test_api.py`

**Interfaces:**
- Consumes: `kinescope_embed_url` (Task 3); `Reference.start`, `video_file`, `video_url` (Task 5).
- Produces:
  - Routes `/citation-assets/view-pdf` (renamed from `/view`) and `/citation-assets/view-video`; `/citation-assets/files/...` serves `.pdf`, `.mp4` and `.webm` with the right media type.
  - `video_embed_url(*, video_file: str | None, video_url: str | None, start: float, title: str | None = None) -> str | None`: a Kinescope embed URL for `video_url`, otherwise a signed `view-video` link for `video_file`, otherwise `None`.

- [ ] **Step 1: Update and write the failing tests.**
  - Change the existing viewer tests and the `/citation-assets/view?` assertions in `tests/test_api.py` to `/view-pdf`.
  - `test_old_view_route_is_gone`: `GET /citation-assets/view` returns 404.
  - `test_mp4_and_webm_are_served_signed_with_media_type_and_range` (`video/mp4`, `video/webm`, 206 for a `Range` request).
  - `test_other_extensions_still_return_404` (the existing `test_non_pdf_returns_404`, now covering `.txt`).
  - `test_video_viewer_serves_html_with_headers`.
  - `test_video_embed_url_for_local_file_is_signed_and_carries_start_and_title`: the query has `asset`, `sig`, `t=125` and `title`; the path is `/citation-assets/view-video`.
  - `test_video_embed_url_for_kinescope_has_no_signature`: equals `https://kinescope.io/embed/abc123?t=125`.
  - `test_video_embed_url_is_none_when_the_file_is_missing_or_outside_the_root` (Review Focus 5).
  - `test_video_embed_url_needs_assets_settings_only_for_local_files`: Kinescope works with the `NOTEBOOK_ASSETS_*` variables unset.
  - In `tests/test_api.py`: `test_video_citation_event_has_embed_url_and_label` and `test_video_citation_with_missing_file_is_a_text_citation`.

- [ ] **Step 2: Run to verify they fail.** `uv run pytest tests/test_asset_citations.py tests/test_api.py -q`. Expected: the new and renamed tests FAIL.

- [ ] **Step 3: Implement.**
  - Extract the config check, root/base URL validation, path resolution and signing from `asset_embed_url` into a shared helper used by both `asset_embed_url` and `video_embed_url`.
  - `_asset_path(root, file, suffixes)` takes the allowed suffixes. A media-type map picks the response type.
  - Rename the `view` route to `view-pdf`; add `view-video`, which serves `video_viewer.html` with the same headers.
  - `video_viewer.html` has the title heading and an HTML5 `<video controls>`. It checks the `asset` parameter is same-origin and under `/citation-assets/files/` like the PDF viewer does. It sets `src` to the signed file URL plus `#t=<start>`, and shows the title with `textContent` only.
  - `_source_event` uses `video_embed_url(...)` when `reference["start"] is not None`, otherwise `asset_embed_url(...)`.

- [ ] **Step 4: Run to verify they pass.** `uv run pytest tests -q --ignore=tests/test_storage.py`. Expected: PASS.

- [ ] **Step 5: Commit.**

```bash
git add notebook/asset_citations.py notebook/video_viewer.html notebook/api.py tests
git commit -m "feat: serve local video and embed Kinescope citations"
```

---

### Task 8: Documentation and end-to-end check

**Files:**
- Modify: `README.md`

- [ ] **Step 1: Document video** in `README.md`: a "Video citations" section covering the `.vtt` plus sibling `.mp4`/`.webm` pairing, `--video-url` for Kinescope with an example command, that transcripts come from a separate transcriber, that the model chooses the start and falls back to the chunk start, and that local video uses `NOTEBOOK_ASSETS_*`. Update the first paragraph and the "PDF citations" heading's wording where it says only PDFs open in the sidebar. Add `tests/test_transcripts.py tests/test_video_hosts.py tests/test_video_start.py` to the provider-independent test command.

- [ ] **Step 2: Run the full check.** `uv run ruff check . && uv run pytest tests -q --ignore=tests/test_storage.py`. Expected: no findings, all tests pass.

- [ ] **Step 3: Manual end-to-end.** With the app running, index a short local `.mp4` plus `.vtt`, and a `.vtt` with `--video-url` for the Kinescope video from Task 1. Ask a question about each.
Expected: each citation shows `video m:ss`; clicking opens the video in the sidebar a little earlier than the label (the local one paused, the Kinescope one at `t`); seeking in the local video works; a second click on the same citation reloads at the start time.
Record the result for the user. If the `no-store` cache header breaks seeking, loosen `Cache-Control` for video files only and add a test for it.

- [ ] **Step 4: Commit.**

```bash
git add README.md
git commit -m "docs: describe video indexing and citations"
```
