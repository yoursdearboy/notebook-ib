# Video indexing and citations

## Goal

A student asks a question and gets an answer with citations. A citation to a video opens that video in Open WebUI's right sidebar, a little before the cited moment, so the student has time to settle in and see the context.

Transcription is out of scope. A separate transcriber produces a WebVTT file before indexing, and this app only reads it.

## Decisions

- **Transcript format:** WebVTT (`.vtt`), an existing standard. Parsed with an existing library (`webvtt-py`), not a hand-written parser.
- **Video sources:** local files (`.mp4`, `.webm`) served from `NOTEBOOK_ASSETS_ROOT`, and Kinescope (`kinescope.io`) as the only third-party host, behind a small adapter so another host can be added later.
- **Registering a third-party video:** the `--video-url` CLI flag on a single-file index. There is no sidecar file and no header in the `.vtt`.
- **Playback start:** chosen by the chat model through structured output, from the transcript just before the cited passage. There is no configurable lead-in offset.
- **Fallback start:** the cited chunk's own first cue.
- **Citation label:** the time of the cited passage, for example `video 12:40`. The player starts earlier than the label.

## 1. Ingestion and storage

- `.vtt` becomes a supported file type. The transcript is the indexed source. `source` stores its path relative to `NOTEBOOK_ASSETS_ROOT`, as for PDFs.
- Cues are parsed with `webvtt-py`. Markup such as `<v Speaker>` tags is stripped.
- **Pairing a video:**
  - Local: a sibling file with the same stem and a `.mp4` or `.webm` extension.
  - Third-party: `--video-url https://kinescope.io/VIDEO_ID` when indexing a single file. Accepted forms are `kinescope.io/VIDEO_ID` and `kinescope.io/embed/VIDEO_ID`. Anything else is an error.
  - A transcript with neither is an error and is not indexed. Both together is an error, because it is ambiguous.
- **Chunking** is cue-aligned and never splits a cue. Cues accumulate up to the existing `chunk_size`. The overlap is the trailing cues that fit within `chunk_overlap`. No new settings.
- **Chunk metadata:**
  - `start` and `end`, in seconds.
  - `cue_start` and `cue_end`, as cue indexes.
  - `video_file` (path relative to the assets root) or `video_url` (canonical Kinescope URL).
  - `lead_in`: the numbered cues, with time and text, in the 60 seconds before the chunk. This is the candidate list for the answer step, so the `.vtt` is not re-read at answer time. The 60 second window is fixed and is part of the split signature, so changing it requires a reindex.
- Reindexing replaces all chunks of the source, as today. A changed `--video-url` takes effect on reindex.
- `--title` works as for other sources.

## 2. Citations and the answer step

- **Location label:** `video 12:40`, or `1:02:40` past an hour. It is the time of the cited passage. `build_references` accepts it alongside `page N` and `lines A-B`.
- **Reference start:** each video reference carries a separate `start` field. It is where playback begins and may be earlier than the label.
- **Evidence in the prompt.** A video chunk is shown as numbered cues with times:

  ```
  [2] Source: Lecture 4 (video 12:40)
  Preceding transcript:
  c41 [12:05] ...
  c42 [12:21] ...
  Evidence text:
  c43 [12:40] ...
  ```

- **Structured output.** When any retrieved chunk is a video, the model call returns `{text, starts: [{citation, cue}]}`. For each video citation the model names the cue where the student should start watching, using the cue IDs from the prompt. The prompt asks for a start early enough to give the context needed to follow the cited passage. When no retrieved chunk is a video, the current plain path is unchanged.
- **Validation.** The citation must be a retrieved video chunk. The cue must be one of that chunk's preceding cues or its own first cue, so it cannot come after the cited passage or outside the lookback window. The answer text still goes through `_validate_citations`.
- **Fallback.** The start is the chunk's own first cue when the model cannot choose: the `extractive:` dev backend, a model without structured-output support, an invalid cue, or a skipped citation.

## 3. Viewer and serving

- **Kinescope.** No signed link and no viewer page. `embed_url` is `https://kinescope.io/embed/VIDEO_ID?t=SECONDS`, built at answer time from `video_url`. `t` sets the start and leaves the whole video seekable. `seek` is not used, because it cuts everything before the start. No new settings.
- **Local files.**
  - `_asset_path` and `asset_file` in `asset_citations.py` accept `.mp4` and `.webm` as well as `.pdf` and pick the media type by extension.
  - A new viewer route `/citation-assets/view-video` shows an HTML5 `<video controls>` pointed at the signed file with the media fragment `#t=SECONDS`. It starts paused, because browsers block autoplay with sound. It shows the title, like the PDF viewer.
  - The PDF viewer is renamed from `/citation-assets/view` to `/citation-assets/view-pdf`. There is no alias, because the app is in development. `asset_embed_url` picks the route by file type, and existing tests change to the new URL.
  - Signatures cover the file path only, as today. The start time and title are not signed, so changing them cannot expose another file.
  - Local video uses the existing `NOTEBOOK_ASSETS_*` settings.

## 4. Errors, testing, scope

**Errors at index time.** Each fails that file; other files continue.
- A transcript with no sibling video and no `--video-url`.
- A transcript with both.
- A `--video-url` that is not a Kinescope URL.
- `--video-url` given for a directory or more than one file.
- A `.vtt` with no usable cues.

**Errors at answer time.** The answer never fails because of video.
- An invalid or missing start uses the fallback.
- Without structured-output support, the plain path is used and all video citations use the fallback.
- A missing local video file means the citation gets no `embed_url`, like a missing PDF today. The Sources entry still shows the excerpt and the time.

**Tests.**
- Parsing and chunking: a fixture `.vtt` gives cue-aligned chunks with the right `start`, `end`, `cue_start`, `cue_end` and `lead_in`.
- Pairing and validation: the sibling-video and `--video-url` cases and the index-time errors above.
- Answer step: a fake chat model returns structured starts. A valid cue is used. Invalid, missing and unsupported cases fall back to the chunk start.
- Citations: the `video 12:40` label, time formatting past an hour, and `build_references` accepting video locations.
- Serving: signed `.mp4` and `.webm` links, rejection of bad signatures and paths, the renamed PDF route, the `view-video` page, and Kinescope `embed_url` with `t=`.
- Manual: one real Kinescope video and one local file played in the Open WebUI sidebar.

**Out of scope.**
- Producing transcripts or integrating a transcriber.
- Other hosts such as YouTube or Vimeo, and SRT input.
- Captions in the local player.
- Citing a time range instead of a start time.
- Per-citation or per-user start settings.
- Re-reading the `.vtt` at answer time.

## Verify early in the plan

These are unverified assumptions. Check them before the rest of the video work.

1. Open WebUI's sidebar can load a cross-origin Kinescope iframe, with no blocking by its CSP or iframe sandbox.
2. Kinescope's dashboard does not restrict embedding by domain in a way that breaks local development. The docs read do not mention a whitelist.
3. The pinned Starlette version's `FileResponse` handles HTTP range requests, so seeking in local video works.
4. `Cache-Control: private, no-store` does not break seeking. If it does, loosen it for video only.
5. How `bin/frontend`'s patch treats a video `embed_url`, so a repeated click on the same citation restarts from the start time.
6. The models in use support structured output through LangChain's `with_structured_output`.
