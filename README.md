# Youtube Captions Export

A command-line utility for exporting **YouTube text transcripts** (subtitles — manual and automatic) **without downloading the video itself**.

Features:

* Supports channels, playlists, and individual videos (`--channel`, `--playlist`, `--video`).
* Picks a subtitle track: first manual, then automatic ASR, then machine translation (`--langs ru,en` — language priorities).
* Cleans up "rolling" repeats in automatic captions.
* Output formats: `txt` (default), `json`, `srt`, `vtt`.

* Python 3.11+
* runtime dependency: `yt-dlp`
* tests: `pytest`; everything else is the standard library
* optionally: `curl_cffi` — enables `--impersonate` (browser TLS impersonation)

## Installation

```bash
pip install -r requirements.txt
```

## Quick start

**Double-click `ytsubs.bat`** — that's the only file you need to run.

* First time it asks for a link (channel, playlist, or video — it detects it) and a format.
* Next time it's convenient to put the link into `YT_SOURCE` at the top of `ytsubs.bat` — then it starts immediately without asking. Next to it are cookies, proxy, number of workers and limit.
* Already exported videos are skipped, so a re-run picks up where it stopped.

From the terminal the same arguments as `ytsubs.bat` work directly:

```bash
python ytsubs.py --channel https://www.youtube.com/@channel --langs ru,en --format txt
python ytsubs.py --video https://www.youtube.com/watch?v=VIDEO_ID --format srt
python ytsubs.py --playlist https://www.youtube.com/playlist?list=LIST_ID \
    --limit 10 --workers 8 --cookies cookies.txt
```

## Use cases

- **LLM context**: exported transcripts can serve as context for neural-network analysis — e.g., thematic analysis, Q&A, summarization.

- **Corpus building**: multiple channels/playlists → combined dataset → LLM.

## Flags

| Flag | Description |
| --- | --- |
| `--channel URL` | channel URL (can be repeated) |
| `--playlist URL` | playlist URL (can be repeated) |
| `--video URL` | video URL (can be repeated) |
| `--langs ru,en` | languages by priority (default `ru,en`) |
| `--format txt\|json\|srt\|vtt` | output format (default `txt`) |
| `--out DIR` | output directory (default `out`) |
| `--limit N` | maximum number of videos |
| `--cookies FILE` | cookies file for yt-dlp |
| `--cookies-from-browser NAME` | take cookies from an installed browser (`chrome`, `edge`, `firefox`, …) |
| `--workers N` | number of download workers (default `4`, `1` — sequential) |
| `--min-interval SEC` | minimum delay between network requests (default `1.0`) |
| `--sleep-subtitles SEC` | pause before loading an auto-translated subtitle (default `60`, `0` — off) |
| `--impersonate CLIENT` | TLS impersonation for yt-dlp/curl_cffi (default `auto`) |
| `--proxy URL` | proxy for yt-dlp (`socks5://…`, `http://…`) |

At least one of `--channel/--playlist/--video` is required.

## Behavior

* The output directory depends on the source: channel → `out/<channel>/`; single video → `out/<video>/`.

* Filename: `<video_id>_<title-slug>.<ext>`; the slug is transliterated (Cyrillic → Latin) and cleaned of forbidden characters.
* Auto-captions are cleaned of "rolling" repeats.
* Videos without subtitles are skipped with a message, processing continues.
* A per-video error doesn't stop the others (the final exit code stays `0`).
* Already exported videos are not downloaded again: the file is matched by video id (`<id>_*.txt),
  so a change of the localized title between runs doesn't force a re-download. A different format (`--format json`) is a separate file.
* The file is written atomically: `<name>.part` is written, then `os.replace`.
* Progress: `[done/total] <video_id> <title> — <lang> (ok|no subs|error)`.
  For errors the reason from yt-dlp is printed at the end of the line with the stage marked:
  `[metadata]` (video card request) or `[subtitles]` (track download) — immediately see where the 429 came from.

## Exit codes

* `0` — success (including partial per-video errors);
* `1` — fatal error (no URL) or the run was stopped due to YouTube's limit;
* `2` — argument error.

## Troubleshooting

### `HTTP Error 429: Too Many Requests`

YouTube hard-limits requests for **auto-translated** subtitles
([yt-dlp issue #13831](https://github.com/yt-dlp/yt-dlp/issues/13831)).

**Main solution.** For Russian videos YouTube usually keeps two tracks next to each other:
`ru` (`tlang=ru`, translation) and `ru-orig` (original speech recognition, no `tlang`).
The utility **prefers `ru-orig`**: it is not subject to the limit and gives a more accurate text.

If 429 hits even original tracks, on manual subtitles and on fresh videos in all formats
(`json3/srv1/srv2/srv3/ttml/srt/vtt`) — it's an IP ban, not about translations; neither cookies,
nor `Referer/Origin`, nor browser headers help. The remaining options: change IP (`--proxy`, VPN)
or wait.

### `Sign in to confirm you're not a bot`

YouTube asks for cookies. Export cookies from the browser into `cookies.txt`
and run with `--cookies cookies.txt`.

### No subtitles

The video has neither manual nor automatic subtitles: it is skipped, processing continues.

## Why not the YouTube Data API?

`captions.download` in the YouTube Data API v3 is available only to the channel owner
(through OAuth) and only for tracks uploaded by the owner themselves.
Automatic (ASR) subtitles and subtitles of other channels are not available via the API.

Therefore the working "logged-in" path here is browser cookies: `--cookies-from-browser firefox`.

## Architecture

The core is network-agnostic and tested via a substitute client:

```python
class YouTubeClient(Protocol):
    def iter_videos(self, url, limit=None) -> Iterable[VideoRef]: ...
    def fetch_transcript(self, video_id, langs) -> Transcript | None: ...
```

* The core (`collect_videos`, `export_videos`, parsers, cleanup, formatting, writing) — pure logic, covered by tests.
* `YtDlpClient` — a thin network adapter on top of `yt-dlp` with retry and exponential backoff. In tests it is always substituted with a fake.
* Parallelism: `ThreadPoolExecutor`, injected as a dependency (`executor_factory`) — in tests a synchronous executor is substituted.

## Tests

```bash
python -m pytest -q
```

Tests are deterministic, without network and without `sleep`; the parallelism proof is via `threading.Barrier`.
