"""ytsubs - export YouTube text transcripts (subtitles) without downloading video.

Core is network-agnostic: it talks to YouTube through the YouTubeClient protocol.
The real implementation (YtDlpClient) is a thin adapter at the bottom of the file.
"""

from __future__ import annotations

import glob
import os
import re
import sys
import threading
import time
from argparse import ArgumentParser, ArgumentTypeError
from collections.abc import Callable, Iterable, Sequence
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol
from urllib.parse import parse_qs, urlsplit, urlunsplit


# --- Data model ------------------------------------------------------------

@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class VideoRef:
    id: str
    title: str
    url: str


@dataclass(frozen=True)
class Transcript:
    video_ref: VideoRef
    lang: str
    segments: list[Segment]
    translated: bool = False


# --- Timed-text parsing ----------------------------------------------------

_TS_RE = re.compile(r"(?:(\d+):)?(\d{1,2}):(\d{2})[.,](\d{1,3})")
_TAG_RE = re.compile(r"<[^>]*>")


def _parse_timestamp(raw: str) -> float:
    m = _TS_RE.search(raw)
    if not m:
        raise ValueError(f"bad timestamp: {raw!r}")
    hours = int(m.group(1)) if m.group(1) else 0
    minutes = int(m.group(2))
    seconds = int(m.group(3))
    millis = int(m.group(4).ljust(3, "0"))
    return hours * 3600 + minutes * 60 + seconds + millis / 1000.0


def _clean_cue_text(lines: list[str]) -> str:
    joined = " ".join(lines)
    joined = _TAG_RE.sub("", joined)
    return re.sub(r"\s+", " ", joined).strip()


def _parse_timed_text(text: str) -> list[Segment]:
    """Parse WebVTT/SRT style text into segments."""
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    segments: list[Segment] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if "-->" not in line:
            i += 1
            continue
        left, _, right = line.partition("-->")
        start = _parse_timestamp(left)
        end_raw = right.strip().split()[0] if right.strip() else ""
        end = _parse_timestamp(end_raw) if end_raw else start
        i += 1
        buf: list[str] = []
        while i < len(lines) and lines[i].strip():
            buf.append(lines[i])
            i += 1
        cue = _clean_cue_text(buf)
        if cue:
            segments.append(Segment(start=start, end=end, text=cue))
    return segments


def parse_vtt(text: str) -> list[Segment]:
    return _parse_timed_text(text)


def parse_srt(text: str) -> list[Segment]:
    return _parse_timed_text(text)


# --- Auto-caption cleanup --------------------------------------------------

def dedupe_rolling(segments: list[Segment]) -> list[Segment]:
    """Collapse YouTube auto-caption 'rolling' repeats.

    Auto-captions re-emit growing prefixes of the same line. We keep only the
    final, longest version of each run and drop exact duplicates.
    """
    out: list[Segment] = []
    for seg in segments:
        text = seg.text.strip()
        if not text:
            continue
        if out:
            prev = out[-1]
            if text == prev.text:
                out[-1] = Segment(prev.start, max(prev.end, seg.end), prev.text)
                continue
            if prev.text.startswith(text):
                continue
            if text.startswith(prev.text):
                out.pop()
        out.append(Segment(seg.start, seg.end, text))
    return out


# --- Text assembly ---------------------------------------------------------

def segments_to_text(segments: list[Segment]) -> str:
    parts = [s.text.strip() for s in segments]
    joined = " ".join(p for p in parts if p)
    return re.sub(r"\s+", " ", joined).strip()


# --- Language selection ----------------------------------------------------

def select_language(available: list[str], wanted: list[str]) -> str | None:
    """Pick the best available language respecting the priority order of `wanted`.

    YouTube names the original (untranslated) ASR track ``<lang>-orig`` and
    keeps the machine translation under ``<lang>``. The translated track is the
    one the timedtext endpoint rate-limits with HTTP 429 (yt-dlp issue #13831),
    so an original track always wins.
    """
    for want in wanted:
        for candidate in (want + "-orig", want):
            if candidate in available:
                return candidate
        regional = [lang for lang in available if lang.startswith(want + "-")]
        for lang in regional:
            if lang.endswith("-orig"):
                return lang
        if regional:
            return regional[0]
    return None


# --- File names ------------------------------------------------------------

_CYR_ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюя"
_LAT_ALPHABET = [
    "a", "b", "v", "g", "d", "e", "e", "zh", "z", "i", "y", "k", "l", "m",
    "n", "o", "p", "r", "s", "t", "u", "f", "kh", "ts", "ch", "sh", "shch",
    "", "y", "", "e", "yu", "ya",
]
_TRANSLIT = str.maketrans(dict(zip(_CYR_ALPHABET, _LAT_ALPHABET)))


def slugify(text: str, max_len: int = 80) -> str:
    text = text.lower().translate(_TRANSLIT)
    text = re.sub(r"[^a-z0-9]+", "-", text).strip("-")
    return text[:max_len].strip("-")


def build_filename(video_id: str, title: str, ext: str) -> str:
    slug = slugify(title) or "video"
    return f"{video_id}_{slug}.{ext}"


# --- Formatting ------------------------------------------------------------

FORMATS = ("txt", "json", "srt", "vtt")


def _fmt_timestamp(seconds: float, sep: str) -> str:
    total_ms = round(seconds * 1000)
    hours, rest = divmod(total_ms, 3_600_000)
    minutes, rest = divmod(rest, 60_000)
    secs, millis = divmod(rest, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def format_transcript(transcript: Transcript, fmt: str) -> str:
    segments = transcript.segments
    if fmt == "txt":
        return segments_to_text(segments)
    if fmt == "json":
        return _format_json(transcript)
    if fmt == "srt":
        return _format_subtitle(segments, sep=",", header="")
    if fmt == "vtt":
        return _format_subtitle(segments, sep=".", header="WEBVTT\n\n")
    raise ValueError(f"unknown format: {fmt!r}")


def _format_json(transcript: Transcript) -> str:
    import json

    ref = transcript.video_ref
    payload = {
        "id": ref.id,
        "title": ref.title,
        "url": ref.url,
        "lang": transcript.lang,
        "segments": [
            {"start": s.start, "end": s.end, "text": s.text}
            for s in transcript.segments
        ],
    }
    return json.dumps(payload, ensure_ascii=False, indent=2)


def _format_subtitle(segments: list[Segment], sep: str, header: str) -> str:
    blocks: list[str] = []
    for i, seg in enumerate(segments, start=1):
        stamp = f"{_fmt_timestamp(seg.start, sep)} --> {_fmt_timestamp(seg.end, sep)}"
        blocks.append(f"{i}\n{stamp}\n{seg.text}")
    body = "\n\n".join(blocks)
    return f"{header}{body}\n" if body else header


# --- YouTube access abstraction --------------------------------------------

class YouTubeClient(Protocol):
    def iter_videos(self, url: str, limit: int | None = None) -> Iterable[VideoRef]: ...
    def fetch_transcript(self, video_id: str, langs: list[str]) -> Transcript | None: ...


def collect_videos(
    client: YouTubeClient, urls: Sequence[str], limit: int | None = None
) -> list[VideoRef]:
    """Enumerate videos from all sources, deduplicate by id, honour the limit."""
    seen: set[str] = set()
    out: list[VideoRef] = []
    for url in urls:
        remaining = None if limit is None else limit - len(out)
        if remaining is not None and remaining <= 0:
            break
        for ref in client.iter_videos(url, limit=remaining):
            if ref.id in seen:
                continue
            seen.add(ref.id)
            out.append(ref)
            if limit is not None and len(out) >= limit:
                break
    return out


# --- Export engine ---------------------------------------------------------

@dataclass(frozen=True)
class VideoResult:
    video_id: str
    title: str
    status: str  # ok | no-subs | error | skipped
    lang: str | None = None
    path: Path | None = None
    error: str | None = None
    translated: bool = False


@dataclass
class RunReport:
    total: int
    results: list[VideoResult] = field(default_factory=list)
    processed: int = 0
    aborted: bool = False

    def count(self, status: str) -> int:
        return sum(1 for r in self.results if r.status == status)

    def count_translated(self) -> int:
        return sum(1 for r in self.results if r.translated)


_STATUS_LABELS = {
    "ok": "ok",
    "skipped": "уже выгружено",
    "no-subs": "нет субтитров",
    "error": "ошибка",
    "aborted": "пропущено (лимит YouTube)",
}


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".part")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def _existing_output(out_dir: Path, video_id: str, fmt: str) -> Path | None:
    """Return an already written transcript for `video_id`, if there is one.

    Matching ignores the title slug on purpose: YouTube localizes titles, so the
    same video can hash to a different file name between runs. Without this a
    resumed run would re-download everything it had already exported.
    """
    pattern = f"{glob.escape(video_id)}_*.{glob.escape(fmt)}"
    for path in sorted(out_dir.glob(pattern)):
        return path
    return None


def _process_video(
    video: VideoRef, client: YouTubeClient, out_dir: Path, langs: list[str], fmt: str
) -> VideoResult:
    existing = _existing_output(out_dir, video.id, fmt)
    if existing is not None:
        return VideoResult(video.id, video.title, "skipped", path=existing)
    target = out_dir / build_filename(video.id, video.title, fmt)
    transcript = client.fetch_transcript(video.id, list(langs))
    if transcript is None:
        return VideoResult(video.id, video.title, "no-subs")
    cleaned = Transcript(
        video_ref=transcript.video_ref,
        lang=transcript.lang,
        segments=dedupe_rolling(transcript.segments),
        translated=transcript.translated,
    )
    _atomic_write(target, format_transcript(cleaned, fmt))
    return VideoResult(video.id, video.title, "ok", lang=cleaned.lang, path=target,
                       translated=cleaned.translated)


def _default_executor_factory(workers: int):
    return ThreadPoolExecutor(max_workers=workers)


def export_videos(
    videos: Sequence[VideoRef],
    client: YouTubeClient,
    out_dir: str | Path,
    langs: Sequence[str],
    fmt: str = "txt",
    workers: int = 4,
    executor_factory: Callable[[int], object] | None = None,
    logger: Callable[[str], None] = print,
) -> RunReport:
    """Fetch and write transcripts for `videos`, optionally in parallel."""
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    total = len(videos)
    report = RunReport(total=total, results=[None] * total)  # type: ignore[list-item]
    if total == 0:
        return report

    lock = threading.Lock()
    abort = threading.Event()
    done = 0
    lang_list = list(langs)

    def work(index: int, video: VideoRef) -> VideoResult:
        nonlocal done
        if abort.is_set():
            result = VideoResult(video.id, video.title, "aborted")
        else:
            try:
                result = _process_video(video, client, out_path, lang_list, fmt)
            except RateLimited as exc:
                abort.set()
                result = VideoResult(video.id, video.title, "error", error=str(exc))
            except Exception as exc:  # noqa: BLE001 - per-video failures are non-fatal
                result = VideoResult(video.id, video.title, "error", error=str(exc))
        label = _STATUS_LABELS.get(result.status, result.status)
        with lock:
            done += 1
            report.results[index] = result
            line = (f"[{done}/{total}] {video.id} {video.title} — "
                    f"{result.lang or '-'} ({label})")
            if result.translated:
                line += " [автоперевод]"
            if result.error:
                line += f": {result.error}"
            logger(line)
        return result

    factory = executor_factory or _default_executor_factory
    executor = factory(workers)
    try:
        futures = [executor.submit(work, i, v) for i, v in enumerate(videos)]
        for future in futures:
            future.result()
    finally:
        executor.shutdown(wait=True)
    report.processed = done
    report.aborted = abort.is_set()
    return report


# --- yt-dlp adapter (thin, network-facing) ---------------------------------

class RateLimited(Exception):
    """YouTube is throttling us; retrying this run is pointless."""


_THROTTLE_MARKERS = ("429", "too many requests", "sign in to confirm", "not a bot")


def is_throttle_error(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _THROTTLE_MARKERS)


def _tag(phase: str, fn):
    """Run `fn`, prefixing a failure with the pipeline phase it came from.

    `RateLimited` is re-raised untouched so the circuit breaker stays visible.
    """
    try:
        return fn()
    except RateLimited:
        raise
    except Exception as exc:  # noqa: BLE001 - add context, keep the cause
        raise RuntimeError(f"[{phase}] {exc}") from exc


class RateLimiter:
    """Global (cross-worker) pacing for network calls.

    Spacing grows on failure (429 / "Sign in to confirm...") and shrinks again
    as soon as requests start succeeding, so a throttled run backs off instead
    of hammering YouTube.
    """

    def __init__(self, min_interval: float = 1.0, max_interval: float = 30.0,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep):
        self.min_interval = min_interval
        self.max_interval = max_interval
        self._clock = clock
        self._sleep = sleep
        self._lock = threading.Lock()
        self._interval = min_interval
        self._next_allowed = 0.0
        self._penalty_until = 0.0
        self._gate = 0.0

    @property
    def interval(self) -> float:
        with self._lock:
            return self._interval

    def wait(self) -> None:
        while True:
            with self._lock:
                now = self._clock()
                delay = max(self._next_allowed, self._penalty_until) - now
                if delay <= 0:
                    self._next_allowed = now + self._interval
                    return
            self._sleep(delay)

    def penalize(self, delay: float) -> None:
        with self._lock:
            now = self._clock()
            self._penalty_until = max(self._penalty_until, now + delay)
            self._interval = min(max(self._interval * 2, delay), self.max_interval)

    def hold(self, seconds: float) -> None:
        """Reserve a slot for a rate-limited request.

        Consecutive holds are queued `seconds` apart, so parallel workers pace
        themselves instead of firing a burst once a shared pause expires.
        """
        if seconds <= 0:
            return
        with self._lock:
            start = max(self._clock(), self._gate)
            self._gate = start + seconds
            self._penalty_until = max(self._penalty_until, start)

    def relax(self) -> None:
        with self._lock:
            self._interval = max(self._interval / 2, self.min_interval)


def pick_caption_url(formats: list[dict]) -> str | None:
    """Choose the best caption track URL, preferring WebVTT.

    Original (non-translated) formats win over machine-translated ones, which
    YouTube rate-limits with HTTP 429 (yt-dlp issue #13831).
    """
    order = ((False, "vtt"), (False, "srt"), (False, "json3"),
             (True, "vtt"), (True, "srt"), (True, "json3"))
    for translated, ext in order:
        for fmt in formats:
            url = fmt.get("url")
            if (fmt.get("ext") == ext and url
                    and is_translated_caption(url) == translated):
                return url
    for fmt in formats:
        if fmt.get("url"):
            return fmt["url"]
    return None


def is_translated_caption(url: str) -> bool:
    """True for auto-translated tracks.

    YouTube's timedtext endpoint answers with HTTP 429 for tracks carrying a
    `tlang` parameter (a machine translation of some source track), while
    manual subtitles and the original-language ASR track are served normally.
    See yt-dlp issue #13831.
    """
    return bool(parse_qs(urlsplit(url).query).get("tlang"))


_BARE_CHANNEL_RE = re.compile(r"^/(?:@[^/]+|channel/[^/]+|c/[^/]+|user/[^/]+)/?$")


def normalize_channel_url(url: str) -> str:
    """Turn a bare channel URL into its `/videos` tab.

    Without the tab suffix yt-dlp returns the channel's tab playlists
    (Videos/Live/Shorts) instead of the videos themselves.
    """
    parts = urlsplit(url)
    host = parts.netloc.lower()
    if not (host == "youtube.com" or host.endswith(".youtube.com")):
        return url
    if not _BARE_CHANNEL_RE.match(parts.path):
        return url
    path = parts.path.rstrip("/") + "/videos"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, parts.fragment))


_CHANNEL_NAME_RE = re.compile(
    r"/(?:@([^/@]+)|c/([^/?]+)|channel/([^/?]+)|user/([^/?]+))"
)


def channel_slug(url: str) -> str:
    """Return the folder slug for a channel source: its name, transliterated."""
    m = _CHANNEL_NAME_RE.search(url)
    name = next((g for g in m.groups() if g), "") if m else ""
    return slugify(name) if name else slugify(url)


def _entries_of(info: dict | None) -> list[dict]:
    if not info:
        return []
    return [e for e in (info.get("entries") or []) if e]


def channel_tab_url(entries: Sequence[dict]) -> str | None:
    """Return the Videos-tab URL when `entries` are a channel's tab playlists.

    A bare channel URL makes yt-dlp return the channel's tabs (Videos, Live,
    Shorts) as nested playlists that all share the channel id. Those are not
    videos, so we expand the Videos tab instead.
    """
    tabs = [e for e in entries if isinstance(e, dict) and e.get("_type") == "playlist"]
    if not tabs or len(tabs) != len(entries):
        return None
    for tab in tabs:
        url = tab.get("webpage_url") or ""
        if url.rstrip("/").endswith("/videos"):
            return url
    return tabs[0].get("webpage_url")


def choose_track(
    subtitles: dict, automatic_captions: dict, langs: Sequence[str]
) -> tuple[str, list[dict]] | None:
    """Pick a caption track: author-provided subtitles win over automatic ones."""
    lang = select_language(list(subtitles), list(langs))
    if lang is not None:
        return lang, subtitles[lang]
    lang = select_language(list(automatic_captions), list(langs))
    if lang is not None:
        return lang, automatic_captions[lang]
    return None


def parse_json3(text: str) -> list[Segment]:
    """Parse YouTube's json3 caption format."""
    import json

    data = json.loads(text)
    segments: list[Segment] = []
    for event in data.get("events", []):
        segs = event.get("segs")
        if not segs:
            continue
        body = "".join(seg.get("utf8", "") for seg in segs)
        body = re.sub(r"\s+", " ", body).strip()
        if not body:
            continue
        start = event.get("tStartMs", 0) / 1000.0
        end = start + event.get("dDurationMs", 0) / 1000.0
        segments.append(Segment(start=start, end=end, text=body))
    return segments


def _extract_browser_cookies(browser: str):
    """Read cookies straight from an installed browser, through yt-dlp."""
    from yt_dlp.cookies import extract_cookies_from_browser

    return extract_cookies_from_browser(browser, logger=_SilentLogger())


class YtDlpClient:
    """YouTubeClient implementation on top of yt-dlp (no video downloads)."""

    def __init__(self, cookies: str | None = None, retries: int = 3,
                 base_delay: float = 2.0, max_delay: float = 30.0,
                 min_interval: float = 1.0, max_interval: float = 10.0,
                 throttle_window: int = 20, throttle_limit: int = 8,
                 sleep_subtitles: float = 0.0, impersonate: str | None = None,
                 proxy: str | None = None,
                 cookies_from_browser: str | None = None,
                 rate_limiter: RateLimiter | None = None):
        self.cookies = cookies
        self.cookies_from_browser = cookies_from_browser
        self.retries = retries
        self.base_delay = base_delay
        self.max_delay = max_delay
        self.throttle_limit = throttle_limit
        self.sleep_subtitles = sleep_subtitles
        self.impersonate = impersonate
        self.proxy = proxy
        self.rate_limiter = rate_limiter or RateLimiter(
            min_interval=min_interval, max_interval=max_interval
        )
        self._attempts: deque[bool] = deque(maxlen=throttle_window)
        self._cookiejar = None
        self._cookie_lock = threading.Lock()

    @property
    def throttle_failures(self) -> int:
        return sum(self._attempts)

    # -- internals --

    def _options(self, extra: dict) -> dict:
        opts = {
            "skip_download": True,
            "ignore_no_formats_error": True,
            "quiet": True,
            "no_warnings": True,
            "noprogress": True,
            "logger": _SilentLogger(),
        }
        if self.sleep_subtitles:
            opts["sleep_interval_subtitles"] = self.sleep_subtitles
        if self.proxy:
            opts["proxy"] = self.proxy
        if self.impersonate:
            from yt_dlp.networking.impersonate import ImpersonateTarget

            opts["impersonate"] = ImpersonateTarget.from_str(self.impersonate.lower())
        opts.update(extra)
        return opts

    @contextmanager
    def _ydl(self, extra: dict):
        """YoutubeDL that reuses one cookie jar and never rewrites the file."""
        import yt_dlp

        with yt_dlp.YoutubeDL(self._options(extra)) as ydl:
            jar = self._cookie_jar()
            if jar is not None:
                # Set on the instance rather than via `cookiefile`: parallel
                # workers would otherwise rewrite the user's cookies file
                # concurrently on close.
                ydl.cookiejar = jar
            yield ydl

    def _cookie_jar(self):
        if not self.cookies and not self.cookies_from_browser:
            return None
        with self._cookie_lock:
            if self._cookiejar is None:
                self._cookiejar = self._load_cookie_jar()
            return self._cookiejar

    def _load_cookie_jar(self):
        """Build one shared jar, so parallel workers never rewrite the source."""
        if self.cookies:
            from yt_dlp.cookies import YoutubeDLCookieJar

            jar = YoutubeDLCookieJar(self.cookies)
            jar.load()
            return jar
        return _extract_browser_cookies(self.cookies_from_browser)

    def _extract_info(self, url: str, extra: dict) -> dict:
        with self._ydl(extra) as ydl:
            return ydl.extract_info(url, download=False)

    def _fetch_caption(self, url: str) -> str:
        """Fetch a caption URL through yt-dlp's networking stack.

        yt-dlp applies cookies, browser headers and TLS impersonation here;
        a bare urlopen gets answered with HTTP 429 far more often.
        """
        with self._ydl({}) as ydl:
            response = ydl.urlopen(url)
            try:
                return response.read().decode("utf-8", errors="replace")
            finally:
                response.close()

    def _retry(self, fn):
        delay = self.base_delay
        last: Exception | None = None
        for attempt in range(self.retries + 1):
            self.rate_limiter.wait()
            try:
                result = fn()
            except Exception as exc:  # noqa: BLE001 - network flakiness
                last = exc
                throttle = is_throttle_error(exc)
                self._note(throttle)
                # Throttling rarely clears within one video: one retry, then stop.
                if attempt == self.retries or (throttle and attempt >= 1):
                    break
                # Shared penalty: every worker waits, not just this one.
                self.rate_limiter.penalize(self._next_delay(exc, delay))
                delay = min(delay * 2, self.max_delay)
            else:
                self._note(False)
                self.rate_limiter.relax()
                return result
        assert last is not None
        raise last

    def _note(self, throttled: bool) -> None:
        """Track throttling over a sliding window and trip a circuit breaker."""
        self._attempts.append(throttled)
        failures = sum(self._attempts)
        if failures >= self.throttle_limit:
            raise RateLimited(
                f"YouTube throttled {failures} of the last {len(self._attempts)} "
                "requests (429 / 'Sign in to confirm you're not a bot')"
            )

    def _next_delay(self, exc: Exception, delay: float) -> float:
        headers = getattr(exc, "headers", None)
        if headers:
            try:
                return min(float(headers.get("Retry-After")), self.max_delay)
            except (TypeError, ValueError):
                pass
        return min(delay, self.max_delay)

    @staticmethod
    def _ref_from_info(info: dict) -> VideoRef | None:
        vid = info.get("id")
        if not vid:
            return None
        return VideoRef(
            id=vid,
            title=info.get("title") or vid,
            url=info.get("webpage_url") or info.get("url")
            or f"https://www.youtube.com/watch?v={vid}",
        )

    # -- YouTubeClient protocol --

    def iter_videos(self, url: str, limit: int | None = None) -> list[VideoRef]:
        url = normalize_channel_url(url)
        info = self._extract_flat(url, limit)
        entries = _entries_of(info)
        tab_url = channel_tab_url(entries)
        if tab_url:
            info = self._extract_flat(tab_url, limit)
            entries = _entries_of(info)
        if not entries:
            single = self._ref_from_info(info) if info else None
            return [single] if single else []
        refs = [self._ref_from_info(e) for e in entries]
        return [r for r in refs if r is not None]

    def _extract_flat(self, url: str, limit: int | None) -> dict:
        extra: dict = {"extract_flat": "in_playlist"}
        if limit:
            extra["playlistend"] = limit
        return self._retry(lambda: self._extract_info(url, extra))

    def fetch_transcript(self, video_id: str, langs: list[str]) -> Transcript | None:
        info = _tag("метаданные", lambda: self._retry(
            lambda: self._extract_info(
                f"https://www.youtube.com/watch?v={video_id}", {}
            )
        ))
        if not info:
            return None
        manual = info.get("subtitles") or {}
        auto = info.get("automatic_captions") or {}
        chosen = choose_track(manual, auto, langs)
        if chosen is None:
            return None
        lang, formats = chosen
        url = pick_caption_url(formats)
        if not url:
            return None
        translated = is_translated_caption(url)
        if translated:
            # YouTube rate-limits translated tracks hard: pace them globally.
            self.rate_limiter.hold(self.sleep_subtitles)
        raw = _tag("субтитры", lambda: self._retry(lambda: self._fetch_caption(url)))
        if url.endswith(".json3") or raw.lstrip().startswith("{"):
            segments = parse_json3(raw)
        elif "-->" in raw and raw.lstrip().startswith("WEBVTT"):
            segments = parse_vtt(raw)
        elif "-->" in raw:
            segments = parse_srt(raw)
        else:
            return None
        ref = self._ref_from_info(info) or VideoRef(
            id=video_id, title=video_id,
            url=f"https://www.youtube.com/watch?v={video_id}",
        )
        return Transcript(video_ref=ref, lang=lang, segments=segments,
                          translated=translated)


class _SilentLogger:
    def debug(self, *args, **kwargs):  # noqa: D102
        pass

    def info(self, *args, **kwargs):  # noqa: D102
        pass

    def warning(self, *args, **kwargs):  # noqa: D102
        pass

    def error(self, *args, **kwargs):  # noqa: D102
        pass


# --- CLI -------------------------------------------------------------------

def _positive_int(raw: str) -> int:
    try:
        value = int(raw)
    except ValueError:
        raise ArgumentTypeError(f"invalid int value: {raw!r}")
    if value < 1:
        raise ArgumentTypeError("--workers must be >= 1")
    return value


def _curl_cffi_available() -> bool:
    import importlib.util

    return importlib.util.find_spec("curl_cffi") is not None


def resolve_impersonate(value: str | None) -> str | None:
    """Resolve --impersonate; the "auto" default enables it when possible."""
    if value is None or value == "auto":
        return "chrome" if _curl_cffi_available() else None
    return value or None


def build_parser() -> ArgumentParser:
    parser = ArgumentParser(
        prog="ytsubs",
        description="Export YouTube text transcripts (subtitles) without video.",
    )
    parser.add_argument("--channel", action="append", default=[], metavar="URL",
                        help="channel URL (e.g. https://www.youtube.com/@name)")
    parser.add_argument("--playlist", action="append", default=[], metavar="URL",
                        help="playlist URL")
    parser.add_argument("--video", action="append", default=[], metavar="URL",
                        help="single video URL")
    parser.add_argument("--langs", default="ru,en",
                        help="comma-separated language priority (default: ru,en)")
    parser.add_argument("--format", choices=FORMATS, default="txt",
                        help="output format (default: txt)")
    parser.add_argument("--out", default="out", metavar="DIR",
                        help="output directory (default: out)")
    parser.add_argument("--limit", type=int, default=None, metavar="N",
                        help="maximum number of videos")
    parser.add_argument("--cookies", default=None, metavar="FILE",
                        help="cookies file passed to yt-dlp")
    parser.add_argument("--cookies-from-browser", dest="cookies_from_browser",
                        default=None, metavar="NAME",
                        help="read cookies from an installed browser "
                             "(chrome, edge, firefox, ...); log in to YouTube there")
    parser.add_argument("--workers", type=_positive_int, default=4, metavar="N",
                        help="parallel download workers (default: 4, use 1 for sequential)")
    parser.add_argument("--min-interval", type=float, default=1.0, metavar="SEC",
                        help="minimum delay between network requests (default: 1.0)")
    parser.add_argument("--sleep-subtitles", type=float, default=60.0, metavar="SEC",
                        help="pause before each auto-translated subtitle, which YouTube "
                             "rate-limits with HTTP 429 (default: 60, 0 disables)")
    parser.add_argument("--impersonate", default="auto", metavar="CLIENT",
                        help="TLS impersonation profile for yt-dlp/curl_cffi "
                             "(default: auto = chrome when curl_cffi is installed; "
                             "pass an empty string to disable)")
    parser.add_argument("--proxy", default=None, metavar="URL",
                        help="route yt-dlp requests through a proxy "
                             "(e.g. socks5://host:1080 or http://host:8080); "
                             "the reliable escape from per-IP rate limiting")
    return parser


def _make_default_factory(min_interval: float, sleep_subtitles: float,
                          impersonate: str | None, proxy: str | None = None,
                          cookies_from_browser: str | None = None):
    return lambda cookies: YtDlpClient(cookies=cookies, min_interval=min_interval,
                                       sleep_subtitles=sleep_subtitles,
                                       impersonate=impersonate, proxy=proxy,
                                       cookies_from_browser=cookies_from_browser)


def main(argv: Sequence[str] | None = None,
         client_factory: Callable[[str | None], YouTubeClient] | None = None) -> int:
    args = build_parser().parse_args(argv)
    urls = [*args.channel, *args.playlist, *args.video]
    if not urls:
        # Interactive fallback, mirroring the ytsubs.bat UX: same menu, same
        # prompts, then run the same pipeline.
        print("=" * 44)
        print("  ytsubs - YouTube transcript exporter")
        print("=" * 44)
        print()
        print("Type a channel, playlist or video link:")
        print("  https://www.youtube.com/@channel")
        print("  https://www.youtube.com/playlist?list=LIST_ID")
        print("  https://www.youtube.com/watch?v=VIDEO_ID")
        print()
        print("Tip: save this link into YT_SOURCE at the top of this file")
        print("     and next time it starts right away.")
        print()
        url = input("Link (Enter to quit): ").strip()
        if not url:
            return 0
        kind = (
            "--playlist" if "list=" in url
            else "--video" if ("watch?v=" in url or "youtu.be/" in url)
            else "--channel"
        )
        fmt = input("Format [txt/json/srt/vtt] (Enter = txt): ").strip() or args.format
        args = build_parser().parse_args([kind, url, "--format", fmt])
        urls = [*args.channel, *args.playlist, *args.video]
    langs = [lang.strip() for lang in args.langs.split(",") if lang.strip()]
    factory = client_factory or _make_default_factory(
        args.min_interval, args.sleep_subtitles,
        resolve_impersonate(args.impersonate), args.proxy,
        args.cookies_from_browser)
    try:
        client = factory(args.cookies)
        videos = collect_videos(client, urls, limit=args.limit)
    except Exception as exc:  # noqa: BLE001 - source-level failure is fatal
        print(f"error: failed to list videos: {exc}", file=sys.stderr)
        return 1
    print(f"Found {len(videos)} video(s); langs={','.join(langs)}; "
          f"format={args.format}; workers={args.workers}")
    out_dir = Path(args.out)
    if args.channel and not args.video and not args.playlist and args.channel:
        out_dir = out_dir / channel_slug(args.channel[0])
    elif args.video and not args.channel and not args.playlist and videos:
        out_dir = out_dir / slugify(videos[0].title)
    report = export_videos(videos, client, out_dir, langs, args.format,
                           workers=args.workers)
    print(f"Done: ok={report.count('ok')}, no-subs={report.count('no-subs')}, "
          f"skipped={report.count('skipped')}, errors={report.count('error')}, "
          f"aborted={report.count('aborted')}")
    translated = report.count_translated()
    if translated:
        print(f"{translated} дорожек — автоперевод (tlang): именно их YouTube "
              "лимитирует (HTTP 429). Помогают --sleep-subtitles, --cookies, "
              "--impersonate.")
    if report.aborted:
        print("YouTube временно ограничил запросы (429 / 'Sign in to confirm').")
        print("Часть видео не обработана. Повторный запуск продолжит с места "
              "остановки: готовые файлы пропускаются.")
        print("Что помогает: --cookies cookies.txt, --impersonate chrome, "
              "--sleep-subtitles 120, --workers 1 и запуск позже.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
