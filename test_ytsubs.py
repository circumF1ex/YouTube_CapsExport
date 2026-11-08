"""Tests for ytsubs core. Deterministic, no network, no sleeps."""

import threading

import pytest

from ytsubs import (
    RateLimiter,
    Segment,
    Transcript,
    VideoRef,
    YtDlpClient,
    build_filename,
    build_parser,
    channel_slug,
    channel_tab_url,
    choose_track,
    collect_videos,
    dedupe_rolling,
    export_videos,
    format_transcript,
    is_translated_caption,
    main,
    parse_json3,
    parse_srt,
    parse_vtt,
    pick_caption_url,
    RateLimited,
    resolve_impersonate,
    segments_to_text,
    select_language,
    slugify,
)


class _Completed:
    def __init__(self, value=None, exc=None):
        self._value = value
        self._exc = exc

    def result(self):
        if self._exc is not None:
            raise self._exc
        return self._value


class SyncExecutor:
    """Deterministic stand-in for ThreadPoolExecutor. Runs inline, no threads."""

    def __init__(self):
        self.submitted = 0

    def submit(self, fn, *args, **kwargs):
        self.submitted += 1
        try:
            return _Completed(fn(*args, **kwargs))
        except Exception as exc:  # noqa: BLE001 - mirror future semantics
            return _Completed(exc=exc)

    def shutdown(self, wait=True):
        pass


class FakeClient:
    """In-memory YouTubeClient for tests. No network, no sleeps."""

    def __init__(self, videos_by_url, transcripts=None, fail_ids=()):
        self._videos = videos_by_url
        self._transcripts = transcripts or {}
        self._fail_ids = set(fail_ids)
        self.fetch_calls: list[str] = []
        self.langs_seen: list[tuple] = []

    def iter_videos(self, url, limit=None):
        return iter(self._videos.get(url, []))

    def fetch_transcript(self, video_id, langs):
        self.fetch_calls.append(video_id)
        self.langs_seen.append((video_id, tuple(langs)))
        if video_id in self._fail_ids:
            raise RuntimeError(f"boom:{video_id}")
        return self._transcripts.get(video_id)


class ConcurrentClient:
    """Client that measures real concurrency; optionally syncs on a barrier."""

    def __init__(self, videos, transcript_for=None, barrier=None, fail_ids=(), lang="ru"):
        self._videos = list(videos)
        self._transcript_for = transcript_for or (
            lambda vid: _transcript(vid, lang, f"text-{vid}")
        )
        self._barrier = barrier
        self._fail_ids = set(fail_ids)
        self._lock = threading.Lock()
        self.active = 0
        self.max_active = 0

    def iter_videos(self, url, limit=None):
        return iter(self._videos)

    def fetch_transcript(self, video_id, langs):
        with self._lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        try:
            if self._barrier is not None:
                self._barrier.wait()
            if video_id in self._fail_ids:
                raise RuntimeError(f"boom:{video_id}")
            return self._transcript_for(video_id)
        finally:
            with self._lock:
                self.active -= 1


def _refs(n: int) -> list[VideoRef]:
    return [VideoRef(id=f"v{i}", title=f"Title {i}", url=f"https://youtu.be/v{i}")
            for i in range(n)]


# --- A. Timed-text parsing -------------------------------------------------

def test_parse_vtt_hhmmss_and_mmss_timestamps():
    raw = (
        "WEBVTT\n"
        "Kind: captions\n"
        "Language: ru\n"
        "\n"
        "00:00:01.000 --> 00:00:03.500\n"
        "Привет\n"
        "\n"
        "01:02.000 --> 01:04.000\n"
        "Мир\n"
    )
    segs = parse_vtt(raw)
    assert segs == [
        Segment(start=1.0, end=3.5, text="Привет"),
        Segment(start=62.0, end=64.0, text="Мир"),
    ]


def test_parse_vtt_skips_note_cue_ids_and_blanks():
    raw = (
        "WEBVTT\n"
        "\n"
        "NOTE this is a comment\n"
        "more comment\n"
        "\n"
        "cue-1\n"
        "00:00:05.000 --> 00:00:06.000\n"
        "text one\n"
        "\n"
        "\n"
        "00:00:07.000 --> 00:00:08.000\n"
        "text two\n"
    )
    segs = parse_vtt(raw)
    assert [s.text for s in segs] == ["text one", "text two"]
    assert segs[0].start == 5.0


def test_parse_vtt_joins_multiline_and_strips_inline_tags():
    raw = (
        "WEBVTT\n"
        "\n"
        "00:00:01.000 --> 00:00:04.000\n"
        "<00:00:01.000><c>hello</c>\n"
        "<c>world</c>\n"
    )
    segs = parse_vtt(raw)
    assert len(segs) == 1
    assert segs[0].text == "hello world"


def test_parse_srt_index_commas_and_multiline():
    raw = (
        "1\n"
        "00:00:01,000 --> 00:00:04,000\n"
        "Hello\n"
        "world\n"
        "\n"
        "2\n"
        "00:01:05,500 --> 00:01:07,000\n"
        "Second cue\n"
    )
    segs = parse_srt(raw)
    assert segs == [
        Segment(start=1.0, end=4.0, text="Hello world"),
        Segment(start=65.5, end=67.0, text="Second cue"),
    ]


# --- B. Auto-caption cleanup ----------------------------------------------

def test_dedupe_rolling_collapses_overlapping_prefix_growth():
    segs = [
        Segment(1.0, 3.0, "and so my fellow"),
        Segment(2.0, 4.0, "and so my fellow Americans"),
        Segment(3.0, 5.0, "and so my fellow Americans ask"),
        Segment(4.0, 6.0, "and so my fellow Americans ask not"),
    ]
    assert dedupe_rolling(segs) == [
        Segment(4.0, 6.0, "and so my fellow Americans ask not"),
    ]


def test_dedupe_rolling_removes_exact_repeats_keeps_distinct():
    segs = [
        Segment(1.0, 2.0, "hello"),
        Segment(2.0, 3.0, "hello"),
        Segment(3.0, 4.0, "world"),
    ]
    assert dedupe_rolling(segs) == [
        Segment(1.0, 3.0, "hello"),
        Segment(3.0, 4.0, "world"),
    ]


# --- C. Text assembly ------------------------------------------------------

def test_segments_to_text_single_spaces_no_word_breaks():
    segs = [
        Segment(0.0, 1.0, "Привет,"),
        Segment(1.0, 2.0, "мир!"),
        Segment(2.0, 3.0, "   "),
        Segment(3.0, 4.0, "Как\tдела?"),
    ]
    assert segments_to_text(segs) == "Привет, мир! Как дела?"


def test_segments_to_text_empty_is_empty_string():
    assert segments_to_text([]) == ""
    assert segments_to_text([Segment(0.0, 1.0, "   ")]) == ""


# --- D. Language selection -------------------------------------------------

def test_select_language_priority_order():
    assert select_language(["ru", "en"], ["ru", "en"]) == "ru"
    assert select_language(["en", "ru"], ["en", "ru"]) == "en"


def test_select_language_fallback_to_second_choice():
    assert select_language(["en"], ["ru", "en"]) == "en"


def test_select_language_none_when_no_match():
    assert select_language(["de", "fr"], ["ru", "en"]) is None
    assert select_language([], ["ru"]) is None


def test_select_language_accepts_regional_variants():
    assert select_language(["en-US", "en"], ["en"]) == "en"
    assert select_language(["en-US"], ["en"]) == "en-US"
    assert select_language(["ru-RU"], ["ru", "en"]) == "ru-RU"


def test_select_language_prefers_original_asr_over_translation():
    # YouTube exposes the original ASR track as "<lang>-orig" and the machine
    # translation as "<lang>"; the translated one is what gets HTTP 429.
    assert select_language(["ru", "ru-orig"], ["ru"]) == "ru-orig"
    assert select_language(["ru-orig", "ru"], ["ru"]) == "ru-orig"
    assert select_language(["ru", "ru-orig", "en"], ["en", "ru"]) == "en"


def test_select_language_prefers_original_regional_variant():
    assert select_language(["en-US", "en-US-orig"], ["en"]) == "en-US-orig"


def test_select_language_falls_back_to_translation_without_original():
    assert select_language(["ru"], ["ru"]) == "ru"


# --- G. File names ---------------------------------------------------------

def test_build_filename_basic_slug():
    assert build_filename("v1", "Hello World!", "txt") == "v1_hello-world.txt"


def test_build_filename_strips_forbidden_characters():
    name = build_filename("v1", "a/b:c*d?e", "srt")
    assert name == "v1_a-b-c-d-e.srt"
    assert not set(name) & set('<>:"/\\|?*')


def test_build_filename_transliterates_cyrillic_and_drops_emoji():
    assert build_filename("v1", "Привет мир 🎉", "txt") == "v1_privet-mir.txt"


def test_build_filename_empty_title_falls_back():
    assert build_filename("v1", "", "txt") == "v1_video.txt"
    assert build_filename("v1", "🎉🎉", "txt") == "v1_video.txt"


# --- F. Formatting ---------------------------------------------------------

def _sample_transcript() -> Transcript:
    return Transcript(
        video_ref=VideoRef(id="v1", title="Hello World", url="https://youtu.be/v1"),
        lang="ru",
        segments=[
            Segment(1.0, 4.0, "Привет мир"),
            Segment(65.5, 67.0, "Вторая реплика 🙃"),
        ],
    )


def test_format_txt_is_plain_text():
    assert format_transcript(_sample_transcript(), "txt") == "Привет мир Вторая реплика 🙃"


def test_format_json_structure_and_unicode():
    import json

    data = json.loads(format_transcript(_sample_transcript(), "json"))
    assert data["id"] == "v1"
    assert data["title"] == "Hello World"
    assert data["lang"] == "ru"
    assert data["segments"][0] == {"start": 1.0, "end": 4.0, "text": "Привет мир"}
    assert data["segments"][1]["text"] == "Вторая реплика 🙃"
    assert "🙃" in format_transcript(_sample_transcript(), "json")


def test_format_srt_numbering_and_timestamps():
    out = format_transcript(_sample_transcript(), "srt")
    assert out == (
        "1\n"
        "00:00:01,000 --> 00:00:04,000\n"
        "Привет мир\n"
        "\n"
        "2\n"
        "00:01:05,500 --> 00:01:07,000\n"
        "Вторая реплика 🙃\n"
    )


def test_format_vtt_header_and_timestamps():
    out = format_transcript(_sample_transcript(), "vtt")
    assert out.startswith("WEBVTT\n\n")
    assert "00:00:01.000 --> 00:00:04.000\nПривет мир" in out
    assert "00:01:05.500 --> 00:01:07.000\nВторая реплика 🙃" in out


# --- E. Collecting videos --------------------------------------------------

def test_collect_videos_applies_limit():
    videos = _refs(5)
    client = FakeClient({"a": videos})
    got = collect_videos(client, ["a"], limit=3)
    assert [v.id for v in got] == ["v0", "v1", "v2"]


def test_collect_videos_dedupes_by_id_across_sources():
    videos = _refs(5)
    client = FakeClient({"a": videos, "b": [videos[2], videos[3], videos[4]]})
    got = collect_videos(client, ["a", "b"])
    assert [v.id for v in got] == ["v0", "v1", "v2", "v3", "v4"]


def test_collect_videos_no_limit_returns_all():
    client = FakeClient({"a": _refs(3)})
    assert [v.id for v in collect_videos(client, ["a"])] == ["v0", "v1", "v2"]


# --- Export engine (sequential) --------------------------------------------

def _transcript(video_id: str, lang: str, text: str) -> Transcript:
    ref = VideoRef(id=video_id, title=f"Title {video_id}", url=f"https://youtu.be/{video_id}")
    return Transcript(video_ref=ref, lang=lang, segments=[Segment(0.0, 1.0, text)])


def _sync_factory(workers):
    return SyncExecutor()


def test_export_writes_file_and_reports_ok(tmp_path):
    client = FakeClient({"a": _refs(1)}, {"v0": _transcript("v0", "ru", "Привет мир")})
    report = export_videos(_refs(1), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.total == 1
    assert report.results[0].status == "ok"
    assert report.results[0].lang == "ru"
    assert (tmp_path / "v0_title-0.txt").read_text(encoding="utf-8") == "Привет мир"


def test_export_missing_subtitles_is_noop(tmp_path):
    client = FakeClient({"a": _refs(2)}, {})
    report = export_videos(_refs(2), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert [r.status for r in report.results] == ["no-subs", "no-subs"]
    assert list(tmp_path.iterdir()) == []


def test_export_one_error_does_not_stop_others(tmp_path):
    client = FakeClient({"a": _refs(3)},
                        {"v0": _transcript("v0", "ru", "a"),
                         "v2": _transcript("v2", "ru", "c")},
                        fail_ids={"v1"})
    report = export_videos(_refs(3), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert [r.status for r in report.results] == ["ok", "error", "ok"]
    assert sorted(p.name for p in tmp_path.iterdir()) == ["v0_title-0.txt", "v2_title-2.txt"]


def test_export_resumes_without_refetching(tmp_path):
    (tmp_path / "v0_title-0.txt").write_text("old", encoding="utf-8")
    client = FakeClient({"a": _refs(1)}, {"v0": _transcript("v0", "ru", "new")})
    report = export_videos(_refs(1), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.results[0].status == "skipped"
    assert client.fetch_calls == []
    assert (tmp_path / "v0_title-0.txt").read_text(encoding="utf-8") == "old"


def test_export_resumes_despite_localized_title(tmp_path):
    # YouTube localizes titles, so the same video can hash to another file name
    # between runs; it must still count as already exported.
    (tmp_path / "v0_inogda-zagolovok.txt").write_text("old", encoding="utf-8")
    client = FakeClient({"a": _refs(1)}, {"v0": _transcript("v0", "ru", "new")})
    report = export_videos(_refs(1), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.results[0].status == "skipped"
    assert client.fetch_calls == []


def test_export_does_not_treat_other_format_as_done(tmp_path):
    (tmp_path / "v0_title-0.srt").write_text("old", encoding="utf-8")
    client = FakeClient({"a": _refs(1)}, {"v0": _transcript("v0", "ru", "new")})
    report = export_videos(_refs(1), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.results[0].status == "ok"
    assert client.fetch_calls == ["v0"]


def test_export_resume_does_not_match_other_video_ids(tmp_path):
    (tmp_path / "v1_title-1.txt").write_text("other", encoding="utf-8")
    client = FakeClient({"a": _refs(2)}, {"v0": _transcript("v0", "ru", "new")})
    report = export_videos(_refs(2), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert {r.video_id: r.status for r in report.results} == {"v0": "ok", "v1": "skipped"}


def test_export_progress_marks_resumed_videos(tmp_path):
    (tmp_path / "v0_title-0.txt").write_text("old", encoding="utf-8")
    lines: list[str] = []
    export_videos(_refs(1), FakeClient({"a": _refs(1)}, {}), tmp_path, ["ru"], "txt",
                  executor_factory=_sync_factory, logger=lines.append)
    assert lines[0] == "[1/1] v0 Title 0 — - (уже выгружено)"


def test_export_empty_video_list_is_noop(tmp_path):
    report = export_videos([], FakeClient({}), tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.total == 0
    assert report.results == []


def test_export_progress_messages_and_counter(tmp_path):
    client = FakeClient({"a": _refs(3)},
                        {"v0": _transcript("v0", "ru", "a")}, fail_ids={"v1"})
    lines: list[str] = []
    report = export_videos(_refs(3), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lines.append)
    assert report.processed == report.total == 3
    assert lines[0].startswith("[1/3] v0 Title 0 — ru (ok)")
    assert "ошибка" in lines[1]
    assert "нет субтитров" in lines[2]


# --- H. Parallelism --------------------------------------------------------

def test_parallel_workers_reach_barrier(tmp_path):
    barrier = threading.Barrier(3, timeout=5)
    client = ConcurrentClient(_refs(3), barrier=barrier)
    report = export_videos(_refs(3), client, tmp_path, ["ru"], "txt", workers=3,
                           logger=lambda *_: None)
    assert client.max_active == 3
    assert report.count("ok") == 3
    assert report.count("error") == 0


def test_workers_one_is_sequential_and_deterministic(tmp_path):
    client = ConcurrentClient(_refs(4))
    lines: list[str] = []
    report = export_videos(_refs(4), client, tmp_path, ["ru"], "txt", workers=1,
                           logger=lines.append)
    assert client.max_active == 1
    assert [r.video_id for r in report.results] == ["v0", "v1", "v2", "v3"]
    assert lines == [f"[{i + 1}/4] v{i} Title {i} — ru (ok)" for i in range(4)]


def test_workers_one_does_not_achieve_concurrency_of_two(tmp_path):
    """Negative control: a 2-party barrier must break under workers=1."""
    barrier = threading.Barrier(2, timeout=0.5)
    client = ConcurrentClient(_refs(2), barrier=barrier)
    report = export_videos(_refs(2), client, tmp_path, ["ru"], "txt", workers=1,
                           logger=lambda *_: None)
    assert client.max_active == 1
    assert report.count("ok") == 0
    assert report.count("error") == 2


def test_error_in_one_worker_keeps_others_parallel(tmp_path):
    client = ConcurrentClient(_refs(4), fail_ids={"v1"})
    report = export_videos(_refs(4), client, tmp_path, ["ru"], "txt", workers=4,
                           logger=lambda *_: None)
    assert report.count("ok") == 3
    assert report.count("error") == 1
    assert len(list(tmp_path.iterdir())) == 3


def test_processed_counter_matches_total_under_load(tmp_path):
    videos = _refs(8)
    barrier = threading.Barrier(4, timeout=5)
    client = ConcurrentClient(videos, barrier=barrier)
    report = export_videos(videos, client, tmp_path, ["ru"], "txt", workers=4,
                           logger=lambda *_: None)
    assert report.processed == report.total == 8
    assert all(r is not None for r in report.results)
    assert report.count("ok") == 8


def test_identical_titles_produce_distinct_files(tmp_path):
    videos = [VideoRef(id=f"id{i}", title="Same Title", url=f"u{i}") for i in range(5)]
    client = ConcurrentClient(
        videos, transcript_for=lambda vid: _transcript(vid, "ru", f"body-{vid}")
    )
    export_videos(videos, client, tmp_path, ["ru"], "txt", workers=4,
                  logger=lambda *_: None)
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [f"id{i}_same-title.txt" for i in range(5)]
    for i in range(5):
        path = tmp_path / f"id{i}_same-title.txt"
        assert path.read_text(encoding="utf-8") == f"body-id{i}"


# --- I. CLI ----------------------------------------------------------------

def test_cli_without_url_returns_fatal_code(capsys):
    rc = main([], client_factory=lambda cookies: FakeClient({}))
    assert rc == 1
    assert "URL" in capsys.readouterr().err


def test_cli_video_end_to_end_srt(tmp_path):
    client = FakeClient({"URL": _refs(1)}, {"v0": _transcript("v0", "ru", "Привет")})
    rc = main(["--video", "URL", "--format", "srt", "--out", str(tmp_path)],
              client_factory=lambda cookies: client)
    assert rc == 0
    out = (tmp_path / "title-0" / "v0_title-0.srt").read_text(encoding="utf-8")
    assert out == "1\n00:00:00,000 --> 00:00:01,000\nПривет\n"


def test_cli_no_subtitles_still_succeeds(tmp_path):
    client = FakeClient({"URL": _refs(2)}, {})
    rc = main(["--video", "URL", "--out", str(tmp_path)], client_factory=lambda c: client)
    assert rc == 0


def test_cli_rejects_non_positive_workers():
    with pytest.raises(SystemExit) as exc:
        main(["--video", "URL", "--workers", "0"], client_factory=lambda c: FakeClient({}))
    assert exc.value.code == 2


def test_cli_limit_and_workers_flags_are_forwarded(tmp_path):
    client = FakeClient({"URL": _refs(5)})
    rc = main(["--channel", "URL", "--limit", "2", "--workers", "1",
               "--langs", "ru,en", "--out", str(tmp_path)],
              client_factory=lambda c: client)
    assert rc == 0
    assert client.langs_seen == [("v0", ("ru", "en")), ("v1", ("ru", "en"))]


def test_cli_unexpected_source_error_is_fatal(tmp_path):
    class Exploding:
        def iter_videos(self, url, limit=None):
            raise RuntimeError("nope")

        def fetch_transcript(self, video_id, langs):
            raise AssertionError("must not be called")

    rc = main(["--video", "URL", "--out", str(tmp_path)],
              client_factory=lambda c: Exploding())
    assert rc == 1


def test_channel_slug_variants():
    assert channel_slug("https://www.youtube.com/@SomeChannel") == "somechannel"
    assert channel_slug("https://www.youtube.com/c/SomeGuy") == "someguy"
    assert channel_slug("https://www.youtube.com/channel/UCabc123") == "ucabc123"
    assert channel_slug("https://www.youtube.com/user/SomeUser") == "someuser"
    assert channel_slug("https://vimeo.com/whatever") == slugify("https://vimeo.com/whatever")


def test_cli_channel_writes_into_channel_folder(tmp_path):
    client = FakeClient({"https://www.youtube.com/@SomeChannel": _refs(1)},
                        {"v0": _transcript("v0", "ru", "Привет")})
    rc = main(["--channel", "https://www.youtube.com/@SomeChannel", "--out", str(tmp_path)],
              client_factory=lambda c: client)
    assert rc == 0
    assert (tmp_path / "somechannel" / "v0_title-0.txt").exists()


def test_cli_video_writes_into_video_folder(tmp_path):
    client = FakeClient({"URL": _refs(1)}, {"v0": _transcript("v0", "ru", "Привет")})
    rc = main(["--video", "URL", "--format", "srt", "--out", str(tmp_path)],
              client_factory=lambda c: client)
    assert rc == 0
    assert (tmp_path / "title-0" / "v0_title-0.srt").exists()


# --- Adapter pure helpers (network isolated) -------------------------------

def test_pick_caption_url_prefers_vtt_then_srt_then_json3():
    formats = [{"ext": "json3", "url": "j"}, {"ext": "vtt", "url": "v"}]
    assert pick_caption_url(formats) == "v"
    assert pick_caption_url([{"ext": "srt", "url": "s"}]) == "s"
    assert pick_caption_url([{"ext": "json3", "url": "j"}]) == "j"


def test_pick_caption_url_prefers_non_translated_format():
    translated = "https://x/t?v=1&lang=en&tlang=ru&fmt=vtt"
    original = "https://x/t?v=1&lang=ru&kind=asr&fmt=vtt"
    assert pick_caption_url([{"ext": "vtt", "url": translated},
                             {"ext": "vtt", "url": original}]) == original


def test_pick_caption_url_falls_back_to_translated_format():
    translated = "https://x/t?v=1&lang=en&tlang=ru&fmt=vtt"
    assert pick_caption_url([{"ext": "vtt", "url": translated}]) == translated
    assert pick_caption_url([]) is None
    assert pick_caption_url([{"ext": "ttml", "url": ""}]) is None


def test_parse_json3_builds_segments_and_skips_blank_events():
    import json

    payload = json.dumps({"events": [
        {"tStartMs": 1000, "dDurationMs": 2000,
         "segs": [{"utf8": "Привет "}, {"utf8": "мир"}]},
        {"tStartMs": 3000, "segs": [{"utf8": "\n"}]},
    ]})
    assert parse_json3(payload) == [Segment(start=1.0, end=3.0, text="Привет мир")]


def test_choose_track_prefers_manual_over_auto():
    manual = {"ru": [{"ext": "vtt", "url": "m-ru"}], "en": []}
    auto = {"en": [{"ext": "vtt", "url": "a-en"}]}
    assert choose_track(manual, auto, ["ru", "en"]) == ("ru", manual["ru"])
    assert choose_track({}, auto, ["ru", "en"]) == ("en", auto["en"])
    assert choose_track(manual, auto, ["de"]) is None


def test_choose_track_prefers_original_auto_over_translated():
    auto = {
        "ru": [{"ext": "json3", "url": "https://x/t?v=1&lang=en&tlang=ru"}],
        "ru-orig": [{"ext": "json3", "url": "https://x/t?v=1&lang=ru&kind=asr"}],
    }
    assert choose_track({}, auto, ["ru"]) == ("ru-orig", auto["ru-orig"])


def test_export_surfaces_error_reason(tmp_path):
    client = ConcurrentClient(_refs(1), fail_ids={"v0"})
    lines: list[str] = []
    export_videos(_refs(1), client, tmp_path, ["ru"], "txt", workers=1,
                  logger=lines.append)
    assert "ошибка" in lines[0]
    assert "boom:v0" in lines[0]


# --- Adapter: channel / playlist expansion (network stubbed) ---------------

def _tab(title, url):
    return {"_type": "playlist", "id": "UC1", "title": title, "webpage_url": url}


class StubYtDlp(YtDlpClient):
    """YtDlpClient with network calls and pacing replaced by test doubles."""

    def __init__(self, responses, captions=None, **kwargs):
        self.clock = FakeTime()
        kwargs.setdefault("rate_limiter", RateLimiter(
            min_interval=0.0, clock=self.clock.clock, sleep=self.clock.sleep))
        super().__init__(**kwargs)
        self.responses = responses
        self.captions = captions or {}
        self.urls: list[str] = []
        self.fetched: list[str] = []

    def _extract_info(self, url, extra):
        self.urls.append(url)
        return self.responses[url]

    def _fetch_caption(self, url):
        self.fetched.append(url)
        return self.captions[url]

    def _extract_info(self, url, extra):
        self.urls.append(url)
        return self.responses[url]


def test_channel_tab_url_picks_videos_by_url_not_localized_title():
    entries = [
        _tab("Канал - Прямые трансляции", "https://x/@c/streams"),
        _tab("Канал - Видео", "https://x/@c/videos"),
        _tab("Канал - Shorts", "https://x/@c/shorts"),
    ]
    assert channel_tab_url(entries) == "https://x/@c/videos"


def test_channel_tab_url_returns_none_for_plain_video_list():
    entries = [{"_type": "url", "id": "v1", "url": "https://youtu.be/v1"}]
    assert channel_tab_url(entries) is None
    assert channel_tab_url([]) is None


def test_iter_videos_expands_bare_channel_into_videos_tab():
    channel = "https://www.youtube.com/@Chan"
    videos_url = channel + "/videos"
    listing = {"_type": "playlist", "id": "UC1", "entries": [
        {"_type": "url", "id": "v1", "title": "One", "url": "https://youtu.be/v1"},
        {"_type": "url", "id": "v2", "title": "Two", "url": "https://youtu.be/v2"},
    ]}
    client = StubYtDlp({videos_url: listing})
    refs = client.iter_videos(channel)
    assert [r.id for r in refs] == ["v1", "v2"]
    assert [r.title for r in refs] == ["One", "Two"]
    assert client.urls == [videos_url]


def test_iter_videos_expands_tab_container_as_fallback():
    featured = "https://www.youtube.com/@Chan/featured"
    container = {"_type": "playlist", "id": "@Chan", "entries": [
        _tab("Chan - Videos", "https://www.youtube.com/@Chan/videos"),
        _tab("Chan - Live", "https://www.youtube.com/@Chan/streams"),
    ]}
    listing = {"_type": "playlist", "id": "UC1", "entries": [
        {"_type": "url", "id": "v1", "title": "One", "url": "https://youtu.be/v1"},
    ]}
    client = StubYtDlp({featured: container,
                        "https://www.youtube.com/@Chan/videos": listing})
    assert [r.id for r in client.iter_videos(featured)] == ["v1"]
    assert client.urls == [featured, "https://www.youtube.com/@Chan/videos"]


def test_normalize_channel_url_variants():
    from ytsubs import normalize_channel_url as norm

    assert norm("https://www.youtube.com/@Chan") == "https://www.youtube.com/@Chan/videos"
    assert norm("https://www.youtube.com/@Chan/") == "https://www.youtube.com/@Chan/videos"
    assert norm("https://youtube.com/@Chan") == "https://youtube.com/@Chan/videos"
    assert norm("https://www.youtube.com/channel/UC1") == "https://www.youtube.com/channel/UC1/videos"
    assert norm("https://www.youtube.com/c/Name") == "https://www.youtube.com/c/Name/videos"
    assert norm("https://www.youtube.com/user/Name") == "https://www.youtube.com/user/Name/videos"

    assert norm("https://www.youtube.com/@Chan/videos") == "https://www.youtube.com/@Chan/videos"
    assert norm("https://www.youtube.com/@Chan/streams") == "https://www.youtube.com/@Chan/streams"
    assert norm("https://www.youtube.com/watch?v=abc") == "https://www.youtube.com/watch?v=abc"
    assert norm("https://www.youtube.com/playlist?list=PL1") == "https://www.youtube.com/playlist?list=PL1"
    assert norm("https://example.com/@Chan") == "https://example.com/@Chan"


def test_iter_videos_does_not_refetch_plain_playlist():
    url = "https://www.youtube.com/playlist?list=PL1"
    listing = {"_type": "playlist", "id": "PL1", "entries": [
        {"_type": "url", "id": "v1", "title": "One", "url": "https://youtu.be/v1"},
    ]}
    client = StubYtDlp({url: listing})
    assert [r.id for r in client.iter_videos(url)] == ["v1"]
    assert client.urls == [url]


def test_iter_videos_single_video_source():
    url = "https://www.youtube.com/watch?v=v9"
    info = {"_type": "video", "id": "v9", "title": "Nine", "webpage_url": url}
    client = StubYtDlp({url: info})
    refs = client.iter_videos(url)
    assert [r.id for r in refs] == ["v9"]
    assert refs[0].url == url


# --- Rate limiting / retry (global, no real sleeps) -------------------------

class FakeTime:
    """Deterministic clock: `sleep` advances time instead of blocking."""

    def __init__(self):
        self.now = 0.0
        self.sleeps: list[float] = []

    def clock(self):
        return self.now

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds


def test_rate_limiter_enforces_min_interval():
    ft = FakeTime()
    rl = RateLimiter(min_interval=1.0, clock=ft.clock, sleep=ft.sleep)
    rl.wait()
    rl.wait()
    assert ft.sleeps == [1.0]


def test_rate_limiter_penalty_blocks_all_callers():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    rl.penalize(5.0)
    rl.wait()
    assert ft.now == 5.0


def test_rate_limiter_penalty_is_taken_as_maximum_not_sum():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    rl.penalize(5.0)
    ft.now = 1.0
    rl.penalize(5.0)
    rl.wait()
    assert ft.now == 6.0


def test_rate_limiter_grows_interval_on_penalty_and_relaxes_on_success():
    ft = FakeTime()
    rl = RateLimiter(min_interval=1.0, max_interval=8.0, clock=ft.clock, sleep=ft.sleep)
    rl.penalize(4.0)
    assert rl.interval == 4.0
    rl.penalize(4.0)
    assert rl.interval == 8.0
    rl.relax()
    rl.relax()
    assert rl.interval == 2.0


def test_retry_applies_exponential_global_backoff():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    client = YtDlpClient(retries=3, base_delay=1.0, max_delay=10.0, rate_limiter=rl)
    calls: list[int] = []

    def fn():
        calls.append(1)
        if len(calls) < 3:
            raise RuntimeError("connection reset by peer")
        return "ok"

    assert client._retry(fn) == "ok"
    assert ft.sleeps == [1.0, 2.0]


def test_retry_honours_retry_after_header():
    client = YtDlpClient()
    exc = RuntimeError("429")
    exc.headers = {"Retry-After": "7"}
    assert client._next_delay(exc, 1.0) == 7.0
    assert client._next_delay(RuntimeError("429"), 1.0) == 1.0


def test_retry_is_shared_between_client_calls():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    client = YtDlpClient(retries=1, base_delay=2.0, max_delay=10.0, rate_limiter=rl)

    def fail():
        raise RuntimeError("429")

    with pytest.raises(RuntimeError):
        client._retry(fail)
    # the same limiter is shared, so a second call still sees the penalty
    assert rl is client.rate_limiter
    assert rl.interval >= 2.0


def test_cli_parses_min_interval():
    args = build_parser().parse_args(["--video", "URL", "--min-interval", "2.5"])
    assert args.min_interval == 2.5
    assert build_parser().parse_args(["--video", "URL"]).min_interval > 0


# --- Throttling circuit breaker --------------------------------------------

def test_throttle_errors_get_at_most_one_retry():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    client = YtDlpClient(retries=3, base_delay=1.0, max_delay=10.0, rate_limiter=rl)
    calls: list[int] = []

    def fn():
        calls.append(1)
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    with pytest.raises(RuntimeError):
        client._retry(fn)
    assert len(calls) == 2
    assert ft.sleeps == [1.0]


def test_throttle_breaker_trips_across_requests():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    client = YtDlpClient(retries=3, base_delay=1.0, max_delay=10.0,
                         throttle_window=10, throttle_limit=3, rate_limiter=rl)

    def throttled():
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    with pytest.raises(RuntimeError):
        client._retry(throttled)
    with pytest.raises(RateLimited):
        client._retry(throttled)


def test_throttle_breaker_tolerates_occasional_429():
    ft = FakeTime()
    rl = RateLimiter(min_interval=0.0, clock=ft.clock, sleep=ft.sleep)
    client = YtDlpClient(retries=1, base_delay=1.0, max_delay=10.0,
                         throttle_window=20, throttle_limit=6, rate_limiter=rl)

    def throttled():
        raise RuntimeError("HTTP Error 429: Too Many Requests")

    for _ in range(2):
        client._retry(lambda: "ok")
        with pytest.raises(RuntimeError):
            client._retry(throttled)
    assert client.throttle_failures == 4


def test_client_breaker_ignores_ordinary_errors():
    ft = FakeTime()
    client = YtDlpClient(retries=1, rate_limiter=RateLimiter(
        min_interval=0.0, clock=ft.clock, sleep=ft.sleep))

    def boom():
        raise RuntimeError("some other problem")

    with pytest.raises(RuntimeError) as exc:
        client._retry(boom)
    assert not isinstance(exc.value, RateLimited)


class ThrottledClient:
    def __init__(self, videos):
        self._videos = list(videos)
        self.calls: list[str] = []

    def iter_videos(self, url, limit=None):
        return iter(self._videos)

    def fetch_transcript(self, video_id, langs):
        self.calls.append(video_id)
        raise RateLimited("HTTP Error 429: Too Many Requests")


def test_export_aborts_remaining_videos_on_throttling(tmp_path):
    videos = _refs(4)
    client = ThrottledClient(videos)
    report = export_videos(videos, client, tmp_path, ["ru"], "txt", workers=1,
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.aborted is True
    assert report.results[0].status == "error"
    assert [r.status for r in report.results[1:]] == ["aborted"] * 3
    assert client.calls == ["v0"]
    assert list(tmp_path.iterdir()) == []


def test_export_without_throttling_is_not_aborted(tmp_path):
    client = FakeClient({"a": _refs(2)},
                        {"v0": _transcript("v0", "ru", "a")})
    report = export_videos(_refs(2), client, tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.aborted is False
    assert report.count("aborted") == 0


def test_cli_aborts_on_throttling_with_fatal_code(tmp_path, capsys):
    client = ThrottledClient(_refs(2))
    rc = main(["--video", "URL", "--out", str(tmp_path), "--workers", "1"],
              client_factory=lambda c: client)
    assert rc == 1
    assert "ограничил" in capsys.readouterr().out


# --- Auto-translated tracks (the real 429 cause) ---------------------------

def _noop_limiter() -> RateLimiter:
    return RateLimiter(min_interval=0.0, clock=lambda: 0.0, sleep=lambda _s: None)


def _json3(text: str) -> str:
    return ('{"events": [{"tStartMs": 0, "dDurationMs": 1000, '
            f'"segs": [{{"utf8": {text!r}}}]}}]}}').replace("'", '"')


def _caption_info(video_id: str, url: str) -> dict:
    return {
        "id": video_id,
        "title": f"Title {video_id}",
        "webpage_url": f"https://www.youtube.com/watch?v={video_id}",
        "subtitles": {},
        "automatic_captions": {"ru": [{"url": url, "ext": "json3"}]},
    }


def test_is_translated_caption_flags_tlang_urls():
    assert is_translated_caption(
        "https://www.youtube.com/api/timedtext?v=x&lang=en&tlang=ru&fmt=json3") is True
    assert is_translated_caption(
        "https://www.youtube.com/api/timedtext?v=x&lang=en-orig&kind=asr&fmt=json3") is False


def test_fetch_transcript_fetches_caption_and_parses_json3():
    url = "https://www.youtube.com/api/timedtext?v=v1&lang=ru&fmt=json3"
    client = StubYtDlp(
        {"https://www.youtube.com/watch?v=v1": _caption_info("v1", url)},
        captions={url: _json3("Привет")})
    transcript = client.fetch_transcript("v1", ["ru"])
    assert transcript.lang == "ru"
    assert segments_to_text(transcript.segments) == "Привет"
    assert client.fetched == [url]


def test_fetch_transcript_paces_consecutive_translated_captions():
    url = "https://www.youtube.com/api/timedtext?v=v1&lang=en&tlang=ru&fmt=json3"
    client = StubYtDlp(
        {"https://www.youtube.com/watch?v=v1": _caption_info("v1", url)},
        captions={url: _json3("hi")}, sleep_subtitles=60.0)
    first = client.fetch_transcript("v1", ["ru"])
    client.fetch_transcript("v1", ["ru"])
    client.fetch_transcript("v1", ["ru"])
    assert first.translated is True
    # the first needs no wait; the following ones are queued 60 s apart
    assert client.clock.sleeps == [60.0, 60.0]


def test_fetch_transcript_does_not_pace_original_tracks():
    url = "https://www.youtube.com/api/timedtext?v=v1&lang=ru&kind=asr&fmt=json3"
    client = StubYtDlp(
        {"https://www.youtube.com/watch?v=v1": _caption_info("v1", url)},
        captions={url: _json3("hi")}, sleep_subtitles=60.0)
    client.fetch_transcript("v1", ["ru"])
    client.fetch_transcript("v1", ["ru"])
    assert client.clock.sleeps == []


def test_fetch_transcript_tags_metadata_errors():
    client = StubYtDlp({})  # KeyError: no metadata response registered
    with pytest.raises(RuntimeError) as ei:
        client.fetch_transcript("v1", ["ru"])
    assert "[метаданные]" in str(ei.value)


def test_fetch_transcript_tags_caption_errors():
    url = "https://www.youtube.com/api/timedtext?v=v1&lang=ru&fmt=json3"
    client = StubYtDlp(
        {"https://www.youtube.com/watch?v=v1": _caption_info("v1", url)},
        captions={})  # KeyError: caption body not registered
    with pytest.raises(RuntimeError) as ei:
        client.fetch_transcript("v1", ["ru"])
    assert "[субтитры]" in str(ei.value)


def test_fetch_transcript_does_not_wrap_rate_limited():
    class _Limited(StubYtDlp):
        def _extract_info(self, url, extra):
            raise RateLimited("stopped")

    client = _Limited({})
    with pytest.raises(RateLimited):
        client.fetch_transcript("v1", ["ru"])


def test_options_carry_sleep_subtitles_and_impersonation():
    client = YtDlpClient(sleep_subtitles=45.0, impersonate="chrome",
                         rate_limiter=_noop_limiter())
    options = client._options({})
    assert options["sleep_interval_subtitles"] == 45.0
    assert str(options["impersonate"]) == "chrome"


def test_options_without_impersonation_are_left_alone():
    options = YtDlpClient(rate_limiter=_noop_limiter())._options({})
    assert "impersonate" not in options


def test_options_carry_proxy():
    client = YtDlpClient(proxy="http://127.0.0.1:8080",
                         rate_limiter=_noop_limiter())
    assert client._options({})["proxy"] == "http://127.0.0.1:8080"


def test_options_without_proxy_are_left_alone():
    options = YtDlpClient(rate_limiter=_noop_limiter())._options({})
    assert "proxy" not in options


def test_options_never_fail_on_missing_formats():
    # we only want captions, so a video with no downloadable format must not abort
    options = YtDlpClient(rate_limiter=_noop_limiter())._options({})
    assert options["skip_download"] is True
    assert options["ignore_no_formats_error"] is True


def test_cli_parses_proxy():
    parser = build_parser()
    assert parser.parse_args(["--video", "URL"]).proxy is None
    args = parser.parse_args(["--video", "URL", "--proxy", "socks5://h:1080"])
    assert args.proxy == "socks5://h:1080"


def test_cookie_jar_from_browser_is_loaded_once(monkeypatch):
    calls = []

    def fake_extract(browser):
        calls.append(browser)
        return "JAR"

    monkeypatch.setattr("ytsubs._extract_browser_cookies", fake_extract)
    client = YtDlpClient(cookies_from_browser="chrome",
                         rate_limiter=_noop_limiter())
    assert client._cookie_jar() == "JAR"
    assert client._cookie_jar() == "JAR"
    assert calls == ["chrome"]


def test_cookie_jar_is_none_without_any_cookie_source():
    client = YtDlpClient(rate_limiter=_noop_limiter())
    assert client._cookie_jar() is None


def test_cli_parses_cookies_from_browser():
    parser = build_parser()
    assert parser.parse_args(["--video", "URL"]).cookies_from_browser is None
    args = parser.parse_args(["--video", "URL",
                              "--cookies-from-browser", "edge"])
    assert args.cookies_from_browser == "edge"


def test_rate_limiter_hold_queues_slots_without_changing_interval():
    ft = FakeTime()
    limiter = RateLimiter(min_interval=1.0, clock=ft.clock, sleep=ft.sleep)
    for _ in range(3):
        limiter.hold(60.0)
        limiter.wait()
    assert ft.sleeps == [60.0, 60.0]
    assert ft.now == 120.0
    assert limiter.interval == 1.0


def test_rate_limiter_hold_of_zero_is_a_noop():
    ft = FakeTime()
    limiter = RateLimiter(min_interval=0.5, clock=ft.clock, sleep=ft.sleep)
    limiter.hold(0.0)
    limiter.wait()
    assert ft.sleeps == []


def test_resolve_impersonate(monkeypatch):
    monkeypatch.setattr("ytsubs._curl_cffi_available", lambda: True)
    assert resolve_impersonate("auto") == "chrome"
    assert resolve_impersonate("firefox") == "firefox"
    assert resolve_impersonate("") is None
    monkeypatch.setattr("ytsubs._curl_cffi_available", lambda: False)
    assert resolve_impersonate("auto") is None


def test_cli_parses_sleep_subtitles_and_impersonate():
    parser = build_parser()
    defaults = parser.parse_args(["--video", "URL"])
    assert defaults.sleep_subtitles == 60.0
    assert defaults.impersonate == "auto"
    args = parser.parse_args(["--video", "URL", "--sleep-subtitles", "0",
                              "--impersonate", "firefox"])
    assert args.sleep_subtitles == 0.0
    assert args.impersonate == "firefox"


def test_export_marks_translated_tracks(tmp_path):
    lines: list[str] = []
    transcript = Transcript(video_ref=_refs(1)[0], lang="ru",
                            segments=[Segment(0.0, 1.0, "hi")], translated=True)
    report = export_videos(_refs(1), FakeClient({}, {"v0": transcript}), tmp_path,
                           ["ru"], "txt", executor_factory=_sync_factory,
                           logger=lines.append)
    assert report.results[0].translated is True
    assert report.count_translated() == 1
    assert "автоперевод" in lines[0]


def test_export_original_track_is_not_marked(tmp_path):
    report = export_videos(_refs(1),
                           FakeClient({}, {"v0": _transcript("v0", "ru", "hi")}),
                           tmp_path, ["ru"], "txt",
                           executor_factory=_sync_factory, logger=lambda *_: None)
    assert report.results[0].translated is False
    assert report.count_translated() == 0


def test_cli_summary_reports_translated_tracks(tmp_path, capsys):
    transcript = Transcript(video_ref=_refs(1)[0], lang="ru",
                            segments=[Segment(0.0, 1.0, "hi")], translated=True)
    client = FakeClient({"URL": _refs(1)}, {"v0": transcript})
    rc = main(["--video", "URL", "--out", str(tmp_path)],
              client_factory=lambda c: client)
    assert rc == 0
    assert "автоперевод" in capsys.readouterr().out
