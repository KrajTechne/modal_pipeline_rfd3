"""The MSA client's cache keys and a3m parsing.

Offline: the network path is a thin wrapper around `requests`, while these are
the parts that can be wrong without anything failing. Both correspond to
defects the plan records in Optimus's `mmseqs2.py` -- a colliding cache key and
block parsing that lets one query's hits bleed into another's.
"""

from __future__ import annotations

import pytest

from rfd3_pipeline.msa import (
    cache_key,
    parse_a3m_blocks,
    read_cached_a3m,
    write_cached_a3m,
)


class TestCacheKey:
    def test_the_same_request_gives_the_same_key(self):
        assert cache_key("MKV", use_env=True, use_filter=True) == cache_key(
            "MKV", use_env=True, use_filter=True
        )

    def test_a_different_sequence_gives_a_different_key(self):
        assert cache_key("MKV", use_env=True, use_filter=True) != cache_key(
            "MKW", use_env=True, use_filter=True
        )

    @pytest.mark.parametrize(
        "changed",
        [{"use_env": False}, {"use_filter": False}, {"paired": True}],
    )
    def test_any_option_that_changes_the_answer_changes_the_key(self, changed):
        """Optimus keyed on the caller's directory and the mode, so two
        sequences in one directory collided. Keying on the sequence alone would
        be the same mistake differently: an alignment built against the
        environmental databases would be served for a request without them."""
        base = dict(use_env=True, use_filter=True)
        assert cache_key("MKV", **base) != cache_key("MKV", **{**base, **changed})

    def test_pairing_strategy_only_matters_when_paired(self):
        unpaired = dict(use_env=True, use_filter=True)
        assert cache_key("MKV", **unpaired, pairing_strategy="greedy") == cache_key(
            "MKV", **unpaired, pairing_strategy="complete"
        )
        assert cache_key(
            "MKV", **unpaired, paired=True, pairing_strategy="greedy"
        ) != cache_key("MKV", **unpaired, paired=True, pairing_strategy="complete")

    def test_whitespace_and_case_do_not_fork_the_cache(self):
        assert cache_key(" mkv \n", use_env=True, use_filter=True) == cache_key(
            "MKV", use_env=True, use_filter=True
        )


class TestBlockParsing:
    def test_queries_come_back_in_input_order(self):
        """ColabFold keys blocks by M-index, which need not match file order."""
        text = ">102\nMKW\n>hitB\nMKW\n\x00>101\nMKV\n>hitA\nMKV\n"
        first, second = parse_a3m_blocks(text, 2)
        assert first.startswith(">101")
        assert second.startswith(">102")

    def test_hits_do_not_bleed_between_queries(self):
        text = ">101\nMKV\n>hitA\nMKVA\n\x00>102\nMKW\n>hitB\nMKWB\n"
        first, second = parse_a3m_blocks(text, 2)
        assert "hitB" not in first
        assert "hitA" not in second

    def test_a_query_with_no_block_is_empty_not_missing(self):
        """The list is positional, so a gap has to stay a gap."""
        blocks = parse_a3m_blocks(">101\nMKV\n", 3)
        assert len(blocks) == 3
        assert blocks[0].startswith(">101")
        assert blocks[1] == "" and blocks[2] == ""

    def test_every_block_ends_with_a_newline(self):
        """Without it, joining blocks from several files concatenates the next
        header onto the last sequence."""
        for block in parse_a3m_blocks(">101\nMKV\n>hit\nMKV\x00>102\nMKW", 2):
            assert block.endswith("\n")

    def test_junk_between_blocks_is_ignored(self):
        text = ">101\nMKV\n\x00\x00   \x00not-a-header\n\x00>102\nMKW\n"
        first, second = parse_a3m_blocks(text, 2)
        assert first.startswith(">101")
        assert second.startswith(">102")


class TestCacheIo:
    def test_a_write_can_be_read_back(self, tmp_path):
        write_cached_a3m(tmp_path, "k", ">101\nMKV\n")
        assert read_cached_a3m(tmp_path, "k") == ">101\nMKV\n"

    def test_a_miss_is_none(self, tmp_path):
        assert read_cached_a3m(tmp_path, "absent") is None

    def test_an_empty_file_reads_as_a_miss(self, tmp_path):
        """A truncated write is worse than a miss: it would be folded against."""
        (tmp_path / "k.a3m").write_text("   \n")
        assert read_cached_a3m(tmp_path, "k") is None

    def test_no_partial_files_are_left_behind(self, tmp_path):
        write_cached_a3m(tmp_path, "k", ">101\nMKV\n")
        assert sorted(p.name for p in tmp_path.iterdir()) == ["k.a3m"]

    def test_a_rewrite_replaces_rather_than_appends(self, tmp_path):
        write_cached_a3m(tmp_path, "k", ">101\nOLD\n")
        write_cached_a3m(tmp_path, "k", ">101\nNEW\n")
        assert read_cached_a3m(tmp_path, "k") == ">101\nNEW\n"
