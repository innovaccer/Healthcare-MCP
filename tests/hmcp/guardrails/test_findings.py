"""Tests for findings helpers."""

from hmcp.shared.guardrails.findings import Finding, locate_all, merge_overlaps, redact


def _finding(start, end, category="NAME", source="pattern", value=""):
    return Finding(category=category, start=start, end=end, source=source, value=value)


def test_merge_overlaps_merges_partial_overlap_into_union():
    text = "abcdefghijklmnopqrst"
    early = _finding(0, 5, category="NAME")
    later = _finding(3, 20, category="DATE")

    merged = merge_overlaps([early, later], text=text)

    assert len(merged) == 1
    union = merged[0]
    assert (union.start, union.end) == (0, 20)
    # longest constituent is the later span
    assert union.category == "DATE"
    assert union.source == "pattern"
    assert union.value == text


def test_merge_overlaps_chain_of_three_becomes_one_union():
    first = _finding(0, 5, category="NAME")
    second = _finding(3, 12, category="DATE")
    third = _finding(10, 15, category="SSN")

    merged = merge_overlaps([first, second, third])

    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 15)
    # second span is the longest constituent
    assert merged[0].category == "DATE"


def test_merge_overlaps_tie_keeps_earliest_constituent():
    earliest = _finding(0, 4, category="NAME")
    later = _finding(2, 6, category="DATE")  # same length as earliest

    merged = merge_overlaps([earliest, later])

    assert len(merged) == 1
    assert (merged[0].start, merged[0].end) == (0, 6)
    assert merged[0].category == "NAME"


def test_merge_overlaps_keeps_adjacent_findings():
    first = _finding(0, 5)
    second = _finding(5, 10)
    assert merge_overlaps([second, first]) == [first, second]


def test_merge_overlaps_union_value_is_text_slice():
    text = "abcdefghij"
    merged = merge_overlaps([_finding(0, 3), _finding(2, 5)], text=text)
    assert len(merged) == 1
    assert merged[0].value == "abcde"


def test_redact_merges_overlaps_and_replaces_right_to_left():
    text = "abcdefghij"
    findings = [
        _finding(0, 2, category="NAME"),
        _finding(1, 4, category="DATE"),
        _finding(6, 8, category="SSN"),
    ]
    # [0,2) and [1,4) merge to [0,4) keeping DATE (longest); [6,8) stays separate
    assert redact(text, findings, "[{category}]") == "[DATE]ef[SSN]ij"


def test_redact_partial_overlap_leaves_no_original_characters():
    text = "abcdefghijklmnopqrst"
    findings = [_finding(0, 5, category="NAME"), _finding(3, 20, category="DATE")]

    result = redact(text, findings, "[{category}]")

    assert result == "[DATE]"
    assert not (set(result) & set(text))


def test_locate_all_finds_all_occurrences():
    assert locate_all("ababab", "ab") == [(0, 2), (2, 4), (4, 6)]
    assert locate_all("abc", "") == []
    assert locate_all("abc", "z") == []


def test_finding_repr_hides_value():
    finding = Finding(
        category="NAME", start=0, end=18, source="pattern", value="Testpatient Alpha"
    )
    assert "Testpatient" not in repr(finding)
