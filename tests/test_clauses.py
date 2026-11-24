from claimspilot.clauses import get_clause, load_clauses


def test_clauses_are_parsed_with_ids():
    c = load_clauses()
    assert {"C1", "C2", "E1", "E4", "N1", "S1", "S4", "F1", "F4"} <= set(c)
    assert "influence" in c["E1"].lower()


def test_multi_line_clause_includes_bullets():
    assert "Rubber" in get_clause("S1") and "Metal parts" in get_clause("S1")


def test_engine_rule_ids_resolve_to_clauses():
    assert get_clause("C2a") == get_clause("C2")
    assert get_clause("D1") == get_clause("S2")
    assert get_clause("Z9") is None
