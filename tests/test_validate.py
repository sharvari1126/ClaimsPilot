from claimspilot.models import AuditStatus as S
from claimspilot.validate import validate_memo

ALLOWED = '[T1] {"paid": 29000, "expected_payout": 28000, "delta": 1000} E1 Driving under the influence 2026-03-10'


def v(memo, status=S.OVERPAID, audit=True, n=1):
    return validate_memo(memo, ALLOWED, n, status, audit)


def test_clean_memo_passes():
    assert v("Overpaid: paid 29,000 vs 28,000, difference 1,000 [T1].") == []


def test_invented_number_is_rejected():
    assert any("4500" in i for i in v("Overpaid by 4,500 [T1]."))


def test_small_numbers_are_free():
    assert v("Overpaid, 2 mistakes found [T1].") == []


def test_unknown_clause_is_rejected_but_known_clause_passes():
    assert any("S9" in i for i in v("Overpaid; see S9 [T1]."))
    assert v("Overpaid; see E1 [T1].") == []


def test_citation_must_exist():
    assert any("[T3]" in i for i in v("Overpaid [T3].", n=1))


def test_verdict_must_be_stated():
    assert any("verdict" in i for i in v("The numbers differ by 1,000 [T1]."))


def test_run_audit_is_required():
    assert any("run_audit" in i for i in v("Overpaid [T1].", audit=False))


def test_ids_and_dates_do_not_trip_the_number_check():
    assert v("Overpaid on CLM-014 under POL-1001, loss 2026-03-10 [T1].") == []
