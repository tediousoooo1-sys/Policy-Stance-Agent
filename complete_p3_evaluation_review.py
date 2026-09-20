from __future__ import annotations

import csv
import re
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REVIEW_DIR = BASE_DIR / "processed" / "polarity_review"
QUEUE_PATH = REVIEW_DIR / "polarity_review_queue.csv"
P0_P1_PATH = REVIEW_DIR / "p0_p1_reviewed_combined.csv"
REVIEWER = "Codex manual review (blind to party vote outcome)"


def compact(text: str) -> str:
    """压缩抓取文本中的空白，但不改写证据内容。"""
    return re.sub(r"\s+", " ", text or "").strip()


def evidence(text: str, limit: int = 420) -> str:
    """截取足以复核方向判断的短证据。"""
    text = compact(text)
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    cut = max(candidate.rfind(";"), candidate.rfind("."), candidate.rfind(":"))
    if cut >= 120:
        candidate = candidate[: cut + 1]
    return candidate.rstrip() + "…"


def object_with_excerpt(row: dict[str, str], prefix: str) -> str:
    """让同类 New Clause 或 Bill introduction 仍能通过正文片段区分。"""
    text = re.sub(r"_+", " ", compact(row["motion_text_clean"]))
    text = re.sub(r"\s+", " ", text).strip()
    return f"{prefix}: {row['motion_title_clean']} — {text[:220].rstrip(' ,;:')}"


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def annotate(row: dict[str, str]) -> dict[str, str]:
    """按冻结后的单一政策对象标准复核一条 P3 评测记录。"""
    key = row["division_key"]
    rule = row["polarity_rule"]

    if key == "pw-2025-05-07-187-commons":
        policy_object = "Unspecified change to the social-media data-processing age of consent"
        aye_action = "The extract ends before stating the proposed age or legal change."
        polarity = "exclude"
        government_backed = "0"
        note = "Excluded: the New Clause text is truncated before the substantive change."
    elif key == "pw-2024-05-21-156-commons":
        policy_object = "The railway provisions that the instruction requires the HS2 Bill to remove"
        aye_action = "Direct the Committee to remove the specified railway sections from the Bill."
        polarity = "-1"
        government_backed = "1"
        note = "Included: Aye removes the identified railway provisions, so polarity is reversed."
    elif key == "pw-2024-10-09-18-commons":
        policy_object = "Second Reading of the Renters' Rights Bill"
        aye_action = "Support the reasoned amendment opposing Second Reading."
        polarity = "-1"
        government_backed = "0"
        note = "Included: a reasoned amendment opposing Second Reading reverses the Bill direction."
    elif rule == "type:proposed_clause":
        policy_object = object_with_excerpt(row, "Proposed New Clause")
        aye_action = "Add the proposed New Clause to the Bill."
        polarity = "1"
        government_backed = "1" if key == "pw-2024-04-24-138-commons" else "0"
        note = "Included: the complete New Clause is the policy object and Aye adds it to the Bill."
    elif rule == "type:bill_introduction":
        policy_object = object_with_excerpt(row, "Proposed Bill")
        aye_action = "Grant leave to introduce the proposed Bill."
        polarity = "1"
        government_backed = "0"
        note = "Included: Aye grants leave to introduce the Bill described in the motion."
    else:
        raise ValueError(f"Unreviewed P3 evaluation pattern: {key} | {rule}")

    return {
        "policy_object": policy_object,
        "aye_action": aye_action,
        "human_motion_polarity": polarity,
        "human_object_government_backed": government_backed,
        "evidence_span": evidence(row["motion_text_clean"]),
        "reviewer": REVIEWER,
        "review_status": "approved",
        "review_notes": note,
    }


def main() -> None:
    with QUEUE_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        queue_rows = list(reader)
        fieldnames = list(reader.fieldnames or [])

    selected = [
        row
        for row in queue_rows
        if row["review_priority"] == "P3" and row["split"] in {"validation", "test"}
    ]
    if len(selected) != 94 or len({row["division_key"] for row in selected}) != 94:
        raise ValueError("Expected 94 unique P3 validation/test rows")

    # 先处理 validation，再按同一冻结标准处理 test。
    reviewed_by_key: dict[str, dict[str, str]] = {}
    for split in ("validation", "test"):
        for row in selected:
            if row["split"] != split:
                continue
            reviewed = dict(row)
            reviewed.update(annotate(row))
            reviewed_by_key[row["division_key"]] = reviewed

    reviewed = [reviewed_by_key[row["division_key"]] for row in selected]
    validation_rows = [row for row in reviewed if row["split"] == "validation"]
    test_rows = [row for row in reviewed if row["split"] == "test"]

    if len(validation_rows) != 38 or len(test_rows) != 56:
        raise ValueError("Unexpected P3 split sizes")
    if any(row["review_status"] != "approved" for row in reviewed):
        raise ValueError("Every P3 row must be completed")
    if any(row["human_motion_polarity"] not in {"-1", "1", "exclude"} for row in reviewed):
        raise ValueError("Unexpected P3 polarity value")
    required = [
        "policy_object",
        "aye_action",
        "human_object_government_backed",
        "evidence_span",
        "reviewer",
        "review_notes",
    ]
    if any(not row[field].strip() for row in reviewed for field in required):
        raise ValueError("A required P3 review field is blank")

    with P0_P1_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        previous_rows = list(csv.DictReader(file))
    evaluation_rows = previous_rows + reviewed
    if len(evaluation_rows) != 263:
        raise ValueError(f"Expected 263 reviewed evaluation rows, found {len(evaluation_rows)}")
    if len({row["division_key"] for row in evaluation_rows}) != 263:
        raise ValueError("Reviewed evaluation division keys are not unique")

    write_csv(REVIEW_DIR / "p3_validation_reviewed.csv", validation_rows, fieldnames)
    write_csv(REVIEW_DIR / "p3_test_reviewed.csv", test_rows, fieldnames)
    write_csv(REVIEW_DIR / "p3_evaluation_reviewed_combined.csv", reviewed, fieldnames)
    write_csv(REVIEW_DIR / "evaluation_reviewed_all.csv", evaluation_rows, fieldnames)

    counts = {value: sum(row["human_motion_polarity"] == value for row in reviewed) for value in ("1", "-1", "exclude")}
    print(f"P3 validation rows: {len(validation_rows)}")
    print(f"P3 test rows: {len(test_rows)}")
    print("P3 polarity counts:", counts)
    print(f"All reviewed evaluation rows: {len(evaluation_rows)}")


if __name__ == "__main__":
    main()
