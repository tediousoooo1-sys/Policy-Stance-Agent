from __future__ import annotations

import csv
import re
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "processed" / "polarity_review" / "polarity_review_queue.csv"
OUTPUT_DIR = BASE_DIR / "processed" / "polarity_review"
REVIEWER = "Codex manual review (blind to party vote outcome)"


def compact(text: str) -> str:
    """压缩网页抓取产生的空白和占位符，保留原始措辞。"""
    text = re.sub(r"\s+", " ", text or "").strip()
    return text


def evidence(text: str, limit: int = 420) -> str:
    """保留足以复核判断的短原文，不复制整段议案。"""
    text = compact(text)
    if len(text) <= limit:
        return text
    candidate = text[:limit]
    cut = max(candidate.rfind(";"), candidate.rfind("."), candidate.rfind(":"))
    if cut >= 120:
        candidate = candidate[: cut + 1]
    return candidate.rstrip() + "…"


def amendment_object(row: dict[str, str]) -> str:
    """让同一 Clause 下的不同 amendment 仍能通过正文片段区分。"""
    text = compact(row["motion_text_clean"])
    text = re.sub(r"_+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    snippet = text[:220].rstrip(" ,;:")
    return f"Proposed amendment to {row['motion_title_clean']}: {snippet}"


def annotation(
    *,
    policy_object: str,
    aye_action: str,
    polarity: str,
    government_backed: str = "unknown",
    note: str,
) -> dict[str, str]:
    return {
        "policy_object": policy_object,
        "aye_action": aye_action,
        "human_motion_polarity": polarity,
        "human_object_government_backed": government_backed,
        "reviewer": REVIEWER,
        "review_status": "approved",
        "review_notes": note,
    }


def include(
    policy_object: str,
    aye_action: str,
    polarity: int,
    government_backed: str = "unknown",
    note: str = "Included: single policy object and clear Aye effect.",
) -> dict[str, str]:
    return annotation(
        policy_object=policy_object,
        aye_action=aye_action,
        polarity=str(polarity),
        government_backed=government_backed,
        note=note,
    )


def exclude(
    policy_object: str,
    aye_action: str,
    reason: str,
    government_backed: str = "unknown",
) -> dict[str, str]:
    return annotation(
        policy_object=policy_object,
        aye_action=aye_action,
        polarity="exclude",
        government_backed=government_backed,
        note=f"Excluded: {reason}",
    )


# 这些议案不能只靠“amendment 默认 +1”处理。字典只依据议案正文，未读取政党投票结果。
SPECIAL: dict[str, dict[str, str]] = {
    # P0：测试集。规则先在 P1 冻结后，再应用到这些记录。
    "pw-2025-11-13-348-commons": exclude(
        "Lords amendment 2 and proposed amendment (a)",
        "The extract only says amendment (a) was proposed.",
        "the substantive wording of both amendments is missing, so the policy object and Aye effect cannot be recovered",
        "1",
    ),
    "pw-2025-10-28-330-commons": include(
        "Publication of government records about the China spying case",
        "Require release of meeting minutes and related correspondence.",
        1,
        "0",
    ),
    "pw-2025-12-02-376-commons": include(
        "The proposed alcohol duty rate schedule",
        "Adopt the replacement alcohol duty schedules.",
        1,
        "1",
    ),
    "pw-2025-02-26-107-commons": exclude(
        "Multiple family-business tax, employment and regulatory policies",
        "Oppose several government measures and request several reversals.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2025-10-28-329-commons": include(
        "Abolition of stamp duty land tax on primary residences bought by UK residents",
        "Call for public-spending reductions to fund abolition.",
        1,
        "0",
    ),
    "pw-2025-12-02-371-commons": include(
        "Future-year income-tax savings rates",
        "Authorise future Finance Bill provisions setting savings rates.",
        1,
        "1",
    ),
    "pw-2025-11-04-338-commons": exclude(
        "Multiple welfare eligibility and assessment reforms",
        "Call for several separate welfare restrictions and reforms.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2026-03-24-460-commons": exclude(
        "Multiple defence, Northern Ireland, Diego Garcia and spending policies",
        "Support and oppose several separate defence-related actions.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2025-12-02-375-commons": include(
        "Inheritance-tax treatment of pension death benefits",
        "Authorise future Finance Bill provisions for this treatment.",
        1,
        "1",
    ),
    "pw-2026-01-28-422-commons": include(
        "UK cession of sovereignty over the British Indian Ocean Territory",
        "Oppose the cession and call on the Government not to proceed with the related Bill.",
        -1,
        "0",
    ),
    "pw-2025-07-15-269-commons": exclude(
        "Multiple National Insurance, income tax, VAT, property relief and council-tax policies",
        "Oppose or request changes to several separate tax policies.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2026-04-27-510-commons": exclude(
        "Carry-over of the Northern Ireland Troubles Bill",
        "Continue unfinished proceedings into the next parliamentary session.",
        "this is a procedural carry-over vote rather than a vote on the Bill's policy substance",
        "1",
    ),
    "pw-2025-02-11-99-commons": exclude(
        "Commons amendment 2 and replacement amendment (a)",
        "Insist on one amendment and propose another in lieu.",
        "the motion contains more than one amendment action and omits their substantive wording",
        "unknown",
    ),
    "pw-2025-03-19-138-commons": exclude(
        "Winter Fuel Payment data, Pension Credit delivery and a government apology",
        "Request several publications, an implementation plan and an apology.",
        "the motion combines several distinct requested actions rather than one policy object",
        "0",
    ),
    "pw-2025-12-02-374-commons": include(
        "Limits and related changes to agricultural and business property inheritance-tax relief",
        "Authorise the stated inheritance-tax relief package.",
        1,
        "1",
    ),
    "pw-2025-09-03-275-commons": exclude(
        "Multiple hospitality-sector tax, employment and industrial policies",
        "Criticise several policies and request a sector strategy.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2025-12-10-391-commons": exclude(
        "Conduct of the Chancellor of the Exchequer",
        "Call for apologies concerning several statements and disclosures.",
        "this is a conduct and accountability motion, not a stable policy object",
        "0",
    ),
    "pw-2025-02-26-108-commons": exclude(
        "The UK-Mauritius deal plus multiple disclosure and confirmation requests",
        "Criticise the deal and request several separate disclosures.",
        "the motion combines opposition to a deal with multiple information requests",
        "0",
    ),
    "pw-2025-09-03-274-commons": exclude(
        "Multiple possible property-tax increases",
        "Oppose an annual levy, council-tax increases, land-value tax and inheritance-tax changes.",
        "the motion combines several distinct tax policy objects",
        "0",
    ),
    "pw-2025-11-12-345-commons": include(
        "Further increases in National Insurance, income-tax rates or VAT",
        "Call on the Government not to introduce the specified tax increases.",
        -1,
        "0",
    ),
    "pw-2025-01-24-91-commons": exclude(
        "Adjournment of debate on the Climate and Nature Bill",
        "Adjourn the debate.",
        "this is a procedural adjournment vote rather than a vote on the Bill's policy substance",
        "unknown",
    ),
    "pw-2025-07-15-268-commons": include(
        "Retention of the two-child benefit cap",
        "Keep the cap in place and deny additional payments for later children.",
        1,
        "0",
    ),
    "pw-2025-05-21-204-commons": exclude(
        "Multiple business tax, employment-law and business-rate policies",
        "Criticise several policies and request a general change of course.",
        "the motion combines several distinct policy objects",
        "0",
    ),
    "pw-2026-01-21-417-commons": exclude(
        "Unspecified Northern Ireland Troubles motion",
        "The extract only records that the Question was put.",
        "the substantive Question is missing from the source extract",
        "unknown",
    ),
    "pw-2025-12-02-372-commons": include(
        "Freezing the income-tax basic-rate limit and personal allowance for 2028-29 to 2030-31",
        "Authorise provisions keeping both amounts at current levels.",
        1,
        "1",
    ),

    # P1：验证集。这里先形成并冻结判断标准。
    "pw-2024-11-06-30-commons": include(
        "The proposed investors' relief rate and lifetime-limit changes",
        "Adopt the stated capital-gains-tax relief package.",
        1,
        "1",
    ),
    "pw-2024-04-22-133-commons": exclude(
        "Extension of the Bill's carry-over period",
        "Extend the period before proceedings lapse.",
        "this is a procedural carry-over extension rather than a vote on the Bill's policy substance",
        "1",
    ),
    "pw-2024-03-14-97-commons": include(
        "Supplementary Estimates 2023-24 funding package",
        "Authorise the stated current resources, capital adjustment and Consolidated Fund grant.",
        1,
        "1",
    ),
    "pw-2024-12-17-70-commons": exclude(
        "Unspecified amendment to the employer secondary Class 1 contribution rate",
        "Insert only an introductory cross-reference to amendments that are absent from the extract.",
        "the extract does not contain the substantive rate or rule change",
        "unknown",
    ),
    "pw-2024-04-29-140-commons": include(
        "Extension of the Post Office (Horizon System) Offences Bill to Scotland",
        "Allow the Committee to add provisions relating to Scotland.",
        1,
        "unknown",
    ),
    "pw-2024-03-01-86-commons": exclude(
        "Closure of debate on the Conversion Practices (Prohibition) Bill",
        "Put the pending Question immediately.",
        "this is a procedural closure vote rather than a vote on the Bill's policy substance",
        "unknown",
    ),
    "pw-2024-10-08-16-commons": include(
        "VAT on independent-school fees",
        "Oppose the charge and call for exemptions and postponement.",
        -1,
        "0",
    ),
    "pw-2024-03-14-96-commons": include(
        "Home Office funding for the year ending 31 March 2024",
        "Authorise the stated resources and Consolidated Fund grant.",
        1,
        "1",
    ),
    "pw-2024-11-06-33-commons": include(
        "Extension of the energy profits levy to 31 March 2030",
        "Authorise provisions extending the levy period.",
        1,
        "1",
    ),
    "pw-2024-01-23-64-commons": include(
        "The Government's steel decarbonisation plan that risks ending UK primary-steel capacity",
        "Oppose the current plan and call for a job-preserving transition.",
        -1,
        "0",
    ),
    "pw-2024-11-06-29-commons": include(
        "Increase in the business asset disposal relief capital-gains-tax rate",
        "Authorise provisions increasing the rate.",
        1,
        "1",
    ),
    "pw-2024-09-10-15-commons": exclude(
        "Unspecified Winter Fuel Payment main Question",
        "The extract records only the division result.",
        "the substantive Question is missing from the source extract",
        "unknown",
    ),
    "pw-2024-12-04-56-commons": include(
        "Increase in employer National Insurance contributions and reduction of the liability threshold",
        "Oppose the rate increase and threshold reduction.",
        -1,
        "0",
    ),
    "pw-2024-11-06-36-commons": include(
        "Increase in stamp duty land tax rates for additional dwellings from 1 April 2025",
        "Authorise provisions increasing the rates.",
        1,
        "1",
    ),
    "pw-2024-01-09-35-commons": include(
        "Disclosure of the costs and documents underlying the Rwanda asylum plan",
        "Require the listed records to be laid before the House.",
        1,
        "0",
    ),
    "pw-2024-12-17-71-commons": exclude(
        "Unspecified amendment to the employer secondary Class 1 contribution rate",
        "Insert only an introductory cross-reference to amendments that are absent from the extract.",
        "the extract does not contain the substantive rate or rule change",
        "unknown",
    ),
    "pw-2024-03-12-88-commons": include(
        "Charging income tax for the 2024-25 tax year",
        "Approve the annual income-tax charge.",
        1,
        "1",
    ),
    "pw-2024-10-08-17-commons": include(
        "Continuation and full use of the stated farming support funds and farming budget",
        "Call for the funding commitments to be honoured without cuts or delay.",
        1,
        "0",
    ),
    "pw-2024-03-06-87-commons": exclude(
        "Four separate tax motions granted provisional statutory effect",
        "Give immediate provisional effect to four distinct tax measures.",
        "the motion packages four distinct policy objects",
        "1",
    ),
}


# 政府背书只在提案人或正式财政/政府动议提供直接证据时填写。
# Bill 本身由政府提出，不足以证明反对党 amendment 也由政府背书。
OPPOSITION_AMENDMENT_KEYS = {
    "pw-2025-11-17-357-commons",
    "pw-2025-06-18-237-commons",
    "pw-2025-09-10-290-commons",
    "pw-2025-01-21-88-commons",
    "pw-2025-04-24-175-commons",
    "pw-2025-01-14-78-commons",
    "pw-2025-04-30-184-commons",
    "pw-2025-03-17-126-commons",
    "pw-2025-02-24-103-commons",
    "pw-2025-05-07-189-commons",
    "pw-2025-10-15-313-commons",
    "pw-2025-06-09-217-commons",
    "pw-2025-06-17-228-commons",
    "pw-2025-11-24-365-commons",
    "pw-2025-10-14-309-commons",
    "pw-2025-10-15-312-commons",
    "pw-2025-04-30-185-commons",
    "pw-2025-06-09-218-commons",
    "pw-2025-01-21-87-commons",
    "pw-2025-10-21-321-commons",
    "pw-2025-06-17-231-commons",
    "pw-2025-07-09-259-commons",
    "pw-2025-07-08-256-commons",
    "pw-2025-02-24-104-commons",
    "pw-2025-11-17-356-commons",
    "pw-2025-10-14-310-commons",
    "pw-2025-05-16-203-commons",
    "pw-2025-07-09-262-commons",
    "pw-2024-01-30-67-commons",
    "pw-2024-09-03-7-commons",
    "pw-2024-03-25-114-commons",
    "pw-2024-02-20-75-commons",
    "pw-2024-09-03-8-commons",
    "pw-2024-10-29-24-commons",
    "pw-2024-02-27-82-commons",
    "pw-2024-02-27-83-commons",
    "pw-2024-02-20-76-commons",
    "pw-2024-10-29-25-commons",
    "pw-2024-09-03-9-commons",
}

GOVERNMENT_FINANCE_AMENDMENT_KEYS = {
    "pw-2025-12-02-370-commons",
    "pw-2025-12-02-373-commons",
    "pw-2024-11-06-31-commons",
    "pw-2024-11-06-37-commons",
    "pw-2024-11-06-35-commons",
    "pw-2024-11-06-28-commons",
    "pw-2024-11-06-32-commons",
}


def annotate_row(row: dict[str, str]) -> dict[str, str]:
    key = row["division_key"]
    rule = row["polarity_rule"]

    if rule == "multi_object_manual_review":
        result = exclude(
            "Multiple policy objects or amendment actions in one motion",
            "The motion both removes/rejects one object and inserts/adopts another.",
            "the source motion must be split before it can receive one policy polarity",
            "unknown",
        )
    elif key in SPECIAL:
        result = SPECIAL[key]
    elif rule == "manual_review" and row["motion_type"] == "amendment":
        gov = "unknown"
        if key in OPPOSITION_AMENDMENT_KEYS:
            gov = "0"
        elif key in GOVERNMENT_FINANCE_AMENDMENT_KEYS:
            gov = "1"
        result = include(
            amendment_object(row),
            "Adopt or insert the proposed amendment.",
            1,
            gov,
        )
    else:
        raise ValueError(
            f"Unreviewed P0/P1 pattern: {key} | {rule} | {row['motion_type']}"
        )

    result["evidence_span"] = evidence(row["motion_text_clean"])
    return result


def write_csv(path: Path, rows: list[dict[str, str]], fieldnames: list[str]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    with INPUT_PATH.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        source_rows = list(reader)
        source_fields = list(reader.fieldnames or [])

    selected = [row for row in source_rows if row["review_priority"] in {"P0", "P1"}]
    if len(selected) != 169:
        raise ValueError(f"Expected 169 P0/P1 rows, found {len(selected)}")
    if len({row["division_key"] for row in selected}) != 169:
        raise ValueError("P0/P1 division_key values are not unique")

    # 先审 P1 并冻结规则，再把同一规则应用到 P0。
    reviewed_by_key: dict[str, dict[str, str]] = {}
    for priority in ("P1", "P0"):
        for row in selected:
            if row["review_priority"] != priority:
                continue
            reviewed = dict(row)
            reviewed.update(annotate_row(row))
            reviewed_by_key[row["division_key"]] = reviewed

    reviewed_rows = [reviewed_by_key[row["division_key"]] for row in selected]
    p1_rows = [row for row in reviewed_rows if row["review_priority"] == "P1"]
    p0_rows = [row for row in reviewed_rows if row["review_priority"] == "P0"]

    if len(p1_rows) != 54 or len(p0_rows) != 115:
        raise ValueError(f"Unexpected split sizes: P1={len(p1_rows)}, P0={len(p0_rows)}")
    if any(row["review_status"] != "approved" for row in reviewed_rows):
        raise ValueError("Every reviewed row must have review_status=approved")
    if any(row["human_motion_polarity"] not in {"-1", "1", "exclude"} for row in reviewed_rows):
        raise ValueError("Unexpected human_motion_polarity value")
    if any(row["human_object_government_backed"] not in {"0", "1", "unknown"} for row in reviewed_rows):
        raise ValueError("Unexpected government-backing value")
    required_text = ["policy_object", "aye_action", "evidence_span", "reviewer", "review_notes"]
    if any(not row[field].strip() for row in reviewed_rows for field in required_text):
        raise ValueError("A required human-review text field is blank")

    output_fields = source_fields
    write_csv(OUTPUT_DIR / "p1_validation_reviewed.csv", p1_rows, output_fields)
    write_csv(OUTPUT_DIR / "p0_test_reviewed.csv", p0_rows, output_fields)
    write_csv(OUTPUT_DIR / "p0_p1_reviewed_combined.csv", reviewed_rows, output_fields)

    summary_rows: list[dict[str, str]] = []
    for priority, rows in (("P1", p1_rows), ("P0", p0_rows), ("P0+P1", reviewed_rows)):
        counts = Counter(row["human_motion_polarity"] for row in rows)
        summary_rows.extend(
            [
                {"scope": priority, "metric": "reviewed_rows", "value": str(len(rows))},
                {"scope": priority, "metric": "included_positive_polarity", "value": str(counts["1"])},
                {"scope": priority, "metric": "included_negative_polarity", "value": str(counts["-1"])},
                {"scope": priority, "metric": "excluded", "value": str(counts["exclude"])},
                {
                    "scope": priority,
                    "metric": "government_backing_unknown",
                    "value": str(sum(row["human_object_government_backed"] == "unknown" for row in rows)),
                },
            ]
        )
    write_csv(
        OUTPUT_DIR / "p0_p1_review_summary.csv",
        summary_rows,
        ["scope", "metric", "value"],
    )

    print(f"P1 rows: {len(p1_rows)}")
    print(f"P0 rows: {len(p0_rows)}")
    print("Polarity counts:", dict(Counter(row["human_motion_polarity"] for row in reviewed_rows)))
    print("All 169 rows have complete review fields and review_status=approved.")


if __name__ == "__main__":
    main()
