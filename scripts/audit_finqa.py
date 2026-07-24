import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PAGE_PATTERN = re.compile(r"page_(\d+)\.pdf$", re.IGNORECASE)
FILES = {
    "train": "train.json",
    "validation": "dev.json",
    "test": "test.json",
}


def is_blank(value: Any) -> bool:
    return value is None or (isinstance(value, str) and not value.strip())


def main() -> None:
    dataset_directory = Path(sys.argv[1])

    anomaly_counts: Counter[str] = Counter()
    examples: dict[str, list[str]] = defaultdict(list)
    example_locations: dict[str, list[str]] = defaultdict(list)
    document_signatures: dict[tuple[str, str], set[str]] = defaultdict(set)

    def report(name: str, split: str, example_id: str, detail: str = "") -> None:
        anomaly_counts[name] += 1

        if len(examples[name]) < 5:
            message = f"{split}: {example_id}"
            if detail:
                message += f" — {detail}"
            examples[name].append(message)

    for split, filename in FILES.items():
        records = json.loads((dataset_directory / filename).read_text(encoding="utf-8"))

        print(f"{split}: {len(records)} records")

        for record in records:
            example_id = str(record.get("id", ""))
            filename_value = str(record.get("filename", ""))

            example_locations[example_id].append(split)

            if is_blank(example_id):
                report("record.id.blank", split, "<missing>")

            if is_blank(filename_value):
                report("record.filename.blank", split, example_id)
            elif PAGE_PATTERN.search(filename_value) is None:
                report(
                    "record.filename.invalid_page",
                    split,
                    example_id,
                    filename_value,
                )

            pre_text = record.get("pre_text", [])
            post_text = record.get("post_text", [])
            table = record.get("table", [])
            qa = record.get("qa", {})

            signature_payload = {
                "pre_text": pre_text,
                "post_text": post_text,
                "table": table,
            }
            signature = hashlib.sha256(
                json.dumps(
                    signature_payload,
                    sort_keys=True,
                    ensure_ascii=False,
                ).encode("utf-8")
            ).hexdigest()
            document_signatures[(split, filename_value)].add(signature)

            if not table:
                report("table.empty", split, example_id)

            for row_index, row in enumerate(table):
                if not row or all(is_blank(cell) for cell in row):
                    report(
                        "table.row.blank",
                        split,
                        example_id,
                        f"row={row_index}",
                    )

            if is_blank(qa.get("question")):
                report("qa.question.blank", split, example_id)

            if is_blank(qa.get("answer")):
                report("qa.answer.blank", split, example_id)

            if is_blank(qa.get("exe_ans")):
                report("qa.exe_ans.blank", split, example_id)

            if is_blank(qa.get("answer")) and is_blank(qa.get("exe_ans")):
                report("qa.answer_and_exe_ans.blank", split, example_id)

            if is_blank(qa.get("program")):
                report("qa.program.blank", split, example_id)

            if is_blank(qa.get("program_re")):
                report("qa.program_re.blank", split, example_id)

            gold_inds = qa.get("gold_inds", {})

            if not gold_inds:
                report("qa.gold_inds.empty", split, example_id)

            text_count = len(pre_text) + len(post_text)

            available_source_keys = {f"text_{index}" for index in range(text_count)}

            available_source_keys.update(
                f"text_{index - text_count}" for index in range(text_count)
            )

            for source_key in gold_inds:
                if source_key not in available_source_keys:
                    report(
                        "qa.gold_inds.unresolved",
                        split,
                        example_id,
                        source_key,
                    )

            steps = qa.get("steps", [])

            if not steps:
                report("qa.steps.empty", split, example_id)

            for step_index, step in enumerate(steps):
                operation = str(step.get("op", ""))

                if is_blank(operation):
                    report(
                        "step.op.blank",
                        split,
                        example_id,
                        f"step={step_index}",
                    )

                if is_blank(step.get("arg1")):
                    report(
                        "step.arg1.blank",
                        split,
                        example_id,
                        f"step={step_index}, op={operation}",
                    )

                if is_blank(step.get("arg2")):
                    report(
                        f"step.arg2.blank:{operation}",
                        split,
                        example_id,
                        f"step={step_index}",
                    )

                if is_blank(step.get("res")):
                    report(
                        f"step.res.blank:{operation}",
                        split,
                        example_id,
                        f"step={step_index}",
                    )

    for example_id, locations in example_locations.items():
        if not example_id:
            continue

        if len(locations) > 1:
            report(
                "example_id.duplicate_or_cross_split",
                ",".join(locations),
                example_id,
            )

    for (split, filename), signatures in document_signatures.items():
        if len(signatures) > 1:
            report(
                "document.conflicting_context",
                split,
                filename,
                f"{len(signatures)} variants",
            )

    print("\nAnomalies:")

    if not anomaly_counts:
        print("  none")
        return

    for name, count in sorted(anomaly_counts.items()):
        print(f"\n{name}: {count}")
        for example in examples[name]:
            print(f"  {example}")


if __name__ == "__main__":
    main()
