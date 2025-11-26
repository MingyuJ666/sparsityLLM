import os
import json
import argparse
from collections import defaultdict

from datasets import load_dataset

from analyze_mmlu_pro_area_difficulty import (
    create_context_variants,
    format_category_name,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT = os.path.join(BASE_DIR, "mmlu_robust_dataset.json")


def select_areas(area_data):
    """Return every academic area present in the dataset."""
    return sorted(area_data.keys())


def prepare_records(selected_areas, area_data, num_samples_per_area):
    """Build dataset entries with noise variants for each sample."""
    records = []
    global_id = 0

    for area in selected_areas:
        samples = area_data[area]
        if num_samples_per_area is None:
            sample_count = len(samples)
        else:
            sample_count = min(num_samples_per_area, len(samples))
        for idx in range(sample_count):
            sample = samples[idx]
            question = sample["question"]
            options = sample["options"]
            correct_answer = sample["answer"]

            if len(correct_answer) == 1 and correct_answer.upper() in "ABCDEFGHIJ":
                correct_idx = ord(correct_answer.upper()) - ord("A")
            else:
                correct_idx = sample.get("answer_index", 0)

            prompts, variant_details = create_context_variants(
                question=question,
                options=options,
                correct_answer_idx=correct_idx,
                category=sample["category"],
                return_details=True,
            )

            record = {
                "id": global_id,
                "area": area,
                "category": sample["category"],
                "question": question,
                "options": options,
                "answer": correct_answer,
                "answer_index": correct_idx,
                "variants": {
                    noise: {
                        "prompt": prompts[noise],
                        "options": variant_details[noise]["options"],
                        "num_options": variant_details[noise]["num_options"],
                    }
                    for noise in ['normal', 'light_noise', 'heavy_noise']
                },
                "meta": {
                    "source_sample_id": int(sample.get("sample_id", idx)),
                    "subject": sample.get("subject", ""),
                },
            }

            records.append(record)
            global_id += 1

    return records


def main():
    parser = argparse.ArgumentParser(description="Build MMLU robustness JSON dataset.")
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples per academic area (default: all samples)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=DEFAULT_OUTPUT,
        help=f"Output JSON path (default: {DEFAULT_OUTPUT})",
    )
    args = parser.parse_args()

    print("Loading MMLU-Pro test split...")
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    validation_data = ds["test"]
    print(f"Total validation samples: {len(validation_data)}")

    area_data = defaultdict(list)
    for idx, sample in enumerate(validation_data):
        area = format_category_name(sample["category"])
        sample["sample_id"] = idx
        area_data[area].append(sample)

    print("Selecting areas...")
    selected_areas = select_areas(area_data)
    if not selected_areas:
        raise ValueError("No academic areas satisfy the minimum sample requirement.")
    print(f"Selected areas ({len(selected_areas)}): {selected_areas}")

    print("Preparing records with noise variants...")
    records = prepare_records(selected_areas, area_data, args.num_samples)
    print(f"Total records: {len(records)}")

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "num_records": len(records),
                    "num_selected_areas": len(selected_areas),
                    "areas": selected_areas,
                },
                "data": records,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print(f"MMLU robustness dataset written to {args.output}")


if __name__ == "__main__":
    main()

