import json
import random
from itertools import product
from pathlib import Path
from typing import Dict, List, Set, Tuple

import click
import yaml
from loguru import logger

MAIN_KEYS = ["眼影", "腮红", "口红"]


def load_config(file_path: str, limit: int = 500) -> Dict[str, List[str]]:
    with open(file_path, "r", encoding="utf-8") as file:
        dict = yaml.safe_load(file)

        for key, value in dict.items():
            if isinstance(value, list):
                pass
            elif isinstance(value, Dict):
                to_group = []
                for _, v in value.items():
                    to_group.append(v)

                group = list(product(*to_group))
                v_group = ["-".join(v) for v in group]

                if len(v_group) > limit:
                    v_group = random.sample(v_group, limit)

                dict[key] = v_group
            else:
                raise TypeError

    # save to json
    file_path = Path(file_path)
    with open(
        file_path.parent / f"{file_path.stem}.json", "w", encoding="utf-8"
    ) as file:
        json.dump(dict, file, ensure_ascii=False, indent=4)

    return dict


def get_main_combinations(config: Dict[str, List[str]]) -> List[Tuple[str, str, str]]:
    logger.info("Get main combinations...")
    total_combinations = 1
    for key in MAIN_KEYS:
        total_combinations *= len(config[key])
    logger.info(f"Len: {total_combinations}")

    return list(product(*(config[key] for key in MAIN_KEYS)))


def filter(
    combination: Tuple[str, str, str], used_combinations: Set[Tuple[str, str, str]]
) -> bool:
    if all(
        sum(a != b for a, b in zip(combination, used_combo)) >= 2
        for used_combo in used_combinations
    ):
        used_combinations.add(combination)
        return True

    return False


def select_main_combinations(
    combinations: List[Tuple[str, str, str]], num_select: int
) -> List[Tuple[str, str, str]]:
    used_combinations: Set[Tuple[str, str, str]] = set()
    selected = []
    random.shuffle(combinations)

    for combo in combinations:
        if filter(combo, used_combinations):
            selected.append(combo)
            if len(selected) >= num_select:
                break

    return selected


def get_other_combinations(config: Dict[str, List[str]]) -> List[Dict[str, str]]:
    logger.info("Get other combinations...")
    other_keys = [key for key in config.keys() if key not in MAIN_KEYS]
    total_combinations = 1
    for key in other_keys:
        total_combinations *= len(config[key])
    logger.info(f"Len: {total_combinations}")

    other_combinations = [
        dict(zip(other_keys, combo))
        for combo in product(*(config[key] for key in other_keys))
    ]
    return other_combinations


def combine_selections(
    main_combinations: List[Tuple[str, str, str]],
    other_combinations: List[Dict[str, str]],
    k: int = 1000,
) -> List[Dict[str, str]]:
    result = []
    for _ in range(k):
        main_combo = random.choice(main_combinations)
        main_combo_dict = dict(zip(MAIN_KEYS, main_combo))
        other_combo = random.choice(other_combinations)
        result.append({**main_combo_dict, **other_combo})
    return result


def generate_combinations(
    file_path: str, limit: int = 100, num_select: int = 100
) -> List[Dict[str, str]]:
    config = load_config(file_path, limit)

    all_main_combinations = get_main_combinations(config)
    selected_main_combinations = select_main_combinations(
        all_main_combinations, num_select
    )

    other_combinations = get_other_combinations(config)
    final_combinations = combine_selections(
        selected_main_combinations, other_combinations
    )

    return final_combinations


@click.command()
@click.option("--yaml-file", "-y", help="Path to the configuration file", required=True)
@click.option("--limit", "-l", default=100, help="Limit for combinations")
@click.option(
    "--num-select", "-n", default=1000, help="Number of combinations to select"
)
@click.option("--seed", "-s", default=9999, help="Random seed for reproducibility")
def main(yaml_file, limit, num_select, seed):
    """
    Main function to generate and process combinations from a YAML configuration file.

    This function performs the following steps:
    1. Generates combinations based on the input YAML file
    2. Prints the number of combinations and some examples
    3. Saves the combinations to a JSONL file
    4. Runs a test to verify the uniqueness of the combinations

    Args:
        yaml_file (str): Path to the YAML configuration file
        limit (int): Limit for the number of combinations to generate
        num_select (int): Number of combinations to select
        seed (int): Random seed for reproducibility

    Returns:
        None
    """
    random.seed(seed)

    combinations = generate_combinations(yaml_file, limit=limit, num_select=num_select)
    logger.info(f"Num of combinations: {len(combinations)}")
    logger.info("Example combinations: {}", combinations[:5])

    # Save to JSONL
    yaml_file = Path(yaml_file)
    output_file = yaml_file.parent / f"{yaml_file.stem}.jsonl"
    with open(output_file, "w", encoding="utf-8") as file:
        for combo in combinations:
            json.dump(combo, file, ensure_ascii=False)
            file.write("\n")

    logger.info(f"The combinations have been saved to {output_file}")

    # Test uniqueness of combinations
    logger.info("🚀 Testing...")
    combos = []
    with open(output_file, "r", encoding="utf-8") as file:
        for line in file:
            combo = json.loads(line)
            combos.append(combo)

    combo_set = set()
    for combo in combos:
        main = tuple(combo[key] for key in MAIN_KEYS)
        if all(sum(a != b for a, b in zip(main, combo)) <= 1 for combo in combos):
            logger.error("❌ Test failed.")
            break
        else:
            combo_set.add(main)
    else:
        logger.success("✅ Test passed.")


if __name__ == "__main__":
    main()
