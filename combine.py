import json
import math
import random
from itertools import product
from pathlib import Path
from typing import Callable, Dict, List, Set, Tuple

import click
import yaml
from loguru import logger

MAIN_KEYS = ["眼影", "腮红", "口红"]
MAX_REPEAT = 1


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

                dict[key] = sorted(v_group)
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
        sum(a == b for a, b in zip(combination, used_combo)) <= MAX_REPEAT
        for used_combo in used_combinations
    ):
        used_combinations.add(combination)
        return True

    return False


def select_main_combinations(
    combinations: List[Tuple[str, str, str]],
) -> List[Tuple[str, str, str]]:
    used_combinations: Set[Tuple[str, str, str]] = set()
    selected = []
    random.shuffle(combinations)

    for combo in combinations:
        if filter(combo, used_combinations):
            selected.append(combo)

    logger.info(f"Selected: {len(selected)}")

    return selected


def get_other_combinations(config: Dict[str, List[str]]) -> List[Dict[str, str]]:
    logger.info("Get other combinations...")
    other_keys = [key for key in config.keys() if key not in MAIN_KEYS]

    if other_keys == []:
        return []

    total_combinations = 1
    for key in other_keys:
        total_combinations *= len(config[key])
    logger.info(f"Len: {total_combinations}")

    other_combinations = [
        dict(zip(other_keys, combo))
        for combo in product(*(config[key] for key in other_keys))
    ]
    return other_combinations


def create_flat_index_mapping(json_file_path):
    with open(json_file_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    flat_mapping = {}
    for _, items in data.items():
        flat_mapping.update({item: index + 1 for index, item in enumerate(items)})

    return flat_mapping


def combine_selections(
    main_combinations: List[Tuple[str, str, str]],
    other_combinations: List[Dict[str, str]],
    k: int = 1000,
) -> List[Dict[str, str]]:
    result = []
    main_combo_len = len(main_combinations)
    repeat_times = math.ceil(k / main_combo_len)

    for main_combo in main_combinations:
        main_combo_dict = dict(zip(MAIN_KEYS, main_combo))
        for _ in range(repeat_times):
            other_combo = (
                random.choice(other_combinations) if len(other_combinations) > 0 else {}
            )
            result.append({**main_combo_dict, **other_combo})

    return random.sample(result, k)


def generate_combinations(
    file_path: str,
    limit: int = 100,
) -> List[Dict[str, str]]:
    config = load_config(file_path, limit)

    all_main_combinations = get_main_combinations(config)
    selected_main_combinations = select_main_combinations(all_main_combinations)

    other_combinations = get_other_combinations(config)
    final_combinations = combine_selections(
        selected_main_combinations, other_combinations, k=limit
    )

    return final_combinations


def reprocess(combo: Dict[str, str]) -> Dict[str, str]:
    if combo.get("装饰", None) is not None:
        if random.random() < 0.99:
            combo["装饰"] = "无"

    return combo


def save(
    combinations: List[Dict[str, str]],
    output_json: str,
    flat_mapping: Dict[str, int],
    reprocess: Callable[[Dict[str, str]], Dict[str, str]] = reprocess,
):
    res, res_num = [], []
    for i, combo in enumerate(combinations):
        res.append({"id": str(i), "makeup": combo})

        if reprocess:
            combo = reprocess(combo)

        combo_num = {}
        for key, value in combo.items():
            combo_num[key] = flat_mapping.get(value, 0)

        res_num.append({"id": str(i), "makeup": combo_num})

    with open(output_json, "w", encoding="utf-8") as file:
        json.dump(res, file, ensure_ascii=False, indent=4)

    output_num_json = str(
        Path(output_json).parent / f"{Path(output_json).stem}_num.json"
    )
    with open(output_num_json, "w", encoding="utf-8") as file:
        json.dump(res_num, file, ensure_ascii=False, indent=4)

    logger.info(f"Saved to {output_json} and {output_num_json}")


@click.command()
@click.option("--yaml-file", "-y", help="Path to the configuration file", required=True)
@click.option("--limit", "-l", default=100, help="Limit for combinations")
@click.option("--seed", "-s", default=9999, help="Random seed for reproducibility")
def main(yaml_file, limit, seed):
    """
    Main function to generate and process combinations from a YAML configuration file.

    This function performs the following steps:
    1. Generates combinations based on the input YAML file
    2. Prints the number of combinations and some examples
    3. Saves the combinations to a JSON file
    4. Runs a test to verify the uniqueness of the combinations

    Args:
        yaml_file (str): Path to the YAML configuration file
        limit (int): Limit for the number of combinations to generate
        seed (int): Random seed for reproducibility

    Returns:
        None
    """
    random.seed(seed)

    combinations = generate_combinations(yaml_file, limit=limit)
    logger.info(f"Num of combinations: {len(combinations)}")
    logger.info("Example combinations: {}", combinations[:5])

    # Save to JSON
    json_cfg_path = Path(yaml_file).parent / f"{Path(yaml_file).stem}.json"
    flat_mapping = create_flat_index_mapping(json_cfg_path)
    output_json = Path(yaml_file).parent / f"{Path(yaml_file).stem}_result.json"
    save(combinations, output_json, flat_mapping)

    # Test uniqueness of combinations
    logger.info("🚀 Testing...")
    combos = []
    with open(output_json, "r", encoding="utf-8") as file:
        combos_list = json.load(file)
        for combo in combos_list:
            combos.append(combo["makeup"])

    combo_set = set()
    for combo in combos:
        main_combo = tuple(combo[key] for key in MAIN_KEYS)
        if len(combo_set) == 0:
            combo_set.add(main_combo)
            continue
        elif main_combo in combo_set:
            continue
        elif all(
            sum(a == b for a, b in zip(main_combo, combo)) > MAX_REPEAT
            for combo in combo_set
        ):
            logger.error(
                f"❌ Test failed. Duplicate combination found: ({', '.join(main_combo)})"
            )
            break
        else:
            combo_set.add(main_combo)

    logger.success("✅ Test passed.")


if __name__ == "__main__":
    main()
