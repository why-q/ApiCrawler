import re


def custom_filter(id1: str, id2: str) -> bool:
    """
    freepik's id in vector databse is like: 57158677.png

    Use re to get the part: 57158677
    """
    pattern = r"(\d+)\."
    match1 = re.search(pattern, id1)
    match2 = re.search(pattern, id2)

    if match1 and match2:
        id1 = int(match1.group(1))
        id2 = int(match2.group(1))

        return abs(id1 - id2) <= 1000

    return False
