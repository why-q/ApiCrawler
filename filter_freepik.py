import re


def custom_filter(id1: str, id2: str) -> bool:
    """
    freepik's id in vector databse is like: 23-2147799044.png

    Use re to get the part: 2147799044
    """
    pattern = r"(\d+)-(\d+)\."
    match1 = re.search(pattern, id1)
    match2 = re.search(pattern, id2)

    if match1 and match2:
        id1_preffix = int(match1.group(1))
        id2_preffix = int(match2.group(1))

        id1_suffix = int(match1.group(2))
        id2_suffix = int(match2.group(2))

        return id1_preffix == id2_preffix and abs(id1_suffix - id2_suffix) <= 100

    return False
