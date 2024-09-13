def custom_filter(id1: str, id2: str) -> bool:
    """
    freepik's id in vector databse is like: 57158677.png or 57158677-1.png

    Use re to get the part: 57158677 or 57158677-1
    """
    id1 = id1.split(".")[0]
    id2 = id2.split(".")[0]

    if "-" in id1 and "-" in id2:
        id1_prefix, id1_suffix = id1.split("-")
        id2_prefix, id2_suffix = id2.split("-")

        return (
            id1_prefix == id2_prefix and abs(int(id1_suffix) - int(id2_suffix)) <= 500
        )
    elif "-" not in id1 and "-" not in id2:
        return abs(int(id1) - int(id2)) <= 500
    else:
        return False
