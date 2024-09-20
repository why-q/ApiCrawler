import json


def merge(yunxiu_json_path, xiuxiu_json_path, output_json_path):
    yunxiu_data = json.load(open(yunxiu_json_path, "r", encoding="utf-8"))
    xiuxiu_data = json.load(open(xiuxiu_json_path, "r", encoding="utf-8"))

    merged_data = []
    current_id = 1

    for item in yunxiu_data:
        new_item = item.copy()
        new_item["id"] = str(current_id).zfill(4)
        new_item["platform"] = "yunxiu"
        merged_data.append(new_item)
        current_id += 1

    for item in xiuxiu_data:
        new_item = item.copy()
        new_item["id"] = str(current_id).zfill(4)
        new_item["platform"] = "xiuxiu"
        merged_data.append(new_item)
        current_id += 1

    merged_data.sort(key=lambda x: x["id"])

    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=4)


yunxiu_json_path = "combine/meituyunxiu_result.json"
xiuxiu_json_path = "combine/meituxiuxiu_result.json"
output_json_path = "combine/妆容方案.json"
merge(yunxiu_json_path, xiuxiu_json_path, output_json_path)

yunxiu_json_path = "combine/meituyunxiu_result_num.json"
xiuxiu_json_path = "combine/meituxiuxiu_result_num.json"
output_json_path = "combine/妆容方案(编号).json"
merge(yunxiu_json_path, xiuxiu_json_path, output_json_path)
