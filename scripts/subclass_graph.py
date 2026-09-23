import json

import common


def node_config(class_name: str) -> dict:
    parent = common.class_parent_name(class_name) or class_name
    path = common.class_dir(parent) / "subclass_graph.json"
    if not path.exists():
        return {}
    bare = class_name.split("/", 1)[-1]
    return json.loads(path.read_text()).get("nodes", {}).get(bare, {})
