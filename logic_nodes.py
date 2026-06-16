from .base_node import LogicNode, NODE_POSTFIX
from .tools import variant_support


class _AnySwitchInputs(dict):
    def __init__(self, count=5):
        super().__init__((f"any_{index:02d}", ("*",)) for index in range(1, count + 1))

    def __contains__(self, key):
        return isinstance(key, str) and key.startswith("any_")

    def __getitem__(self, key):
        if key in self.keys():
            return super().__getitem__(key)
        if key in self:
            return ("*",)
        raise KeyError(key)


def _any_input_sort_key(key):
    suffix = key.rsplit("_", 1)[-1]
    return int(suffix) if suffix.isdigit() else 0


def _is_empty_any_value(value):
    if value is None:
        return True
    if isinstance(value, dict) and "model" in value and "clip" in value:
        return all(item is None for item in value.values())
    return False


@variant_support()
class AnySwitch(LogicNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": _AnySwitchInputs(),
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("*",)
    FUNCTION = "switch"

    def switch(self, **kwargs):
        keys = sorted(
            (key for key in kwargs if key.startswith("any_")),
            key=_any_input_sort_key,
        )
        for key in keys:
            value = kwargs.get(key)
            if not _is_empty_any_value(value):
                return (value,)
        return (None,)


LOGIC_NODE_CLASS_MAPPINGS = {
    f"AnySwitch {NODE_POSTFIX}": AnySwitch,
}

LOGIC_NODE_DISPLAY_NAME_MAPPINGS = {
    f"AnySwitch {NODE_POSTFIX}": f"Any Switch {NODE_POSTFIX}",
}

