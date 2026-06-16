import datetime
import json
import re

from .base_node import UtilityNode

BOOLEAN_CHOICES = ["true", "false"]
DATE_TOKEN_RE = re.compile(r"%date:(.*?)%")
PATTERN_RE = re.compile(r"%([^%]+)%")
INVALID_FILENAME_CHARS_RE = re.compile(r'[<>:"/\\|?*]')


class FileNamePrefixDateDirFirst(UtilityNode):
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "date": (BOOLEAN_CHOICES, {"default": "true"}),
                "date_directory": (BOOLEAN_CHOICES, {"default": "true"}),
                "custom_directory": ("STRING", {"default": ""}),
                "custom_text": ("STRING", {"default": ""}),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename_prefix",)
    FUNCTION = "get_filename_prefix"

    def get_filename_prefix(
        self,
        date,
        date_directory,
        custom_directory,
        custom_text,
        prompt=None,
        extra_pnginfo=None,
    ):
        now = datetime.datetime.now()
        prefix = ""

        if date_directory == "true":
            prefix += now.strftime("%y%m%d") + "/"

        if custom_directory:
            directory = _replace_placeholders(custom_directory, extra_pnginfo, prompt, now)
            directory = _normalize_directory_component(directory)
            if directory:
                prefix += directory + "/"

        if date == "true":
            prefix += now.strftime("%y%m%d%H%M%S")

        if custom_text:
            text = _replace_placeholders(custom_text, extra_pnginfo, prompt, now)
            text = INVALID_FILENAME_CHARS_RE.sub("", text)
            if text:
                prefix += "_" + text

        return (prefix,)


def _replace_placeholders(text, extra_pnginfo, prompt, now):
    if not isinstance(text, str):
        return str(text)

    text = DATE_TOKEN_RE.sub(lambda match: _format_date_token(match.group(1), now), text)
    if extra_pnginfo is None or prompt is None:
        return text

    extra_pnginfo = _parse_json_if_needed(extra_pnginfo)
    prompt = _parse_json_if_needed(prompt)
    if not isinstance(extra_pnginfo, dict) or not isinstance(prompt, dict):
        return text

    workflow_nodes = extra_pnginfo.get("workflow", {}).get("nodes", [])
    if not isinstance(workflow_nodes, list):
        return text

    node_name_to_id = {}
    for node in workflow_nodes:
        if not isinstance(node, dict):
            continue
        properties = node.get("properties", {})
        if not isinstance(properties, dict):
            continue
        node_name = properties.get("Node name for S&R")
        if node_name:
            node_name_to_id[str(node_name)] = str(node.get("id"))

    for pattern in PATTERN_RE.findall(text):
        if "." not in pattern:
            continue
        node_name, widget_name = pattern.split(".", 1)
        node_id = node_name_to_id.get(node_name, node_name)
        prompt_node = prompt.get(str(node_id))
        if not isinstance(prompt_node, dict):
            continue
        inputs = prompt_node.get("inputs", {})
        if not isinstance(inputs, dict) or widget_name not in inputs:
            continue
        text = text.replace(f"%{pattern}%", str(inputs[widget_name]))

    return text


def _format_date_token(pattern, now):
    token_map = {
        "yyyy": now.strftime("%Y"),
        "yy": now.strftime("%y"),
        "MM": now.strftime("%m"),
        "M": now.strftime("%m").lstrip("0"),
        "dd": now.strftime("%d"),
        "d": now.strftime("%d").lstrip("0"),
        "hh": now.strftime("%H"),
        "h": now.strftime("%H").lstrip("0"),
        "mm": now.strftime("%M"),
        "m": now.strftime("%M").lstrip("0"),
        "ss": now.strftime("%S"),
        "s": now.strftime("%S").lstrip("0"),
    }
    tokens = sorted(token_map, key=len, reverse=True)
    output = []
    cursor = 0
    while cursor < len(pattern):
        for token in tokens:
            if pattern.startswith(token, cursor):
                output.append(token_map[token])
                cursor += len(token)
                break
        else:
            output.append(pattern[cursor])
            cursor += 1
    return "".join(output)


def _parse_json_if_needed(value):
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _normalize_directory_component(value):
    value = str(value).strip().replace("\\", "/")
    parts = [part for part in value.split("/") if part not in ("", ".", "..")]
    parts = [INVALID_FILENAME_CHARS_RE.sub("", part) for part in parts]
    parts = [part for part in parts if part]
    return "/".join(parts)


FILENAME_NODE_CLASS_MAPPINGS = {
    "AK_FileNamePrefixDateDirFirst": FileNamePrefixDateDirFirst,
}

FILENAME_NODE_DISPLAY_NAME_MAPPINGS = {
    "AK_FileNamePrefixDateDirFirst": "Akatz File Name Prefix Date Dir First",
}
