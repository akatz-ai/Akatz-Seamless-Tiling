def make_smart_type(value):
    if isinstance(value, str):
        return SmartType(value)
    return value


class SmartType(str):
    def __ne__(self, other):
        if self == "*" or other == "*":
            return False
        self_set = set(self.split(","))
        other_set = set(str(other).split(","))
        return not self_set.issubset(other_set)


def variant_support():
    def decorator(cls):
        if hasattr(cls, "INPUT_TYPES"):
            original_input_types = getattr(cls, "INPUT_TYPES")

            def input_types_with_variants(*args, **kwargs):
                types = original_input_types(*args, **kwargs)
                for category in ("required", "optional"):
                    for key, value in types.get(category, {}).items():
                        if isinstance(value, tuple):
                            types[category][key] = (
                                make_smart_type(value[0]),
                            ) + value[1:]
                return types

            setattr(cls, "INPUT_TYPES", input_types_with_variants)

        if hasattr(cls, "RETURN_TYPES"):
            cls.RETURN_TYPES = tuple(make_smart_type(item) for item in cls.RETURN_TYPES)

        if hasattr(cls, "VALIDATE_INPUTS"):
            raise NotImplementedError("variant_support does not support VALIDATE_INPUTS")

        def validate_inputs(input_types):
            declared_inputs = cls.INPUT_TYPES()

            def validate_one(type_map):
                for key, value in type_map.items():
                    if isinstance(value, SmartType):
                        continue
                    expected_type = None
                    if key in declared_inputs.get("required", {}):
                        expected_type = declared_inputs["required"][key][0]
                    elif key in declared_inputs.get("optional", {}):
                        expected_type = declared_inputs["optional"][key][0]
                    if expected_type is not None and make_smart_type(value) != expected_type:
                        return f"Invalid type of {key}: {value} (expected {expected_type})"
                return True

            if isinstance(input_types, list):
                for type_map in input_types:
                    result = validate_one(type_map)
                    if isinstance(result, str):
                        return result
                return True

            return validate_one(input_types)

        setattr(cls, "VALIDATE_INPUTS", validate_inputs)
        return cls

    return decorator

