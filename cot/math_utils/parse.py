import re


def _fix_fracs(string: str) -> str:
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            if len(substr) == 0:
                new_str += "\\frac"
                continue

            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}{" + b + "}" + post_substr
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        post_substr = substr[2:]
                        new_str += "{" + a + "}" + b + post_substr
                    else:
                        new_str += "{" + a + "}" + b
    string = new_str
    return string


def _fix_a_slash_b(string: str) -> str:
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == "{}/{}".format(a, b)
        new_string = "\\frac{" + str(a) + "}{" + str(b) + "}"
        return new_string
    except Exception:
        return string


def _remove_right_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _remove_ext_units(string: str) -> str:
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        return splits[0]
    return string


def _fix_sqrt(string: str) -> str:
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if len(split) == 0:
            new_string += "\\sqrt"
            continue
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = _remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    string = _fix_sqrt(string)
    string = string.replace(" ", "")
    string = _fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = _fix_a_slash_b(string)

    return string


def remove_boxed(string: str):
    left = "\\boxed{"
    try:
        assert string[: len(left)] == left
        assert string[-1] == "}"
        return string[len(left) : -1]
    except AssertionError:
        return None


def last_boxed_only_string(string: str):
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None

    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1

    if right_brace_idx is None:
        retval = None
    else:
        retval = string[idx : right_brace_idx + 1]

    return retval


def extract_ans(string: str):
    if '$' in string:
        matches = re.findall(r'\$(.*?)\$', string, re.DOTALL)
        string = matches[0] if matches else None

    index = string.find('{')
    result = string[index + 1:]

    index = result.rfind('}')
    result = result[: index]

    return result


def get_answer(str1: str):
    boxed_answer = last_boxed_only_string(str1)
    if boxed_answer:
        return remove_boxed(boxed_answer)

    dollar_patterns = [
        r'(?:is|=|equals?)\s*\$([^$]+)\$',
        r'answer is\s*\$([^$]+)\$',
        r'final answer is\s*\$([^$]+)\$',
        r'result is\s*\$([^$]+)\$',
    ]

    for pattern in dollar_patterns:
        matches = re.findall(pattern, str1, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    specific_patterns = [
        r'(?:the\s+)?(?:final\s+)?answer\s+is\s+([+-]?\d+(?:\.\d+)?)',
        r'(?:the\s+)?(?:final\s+)?result\s+is\s+([+-]?\d+(?:\.\d+)?)',
        r'(?:the\s+)?(?:total|sum)\s+(?:is|equals?)\s+([+-]?\d+(?:\.\d+)?)',
    ]

    for pattern in specific_patterns:
        matches = re.findall(pattern, str1, re.IGNORECASE)
        if matches:
            return matches[-1].strip()

    general_patterns = [
        r'(?:is|=|equals?)\s+([+-]?\d+(?:\.\d+)?)\s*[.\n]*$',
        r'(?:is|=|equals?)\s+\$?([+-]?\d+(?:\.\d+)?)\$?\s*[.\n]*$',
    ]

    for pattern in general_patterns:
        match = re.search(pattern, str1, re.IGNORECASE | re.MULTILINE)
        if match:
            return match.group(1).strip()

    final_eq_match = re.search(r'=\s*([^=\n]+?)[\s.]*$', str1)
    if final_eq_match:
        candidate = final_eq_match.group(1).strip()
        if re.match(r'^[0-9+\-*/^().\s\\a-zA-Z{}]+$', candidate) and len(candidate) < 50:
            return candidate

    return None

