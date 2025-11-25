import re

def _fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            # Check whether substr is empty
            if len(substr) == 0:
                new_str += "\\frac"
                continue
            
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except:
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

def _fix_a_slash_b(string):
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
    except:
        return string

def _remove_right_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        # if len(splits) < 2:
        #     return string
        # assert len(splits) == 2
        return splits[0]
    else:
        return string

def _remove_ext_units(string):
    # "\\text{ " only ever occurs (at least in the val set) when describing units
    if "\text{ " in string:
        splits = string.split("\text{ ")
        # if len(splits) < 2:
        #     return string
        # assert len(splits) == 2
        return splits[0]
    else:
        return string

def _fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0] 
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string

def _strip_string(string):
    # linebreaks  
    string = string.replace("\n", "")
    #print(string)

    # remove inverse spaces
    string = string.replace("\\!", "")
    #print(string)

    # replace \\ with \
    string = string.replace("\\\\", "\\")
    #print(string)

    # replace tfrac and dfrac with frac
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    #print(string)

    # remove \left and \right
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    #print(string)
    
    # Remove circ (degrees)
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")

    # remove dollar signs
    string = string.replace("\\$", "")
    
    # remove units (on the right)
    string = _remove_right_units(string)

    # remove percentage
    string = string.replace("\\%", "")
    string = string.replace("\%", "")

    # " 0." equivalent to " ." and "{0." equivalent to "{." Alternatively, add "0" if "." is the start of the string
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    # if empty, return empty string
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string

    # to consider: get rid of e.g. "k = " or "q = " at beginning
    if len(string.split("=")) == 2:
        if len(string.split("=")[0]) <= 2:
            string = string.split("=")[1]

    # fix sqrt3 --> sqrt{3}
    string = _fix_sqrt(string)

    # remove spaces
    string = string.replace(" ", "")
    
    # \frac1b or \frac12 --> \frac{1}{b} and \frac{1}{2}, etc. Even works with \frac1{72} (but not \frac{72}1). Also does a/b --> \\frac{a}{b}
    string = _fix_fracs(string)

    # manually change 0.5 --> \frac{1}{2}
    if string == "0.5":
        string = "\\frac{1}{2}"
        
    # NOTE: X/Y changed to \frac{X}{Y} in dataset, but in simple cases fix in case the model output is X/Y
    string = _fix_a_slash_b(string)

    return string

def extract_ans(string):
    if '$' in string:
        matches = re.findall(r'\$(.*?)\$', string, re.DOTALL)
        string = matches[0] if matches else None

    index = string.find('{')
    result = string[index + 1:]

    index = result.rfind('}')
    result = result[: index]

    return result


def remove_boxed(s):
    left = "\\boxed{"
    try:
        assert s[:len(left)] == left
        assert s[-1] == "}"
        return s[len(left):-1]
    except:
        return None

def last_boxed_only_string(string):
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
    
    if right_brace_idx == None:
        retval = None
    else:
        retval = string[idx:right_brace_idx + 1]
    
    return retval

def get_answer(str1):

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


def is_equiv(str1, str2, verbose=False):
    str1 = get_answer(str1)

    if str1 is None and str2 is None:
        print("WARNING: Both None")
        return True
    if str1 is None or str2 is None:
        return False

    str1 = _strip_string(str1)
    str2 = _strip_string(str2)

    if verbose:
        print(f"Comparing: {str1} vs {str2}")
    
    return str1 == str2 or str1 in str2

