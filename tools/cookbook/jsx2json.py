"""Convert SGLang cookbook config/benchmark JSX object literals into JSON."""
import json
import re
import sys


def tokenize_to_json(src: str) -> str:
    out = []
    i, n = 0, len(src)
    while i < n:
        c = src[i]
        if src.startswith("//", i):
            j = src.find("\n", i)
            i = n if j < 0 else j
            continue
        if src.startswith("/*", i):
            j = src.find("*/", i + 2)
            i = n if j < 0 else j + 2
            continue
        if c in "\"'`":
            q = c
            j = i + 1
            buf = []
            while j < n and src[j] != q:
                if src[j] == "\\" and j + 1 < n:
                    nxt = src[j + 1]
                    if nxt == "\n":  # line continuation inside template literal
                        buf.append("\\\n")
                    else:
                        buf.append("\\" + nxt)
                    j += 2
                    continue
                buf.append(src[j])
                j += 1
            raw = "".join(buf)
            # decode JS escapes loosely, then re-encode as JSON
            try:
                val = json.loads('"' + raw.replace('"', '\\"').replace("\\'", "'").replace("\n", "\\n").replace("\\`", "`") + '"')
            except Exception:
                val = raw
            out.append(json.dumps(val))
            i = j + 1
            continue
        m = re.match(r"[A-Za-z_$][\w$]*", src[i:])
        if m:
            word = m.group(0)
            k = i + len(word)
            rest = src[k:].lstrip()
            if rest.startswith(":") and not rest.startswith("::"):
                out.append(json.dumps(word))
            elif word in ("true", "false", "null"):
                out.append(word)
            elif word == "undefined":
                out.append("null")
            else:
                out.append(json.dumps("$" + word))  # identifier reference
            i = k
            continue
        out.append(c)
        i += 1
    s = "".join(out)
    s = re.sub(r",(\s*[}\]])", r"\1", s)
    return s


def extract(path: str, name: str):
    src = open(path, encoding="utf-8").read()
    m = re.search(r"export\s+const\s+" + name + r"\s*=\s*", src)
    if not m:
        return None
    start = m.end()
    # find matching bracket
    open_c = src[start]
    close_c = {"{": "}", "[": "]"}[open_c]
    depth, i, in_str = 0, start, None
    while i < len(src):
        c = src[i]
        if in_str:
            if c == "\\":
                i += 2
                continue
            if c == in_str:
                in_str = None
        elif src.startswith("//", i):
            i = src.find("\n", i)
            continue
        elif c in "\"'`":
            in_str = c
        elif c == open_c:
            depth += 1
        elif c == close_c:
            depth -= 1
            if depth == 0:
                break
        i += 1
    literal = src[start : i + 1]
    return json.loads(tokenize_to_json(literal))


if __name__ == "__main__":
    path, name = sys.argv[1], sys.argv[2]
    print(json.dumps(extract(path, name), ensure_ascii=False, indent=1))
