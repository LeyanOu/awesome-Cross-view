import re

def renumber_headings(content):
    lines = content.splitlines()
    new_lines = []
    level3_counter = 0

    for line in lines:
        if line.startswith('## '):
            level3_counter = 0
            new_lines.append(line)
        elif line.startswith('### '):
            level3_counter += 1
            line_without_number = re.sub(r'^(###\s*)(\d+\.\s*)?', r'\1', line)
            new_line = re.sub(r'^(###\s*)(.*)', rf'\1{level3_counter}. \2', line_without_number)
            new_lines.append(new_line)
        else:
            new_lines.append(line)
    return "\n".join(new_lines)

if __name__ == "__main__":
    md_file = "README.md"
    with open(md_file, "r", encoding="utf-8") as f:
        content = f.read()
    new_content = renumber_headings(content)
    with open(md_file, "w", encoding="utf-8") as f:
        f.write(new_content)
    print("Automatically updated level 3 headings numbering under each level 2 heading.")
