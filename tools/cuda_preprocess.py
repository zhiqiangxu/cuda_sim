#!/usr/bin/env python3
"""Preprocessor: converts CUDA <<<>>> launch syntax to _launch() calls.

Usage:
    python cuda_preprocess.py input.cpp -o output.cpp

Transforms:
    kernel<<<grid, block>>>(a, b, c);
    kernel<<<grid, block, shared_bytes>>>(a, b, c);
Into:
    kernel_launch(a, b, c, grid, block);
    kernel_launch(a, b, c, grid, block, shared_bytes);
"""

import argparse
import re
import sys


def preprocess(text: str) -> str:
    """Replace <<<>>> launch syntax with _launch() calls."""

    # Pattern: identifier<<<expr, expr[, expr]>>>(args);
    # We need to handle nested parentheses in args and nested <> or commas in template params.
    # Strategy: find <<<, then >>>, then the matching (...);

    result = []
    i = 0
    while i < len(text):
        # Look for <<<
        match = re.search(r'(\w+)\s*<<<', text[i:])
        if not match:
            result.append(text[i:])
            break

        # Append everything before the match
        start = i + match.start()
        result.append(text[i:start])

        kernel_name = match.group(1)
        pos = i + match.end()  # position after <<<

        # Parse launch config: grid, block[, shared_bytes] up to >>>
        config_start = pos
        depth = 1  # we're inside one level of <<<
        while pos < len(text) and depth > 0:
            if text[pos:pos+3] == '<<<':
                depth += 1
                pos += 3
            elif text[pos:pos+3] == '>>>':
                depth -= 1
                if depth == 0:
                    break
                pos += 3
            else:
                pos += 1

        config = text[config_start:pos].strip()
        pos += 3  # skip >>>

        # Skip whitespace
        while pos < len(text) and text[pos] in ' \t':
            pos += 1

        # Parse arguments: (args);
        if pos < len(text) and text[pos] == '(':
            paren_start = pos + 1
            paren_depth = 1
            pos += 1
            while pos < len(text) and paren_depth > 0:
                if text[pos] == '(':
                    paren_depth += 1
                elif text[pos] == ')':
                    paren_depth -= 1
                pos += 1
            args = text[paren_start:pos-1].strip()
        else:
            args = ""

        # Split config into grid, block, [shared_bytes]
        config_parts = [p.strip() for p in config.split(',')]
        grid = config_parts[0] if len(config_parts) > 0 else "1"
        block = config_parts[1] if len(config_parts) > 1 else "1"
        shared = config_parts[2] if len(config_parts) > 2 else None

        # Build _launch() call
        launch_args = []
        if args:
            launch_args.append(args)
        launch_args.append(grid)
        launch_args.append(block)
        if shared:
            launch_args.append(shared)

        result.append(f"{kernel_name}_launch({', '.join(launch_args)})")

        i = pos

    return ''.join(result)


def main():
    parser = argparse.ArgumentParser(
        description="Convert CUDA <<<>>> syntax to _launch() calls")
    parser.add_argument("input", help="Input .cpp/.cu file")
    parser.add_argument("-o", "--output", help="Output file (default: stdout)")
    args = parser.parse_args()

    with open(args.input, "r") as f:
        text = f.read()

    result = preprocess(text)

    if args.output:
        with open(args.output, "w") as f:
            f.write(result)
        print(f"Preprocessed: {args.input} → {args.output}", file=sys.stderr)
    else:
        print(result)


if __name__ == "__main__":
    main()
