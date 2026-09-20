"""对仓库中的 Notebook 做不执行代码的静态检查。"""

from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def iter_code_sources(notebook: dict):
    """逐个返回代码单元的源码。"""
    for index, cell in enumerate(notebook.get("cells", []), start=1):
        if cell.get("cell_type") != "code":
            continue
        source = cell.get("source", [])
        yield index, "".join(source) if isinstance(source, list) else str(source)


def python_only_source(source: str) -> str:
    """移除只能由 IPython 解释的行魔法和 shell 命令。"""
    return "\n".join(
        line
        for line in source.splitlines()
        if not line.lstrip().startswith(("%", "!"))
    )


def main() -> int:
    """检查 JSON、代码语法和 Notebook 编号。"""
    notebook_paths = sorted(ROOT.glob("[0-9][0-9]*.ipynb"))
    if not notebook_paths:
        raise SystemExit("没有找到编号 Notebook。")

    failures: list[str] = []
    checked_cells = 0

    for path in notebook_paths:
        try:
            notebook = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{path.name}: JSON 无法读取：{exc}")
            continue

        if notebook.get("nbformat") != 4:
            failures.append(f"{path.name}: nbformat 不是 4")

        for cell_index, source in iter_code_sources(notebook):
            checked_cells += 1
            source = python_only_source(source)
            if not source.strip():
                continue
            try:
                ast.parse(source)
            except SyntaxError as exc:
                failures.append(
                    f"{path.name} 第 {cell_index} 个 cell：{exc.msg}，行 {exc.lineno}"
                )

    if failures:
        print("静态检查发现问题：")
        for failure in failures:
            print(f"- {failure}")
        return 1

    print(
        f"静态检查通过：{len(notebook_paths)} 个 Notebook，"
        f"{checked_cells} 个代码单元。"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
