"""files 包标记。

让 `python -c "from files.tools import ALL_TOOLS"` 这种从仓库根的 import 也能工作。
项目通常的运行方式是 cwd=files/,直接 `import main`;但加这个空 __init__.py 不影响那条路径。
"""
