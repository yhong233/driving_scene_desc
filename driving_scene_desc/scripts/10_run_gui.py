from __future__ import annotations

from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"


def main():
    if not APP_PATH.exists():
        raise FileNotFoundError(f"未找到 Streamlit 界面文件: {APP_PATH}")

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(APP_PATH),
        "--server.address",
        "localhost",
        "--server.port",
        "8501",
    ]

    print("正在启动可视化界面...")
    print("运行命令:", " ".join(cmd))
    print("如果浏览器没有自动打开，请访问: http://localhost:8501")

    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=False)


if __name__ == "__main__":
    main()
