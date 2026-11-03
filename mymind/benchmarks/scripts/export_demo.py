"""导出独立策略服务的配置与请求示例。"""
import argparse
import json
import os
from pathlib import Path


def main():
    source = Path(__file__).resolve().parents[1] / "templates/demo"
    root = Path(os.environ.get("MYMIND_WORK_ROOT", Path(__file__).resolve().parents[3]))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "output/demo")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        (args.output_dir / path.name).write_text(
            json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(args.output_dir)


if __name__ == "__main__":
    main()
