"""在服务器原正式账本追加本次授权，保留完整前态和原消耗事件。"""
import json
import os
from pathlib import Path


def main():
    root = Path(os.environ["MYMIND_WORK_ROOT"])
    run = root / "runs/stage8-20261016"
    ledger = root / "runs/stage7-preparation-20261014/formal/limits.json"
    if (run / "authorization.json").exists():
        raise RuntimeError("阶段8授权已接入，不能重复追加")
    before = json.loads(ledger.read_text(encoding="utf-8"))
    request = json.loads((run / "authorization-request.json").read_text(encoding="utf-8"))
    (run / "limits-before.json").write_bytes(ledger.read_bytes())
    request["consumed_before"] = dict(before["consumed"])
    request["events_before"] = len(before["events"])
    request["limits_before"] = dict(before["limits"])
    for key, amount in request["additional_limits"].items():
        before["limits"][key] += amount
    before["events"].append({"kind": "authorization", "stage": "8A+8B", "date": request["date"],
                             "additional_limits": request["additional_limits"], "source": request["source"]})
    for path, data in ((run / "authorization.json", request), (ledger, before)):
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"limits": before["limits"], "consumed": before["consumed"], "events": len(before["events"])}, ensure_ascii=False))


if __name__ == "__main__":
    main()
