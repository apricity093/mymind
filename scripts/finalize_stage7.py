"""固定当前源码基线；历史修复快照与大体积数据保持原位。"""
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT / "agentic_runs/stage7-preparation-20261014"


def write(path,value):
    path.write_text(json.dumps(value,ensure_ascii=False,indent=2),encoding="utf-8")


def main():
    files=(ROOT / "scripts/stage7_sync_files.txt").read_text(encoding="utf-8").splitlines()
    for name in files:
        if not name or name.startswith("#"):
            continue
        target=RUN / "source" / name
        target.parent.mkdir(parents=True,exist_ok=True)
        shutil.copy2(ROOT / name,target)
    active={"revision":"stage7-baseline-20261014-v13-v5-158", "directory":"stage7-preparation-20261014",
        "baseline":"baseline.json", "source_files":files,
        "status":"修复有限复核通过，正式入口及有限预检已交付；RL更新关键证据未成立", "grading_revision":"pre-stage7-grading-v13-20261014",
        "standard":"pre-stage7-fixes-20261014/standards/formal/private_standards.v5.json",
        "fixed_pool":158,"formal_training_executed":False,"final_test_policy_runs":0,
        "new_independent_review":False,"date":"2026-10-14", "preflight_passed":False,
        "preflight_result":"stage7-preparation-20261014/preflight-result.json"}
    write(ROOT / "agentic_runs/active_run_baseline.json",active)
    pointer=ROOT / "agentic_runs/active_fix_candidate.json"
    candidate=json.loads(pointer.read_text(encoding="utf-8"))
    candidate.update(status="已验收，正式运行基线已固定（本会话有限复核）",
        accepted_run_baseline=active["revision"], server_source_data_updated=True,
        acceptance="用户授权本会话F1有限复核及版本收尾；F2/F3沿用此前独立源码/离线复审；本会话新增训练入口为自验",
        acceptance_record="agentic_runs/stage7-preparation-20261014/READY.md",
        stage7_preparation_authorized=True, formal_stage7_training_authorized=False)
    write(pointer,candidate)
    write(RUN / "FIX_ACCEPTANCE.json",dict(revision=candidate["revision"],baseline=active["revision"],
        F1="103项复跑、源码及1037零差异回执对应；补充缺标准不推断unknown对照通过",
        F2_F3="此前独立复审通过的源码与离线范围",new_independent_review=False,
        original_candidate_manifest="pre-stage7-fixes-20261014/scope-fix-v2/candidate_manifest.json",
        current_training_entrypoints="stage7-preparation-20261014/source"))


if __name__=="__main__":
    main()
