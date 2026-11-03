"""用学生模板导出真实合格轨迹及助手监督区间。"""
from copy import deepcopy
import json

from agents.customer_policy.generation import TOOLS
from agents.customer_policy.protocol import VisibleTrajectory
from rl.grading import grade_result
from rl.runtime import initialize, write_json


def student_messages(row):
    messages = []
    for original in row["messages"]:
        message = {"role": original["role"], "content": original.get("content") or ""}
        if original.get("tool_calls"):
            message["tool_calls"] = deepcopy(original["tool_calls"])
            for call in message["tool_calls"]:
                arguments = call["function"]["arguments"]
                if isinstance(arguments, str):
                    call["function"]["arguments"] = json.loads(arguments)
        if message["role"] == "tool":
            message["name"] = original["name"]
            message["tool_call_id"] = original["tool_call_id"]
            if not isinstance(original["content"], str):
                raise ValueError("工具观察须为 JSON 字符串")
            json.loads(original["content"])
        messages.append(message)
    return messages


def encode_trajectory(tokenizer, messages):
    def encode(items, generation=False):
        return tokenizer.apply_chat_template(items, tools=TOOLS, tokenize=True,
                                             add_generation_prompt=generation, return_dict=False)
    full = encode(messages)
    labels = [-100] * len(full)
    spans = []
    for index, message in enumerate(messages):
        if message["role"] != "assistant":
            continue
        prefix = encode(messages[:index], True)
        through = encode(messages[:index + 1])
        if full[:len(prefix)] != prefix or full[:len(through)] != through:
            position = next((i for i, (a, b) in enumerate(zip(full, through)) if a != b), len(prefix))
            raise ValueError(f"学生模板边界不一致：message={index}, prefix={len(prefix)}, through={len(through)}, "
                             f"position={position}, full={tokenizer.decode(full[max(0, position-20):position+20])!r}, "
                             f"partial={tokenizer.decode(through[max(0, position-20):position+20])!r}")
        start, end = len(prefix), len(through)
        labels[start:end] = full[start:end]
        spans.append({"message_index": index, "start": start, "end": end,
                      "kind": "tool_action" if message.get("tool_calls") else "final",
                      "text": tokenizer.decode(full[start:end])})
    if not spans or spans[-1]["kind"] != "final":
        raise ValueError("完整轨迹缺少助手终止输出")
    return {"input_ids": full, "labels": labels, "attention_mask": [1] * len(full), "spans": spans}


def main():
    from transformers import AutoTokenizer
    root, config, output, tasks, standards, clauses = initialize()
    tokenizer = AutoTokenizer.from_pretrained(root / config["model_dir"], local_files_only=True)
    rows = json.loads((root / config["trajectories"]).read_text(encoding="utf-8"))
    exported, seen = [], set()
    for row in rows:
        task_id = row["task_id"]
        if task_id not in config["task_ids"]:
            continue
        allowed_split = "train" if config.get("dataset_mode") in {"calibration", "formal"} else "dev"
        if task_id in seen or standards[task_id].split != allowed_split:
            raise ValueError("导出要求每题一条、符合配置允许的划分")
        grade = grade_result(standards[task_id], row["raw_result"],
                             VisibleTrajectory.model_validate(row["trajectory"]), clauses)
        if not grade.complete_success or row["infrastructure_failure"] or row["runtime_error"]:
            raise ValueError("只导出通过验收的真实完整轨迹")
        messages = student_messages(row)
        public = tasks[task_id].messages()
        same_public_input = messages[:2] == public
        if config.get("dataset_mode") == "calibration":
            # 校准保留提示修订前的首条合格轨迹；导出实际使用的提示和观察。
            same_public_input = (messages[1] == public[1] and messages[0]["role"] == "system" and
                messages[0]["content"].startswith("role:1\n" + tasks[task_id].materials()[0].content + "\n"))
        if not same_public_input:
            raise ValueError("教师公开输入与当前任务不一致")
        encoded = encode_trajectory(tokenizer, messages)
        if len(encoded["input_ids"]) > config["context_tokens"]:
            raise ValueError("完整示范超出上下文，不截断监督目标")
        exported.append({"task_id": task_id, "messages": messages, **encoded})
        seen.add(task_id)
    if seen != set(config["task_ids"]):
        raise ValueError("配置任务池缺少合格示范")
    write_json(output / "student_data.json", exported)
    write_json(output / "mask_audit.json", [{"task_id": row["task_id"],
        "tokens": len(row["input_ids"]), "supervised_tokens": sum(x != -100 for x in row["labels"]),
        "roles": [m["role"] for m in row["messages"]], "spans": row["spans"],
        "context_tokens": config["context_tokens"]} for row in exported])
    (output / "chat_template.jinja").write_text(tokenizer.chat_template, encoding="utf-8")
    label = config.get("dataset_mode", "development")
    print(f"exported {len(exported)} accepted {label} trajectories", flush=True)


if __name__ == "__main__":
    main()
