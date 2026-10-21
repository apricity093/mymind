"""受控故障验收；真实 rollout 完成后标记实验故障，不伪造服务记录。"""
import os
from experiments.agentic_rl.grpo import main


if __name__ == "__main__":
    mode = os.environ["MYMIND_FAULT_VALIDATION"]
    if mode not in {"recover", "repeat"}:
        raise ValueError("故障验收模式须为 recover 或 repeat")
    main(fault_injector=lambda attempt, step: step == 0 and (mode == "repeat" or attempt == 0))
