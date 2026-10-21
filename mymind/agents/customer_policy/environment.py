"""单任务环境；不读取答案、判分或训练资源。"""
from agents.customer_policy.protocol import SearchAction, SearchObservation, SearchTurn, ToolStatus, VisibleTrajectory


class PolicyEnvironment:
    max_searches = 3

    def __init__(self, retrieval):
        self.retrieval = retrieval

    def reset(self, task):
        self.task = task
        self.searches = []
        self.diagnostics = []
        return task.messages(max_searches=self.max_searches)

    async def step(self, name, arguments):
        number = len(self.searches) + 1
        if number > self.max_searches:
            raise ValueError("检索预算已用尽")
        try:
            if name != "search":
                raise ValueError("未知工具")
            action = SearchAction.model_validate(arguments)
        except ValueError:
            action = arguments
            observation = SearchObservation(request_number=number,
                status=ToolStatus(reference_id=f"search:{number}:status", state="invalid_arguments",
                                  detail="只支持 search，参数为非空 query 字符串"), snippets=[])
            diagnostics = {"invalid_tool_name": name}
        else:
            observation, diagnostics = await self.retrieval.search(action.query, number)
        self.searches.append(SearchTurn(action=action, observation=observation))
        self.diagnostics.append(diagnostics)
        return observation

    def trajectory(self):
        return VisibleTrajectory(task_id=self.task.task_id, materials=self.task.materials(), searches=self.searches)
