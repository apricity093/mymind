package com.mymind.agent;

import com.mymind.intent.IntentCategory;

public record OrchestratorResult(
        String requestId,
        String response,
        AgentType agentType,
        IntentCategory intent,
        boolean escalated,
        long latencyMs
) {
}
