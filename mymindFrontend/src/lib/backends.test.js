import test from 'node:test'
import assert from 'node:assert/strict'

import { normalizeChatResponse } from './backends.js'

test('normalizes rich Python diagnostics without changing legacy fields', () => {
  const result = normalizeChatResponse('python', {
    conv_id: 'c1', response: 'ok', intent: 'refund', intent_group: 'billing',
    agent_type: 'billing', agent_types: ['billing', 'technical'],
    primary_agent: 'billing', supporting_agents: ['technical'],
    routing_reason: 'intent=refund', routing_confidence: 0.9,
    entities: { order_id: ['A123'] }, intent_confidence: 0.8,
    intent_source_scores: { llm: 0.8 }, knowledge_used: true,
    knowledge_status: 'used', knowledge_reason: 'intent:refund'
  })

  assert.equal(result.conversationId, 'c1')
  assert.equal(result.agentType, 'billing')
  assert.deepEqual(result.supportingAgents, ['technical'])
  assert.equal(result.knowledgeStatus, 'used')
})

test('keeps Java legacy responses compatible when diagnostics are absent', () => {
  const result = normalizeChatResponse('java', {
    conversationId: 'j1', response: 'ok', intent: 'billing', agentType: 'billing',
    knowledgeUsed: false, verified: true, grounded: true
  })

  assert.equal(result.primaryAgent, 'billing')
  assert.deepEqual(result.agentTypes, ['billing'])
  assert.equal(result.knowledgeStatus, 'skipped')
  assert.deepEqual(result.entities, {})
})
