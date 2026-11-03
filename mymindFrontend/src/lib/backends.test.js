import test from 'node:test'
import assert from 'node:assert/strict'

import {
  normalizeChatResponse,
  normalizeSearchResponse,
  normalizeSearchResults,
  requestSearch,
  requestChat
} from './backends.js'

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
    conversation_id: 'j1', response: 'ok', intent: 'billing', agent_type: 'billing',
    knowledge_used: false, verified: true, grounded: true
  })

  assert.equal(result.primaryAgent, 'billing')
  assert.deepEqual(result.agentTypes, ['billing'])
  assert.equal(result.knowledgeStatus, 'skipped')
  assert.deepEqual(result.entities, {})
})

test('normalizes Python search chunk fields to one stable id', () => {
  const results = normalizeSearchResults([
    { title: '退款政策', content: '七天内退款', score: 0.91, chunk: 1, chunk_id: 'chk-1', source_id: 'src-1', section_path: '退款/时效', fusion_score: 0.021, retrieval_sources: ['vector', 'bm25'] }
  ])

  assert.equal(results[0].id, 'chk-1')
  assert.equal(results[0].chunkId, 'chk-1')
  assert.equal(results[0].sourceId, 'src-1')
  assert.equal(results[0].sectionPath, '退款/时效')
  assert.equal(results[0].fusionScore, 0.021)
  assert.deepEqual(results[0].retrievalSources, ['vector', 'bm25'])
})

test('normalizes Java old search responses without new fields', () => {
  const results = normalizeSearchResults([
    { title: '退款政策', content: '七天内退款', score: 0.91, chunk: 1 }
  ])

  assert.match(results[0].id, /^search-result-0$/)
  assert.equal(results[0].sourceId, '')
  assert.deepEqual(results[0].retrievalSources, [])
})

test('normalizeSearchResponse keeps old query/results/reranked contract', () => {
  const result = normalizeSearchResponse('python', {
    query: '退款多久到账',
    results: [{ title: '退款政策', content: '5-7 个工作日', score: 0.8, chunk: 0 }],
    reranked: true,
    requested_top_k: 7,
    index_version: 'rag-index-v1'
  })

  assert.equal(result.query, '退款多久到账')
  assert.equal(result.results.length, 1)
  assert.equal(result.reranked, true)
  assert.equal(result.requestedTopK, 7)
  assert.equal(result.indexVersion, 'rag-index-v1')
})

test('requestSearch sends top_k to Python and topK to Java', async () => {
  const calls = []
  const originalFetch = global.fetch
  global.fetch = async (url, options) => {
    calls.push([String(url), options])
    return { ok: true, status: 200, statusText: 'OK', text: async () => '{"query":"q","results":[],"reranked":false}' }
  }
  try {
    await requestSearch('python', { endpoints: { python: 'http://python.test' } }, 'q', 7)
    await requestSearch('java', { endpoints: { java: 'http://java.test' } }, 'q', 7)
  } finally {
    global.fetch = originalFetch
  }

  assert.ok(calls[0][0].includes('top_k=7'))
  assert.ok(calls[1][0].includes('topK=7'))
})


test('chat adapter preserves each backend wire contract and returns one page model', async (t) => {
  const calls = []
  t.mock.method(global, 'fetch', async (url, options) => {
    calls.push({ url, payload: JSON.parse(options.body) })
    const raw = url.includes('/python/')
      ? { conv_id: 'p-next', request_id: 'trace-1', response: 'ok', agent_type: 'billing' }
      : { conversation_id: 'j-next', response: 'ok', agent_type: 'billing', verified: true }
    return { ok: true, text: async () => JSON.stringify(raw) }
  })
  const settings = { userId: 'u1', conversationId: 'existing', endpoints: {
    python: 'http://test/python/', java: 'http://test/java/'
  } }
  const python = await requestChat('python', settings, '退款')
  const java = await requestChat('java', settings, '退款')
  assert.deepEqual(calls, [
    { url: 'http://test/python/chat', payload: { message: '退款', user_id: 'u1', conv_id: 'existing' } },
    { url: 'http://test/java/chat', payload: { message: '退款', user_id: 'u1', conversation_id: 'existing' } }
  ])
  assert.equal(python.conversationId, 'p-next')
  assert.equal(java.conversationId, 'j-next')
  assert.equal(python.requestId, 'trace-1')
  assert.equal(java.verified, true)
  assert.equal(python.agentType, java.agentType)
})
