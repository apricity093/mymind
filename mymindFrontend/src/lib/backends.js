const ENV = import.meta.env || {}

const DEFAULT_BACKENDS = {
  python: {
    id: 'python',
    label: 'Python',
    baseUrl: ENV.VITE_PYTHON_API_URL || '/api/python',
    port: '8000'
  },
  java: {
    id: 'java',
    label: 'Java',
    baseUrl: ENV.VITE_JAVA_API_URL || '/api/java',
    port: '8080'
  }
}

export function createInitialSettings() {
  const saved = readSettings()
  return {
    backend: saved.backend || 'java',
    userId: saved.userId || 'u1001',
    conversationId: saved.conversationId || '',
    endpoints: {
      python: saved.endpoints?.python || DEFAULT_BACKENDS.python.baseUrl,
      java: saved.endpoints?.java || DEFAULT_BACKENDS.java.baseUrl
    }
  }
}

export function saveSettings(settings) {
  localStorage.setItem('mymind.frontend.settings', JSON.stringify(settings))
}

export function backendMeta(type, settings) {
  const meta = DEFAULT_BACKENDS[type] || DEFAULT_BACKENDS.java
  return {
    ...meta,
    baseUrl: normalizeBaseUrl(settings.endpoints[type] || meta.baseUrl)
  }
}

export async function requestHealth(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/health')
}

export async function requestMonitor(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/monitor')
}

export async function requestKnowledgeStats(type, settings) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/stats')
}

export async function requestSearch(type, settings, query, topK = 5) {
  // Python /search 使用 top_k，Java 使用 topK。前端必须按后端类型发送正确参数。
  const paramName = type === 'python' ? 'top_k' : 'topK'
  const params = new URLSearchParams({ query, [paramName]: String(topK) })
  const raw = await requestJson(backendMeta(type, settings).baseUrl, `/search?${params}`, { method: 'POST' })
  return normalizeSearchResponse(type, raw)
}

export function normalizeSearchResponse(type, raw) {
  return {
    backend: type,
    query: raw.query ?? '',
    results: normalizeSearchResults(raw.results),
    reranked: Boolean(raw.reranked),
    requestedTopK: raw.requested_top_k ?? raw.requestedTopK ?? null,
    returned: raw.returned ?? (Array.isArray(raw.results) ? raw.results.length : 0),
    indexVersion: raw.index_version ?? raw.indexVersion ?? '',
    chunkConfig: raw.chunk_config ?? raw.chunkConfig ?? null,
    raw
  }
}

export function normalizeSearchResults(items) {
  return (Array.isArray(items) ? items : []).map((item, index) => {
    const id = item.chunk_id ?? item.chunkId ?? item.id ?? `search-result-${index}`
    return {
      id,
      chunkId: id,
      title: item.title ?? '',
      content: item.content ?? '',
      score: toNumber(item.score),
      chunk: toNumber(item.chunk ?? item.chunkIndex ?? item.chunk_index),
      sourceId: item.source_id ?? item.sourceId ?? '',
      sectionPath: item.section_path ?? item.sectionPath ?? '',
      fusionScore: toNullableNumber(item.fusion_score ?? item.fusionScore),
      retrievalSources: item.retrieval_sources ?? item.retrievalSources ?? [],
      indexVersion: item.index_version ?? item.indexVersion ?? '',
      chunkConfig: item.chunk_config ?? item.chunkConfig ?? null,
      raw: item
    }
  })
}

export async function requestChat(type, settings, message) {
  const meta = backendMeta(type, settings)
  const payload = buildChatPayload(type, settings, message)
  const raw = await requestJson(meta.baseUrl, '/chat', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload)
  })
  return normalizeChatResponse(type, raw)
}

export async function addKnowledge(type, settings, documents) {
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/add', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ documents })
  })
}

export async function uploadKnowledge(type, settings, file) {
  const form = new FormData()
  form.append('file', file)
  return requestJson(backendMeta(type, settings).baseUrl, '/knowledge/upload', {
    method: 'POST',
    body: form
  })
}

function buildChatPayload(type, settings, message) {
  if (type === 'python') {
    return {
      message,
      user_id: settings.userId || 'anonymous',
      conv_id: settings.conversationId || undefined
    }
  }
  return {
    message,
    user_id: settings.userId || 'anonymous',
    conversation_id: settings.conversationId || undefined
  }
}

export function normalizeChatResponse(type, raw) {
  const agentType = raw.agent_type || raw.agentType || ''
  const primaryAgent = raw.primary_agent || raw.primaryAgent || agentType
  return {
    backend: type,
    conversationId: raw.conversation_id || raw.conversationId || raw.conv_id || '',
    response: raw.response || '',
    intent: raw.intent || 'other',
    agentType,
    intentGroup: raw.intent_group || raw.intentGroup || raw.intent || 'other',
    entities: raw.entities || {},
    intentConfidence: Number(raw.intent_confidence ?? raw.intentConfidence ?? 0),
    intentSourceScores: raw.intent_source_scores || raw.intentSourceScores || {},
    agentTypes: raw.agent_types || raw.agentTypes || (agentType ? [agentType] : []),
    primaryAgent,
    supportingAgents: raw.supporting_agents || raw.supportingAgents || [],
    routingReason: raw.routing_reason || raw.routingReason || '',
    routingConfidence: Number(raw.routing_confidence ?? raw.routingConfidence ?? 0),
    escalated: Boolean(raw.escalated),
    latencyMs: Number(raw.latency_ms ?? raw.latencyMs ?? 0),
    knowledgeUsed: Boolean(raw.knowledge_used ?? raw.knowledgeUsed),
    knowledgeStatus: raw.knowledge_status || raw.knowledgeStatus || ((raw.knowledge_used ?? raw.knowledgeUsed) ? 'used' : 'skipped'),
    knowledgeReason: raw.knowledge_reason || raw.knowledgeReason || '',
    verified: raw.verified,
    grounded: raw.grounded,
    raw
  }
}

async function requestJson(baseUrl, path, options = {}) {
  const url = `${normalizeBaseUrl(baseUrl)}${path}`
  const response = await fetch(url, options)
  const text = await response.text()
  let data = null
  try {
    data = text ? JSON.parse(text) : null
  } catch {
    data = text
  }
  if (!response.ok) {
    const detail = typeof data === 'string' ? data : JSON.stringify(data)
    throw new Error(`${response.status} ${response.statusText}: ${detail}`)
  }
  return data
}

function normalizeBaseUrl(value) {
  return String(value || '').replace(/\/+$/, '')
}

function toNumber(value) {
  if (value === null || value === undefined || value === '') return 0
  const number = Number(value)
  return Number.isFinite(number) ? number : 0
}

function toNullableNumber(value) {
  if (value === null || value === undefined || value === '') return null
  const number = Number(value)
  return Number.isFinite(number) ? number : null
}

function readSettings() {
  try {
    return JSON.parse(localStorage.getItem('mymind.frontend.settings') || '{}')
  } catch {
    return {}
  }
}
