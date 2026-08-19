<template>
  <main class="app-shell">
    <aside class="sidebar">
      <section class="brand">
        <div class="brand-mark">EM</div>
        <div>
          <h1>mymind Console</h1>
          <p>统一调试 Python 与 Java 版本</p>
        </div>
      </section>

      <a class="profile-card" href="https://xhslink.com/m/558VOQs4Otc" target="_blank" rel="noreferrer">
        <span>我的主页</span>
        <strong>小红书 69.6K 次赞与收藏</strong>
        <em>来看看我的主页 &gt;&gt;</em>
      </a>

      <section class="panel">
        <div class="panel-heading">
          <h2>后端</h2>
          <span class="pill">{{ currentBackend.label }}</span>
        </div>
        <div class="segmented">
          <button :class="{ active: settings.backend === 'java' }" @click="switchBackend('java')">Java</button>
          <button :class="{ active: settings.backend === 'python' }" @click="switchBackend('python')">Python</button>
        </div>

        <label>
          <span>Java API</span>
          <input v-model="settings.endpoints.java" @change="persist" placeholder="/api/java" />
        </label>
        <label>
          <span>Python API</span>
          <input v-model="settings.endpoints.python" @change="persist" placeholder="/api/python" />
        </label>
        <label>
          <span>用户 ID</span>
          <input v-model="settings.userId" @change="persist" placeholder="u1001" />
        </label>
        <label>
          <span>会话 ID</span>
          <input v-model="settings.conversationId" @change="persist" placeholder="自动生成" />
        </label>

        <div class="actions">
          <button @click="checkHealth">健康检查</button>
          <button @click="loadStats">刷新状态</button>
        </div>
      </section>

      <section class="panel status-panel">
        <div class="panel-heading">
          <h2>状态</h2>
          <span :class="['status-dot', healthOk ? 'online' : 'offline']"></span>
        </div>
        <dl>
          <div>
            <dt>当前后端</dt>
            <dd>{{ currentBackend.label }}</dd>
          </div>
          <div>
            <dt>健康状态</dt>
            <dd :class="healthOk ? 'ok' : 'muted'">{{ healthLabel }}</dd>
          </div>
          <div>
            <dt>知识片段</dt>
            <dd>{{ knowledgeCount }}</dd>
          </div>
        </dl>
        <pre v-if="statusText">{{ statusText }}</pre>
      </section>
    </aside>

    <section class="workspace">
      <header class="workspace-header">
        <div>
          <span class="eyebrow">mymind Workspace</span>
          <h2>对话调试</h2>
          <p>{{ currentBackend.baseUrl }}</p>
        </div>
        <div class="header-actions">
          <a class="profile-link" href="https://xhslink.com/m/558VOQs4Otc" target="_blank" rel="noreferrer">小红书主页</a>
          <a :href="docsUrl" target="_blank" rel="noreferrer">API 文档</a>
        </div>
      </header>

      <section class="chat-panel">
        <div class="messages" ref="messageList">
          <article v-for="item in messages" :key="item.id" :class="['message', item.role]">
            <div class="message-meta">
              <span>{{ item.role === 'user' ? '用户' : currentBackend.label }}</span>
              <small v-if="item.meta">{{ item.meta }}</small>
            </div>
            <p>{{ item.content }}</p>
            <details v-if="item.diagnostics" class="diagnostics">
              <summary>诊断详情</summary>
              <dl>
                <div><dt>意图分组</dt><dd>{{ item.diagnostics.intentGroup }}</dd></div>
                <div><dt>主 Agent</dt><dd>{{ item.diagnostics.primaryAgent || '-' }}</dd></div>
                <div><dt>辅助 Agent</dt><dd>{{ item.diagnostics.supportingAgents.join(', ') || '-' }}</dd></div>
                <div><dt>意图置信度</dt><dd>{{ formatConfidence(item.diagnostics.intentConfidence) }}</dd></div>
                <div><dt>路由置信度</dt><dd>{{ formatConfidence(item.diagnostics.routingConfidence) }}</dd></div>
                <div><dt>知识检索</dt><dd>{{ item.diagnostics.knowledgeStatus }} · {{ item.diagnostics.knowledgeReason || '-' }}</dd></div>
              </dl>
              <div v-if="item.diagnostics.routingReason" class="diagnostic-block">
                <strong>路由原因</strong>
                <p>{{ item.diagnostics.routingReason }}</p>
              </div>
              <div v-if="hasValues(item.diagnostics.entities)" class="diagnostic-block">
                <strong>实体</strong>
                <pre>{{ formatJson(item.diagnostics.entities) }}</pre>
              </div>
              <div v-if="hasValues(item.diagnostics.intentSourceScores)" class="diagnostic-block">
                <strong>意图来源分数</strong>
                <pre>{{ formatJson(item.diagnostics.intentSourceScores) }}</pre>
              </div>
            </details>
          </article>
          <div v-if="messages.length === 0" class="empty-state">
            <h3>开始一次客服对话</h3>
            <p>可切换 Java 或 Python 后端，前端会自动适配响应字段。</p>
          </div>
        </div>

        <form class="composer" @submit.prevent="sendMessage">
          <textarea v-model="draft" rows="3" placeholder="输入问题，例如：我想申请退款，订单号是 #12345"></textarea>
          <button :disabled="busy || !draft.trim()">{{ busy ? '发送中' : '发送' }}</button>
        </form>
      </section>

      <section class="tools-grid">
        <article class="tool-panel">
          <div class="panel-heading">
            <h2>知识库检索</h2>
            <span class="pill soft">RAG</span>
          </div>
          <div class="inline-form">
            <input v-model="searchQuery" placeholder="退款多久能到账" />
            <label class="topk-control">
              <span>Top-K</span>
              <input v-model.number="searchTopK" type="number" min="1" max="20" step="1" />
            </label>
            <button @click="searchKnowledge" :disabled="busy || !searchQuery.trim()">检索</button>
          </div>
          <div class="result-list">
            <article v-for="item in searchResults" :key="item.id" class="result-item">
              <div class="result-heading">
                <strong>{{ item.title || '未命名结果' }}</strong>
                <span>score {{ item.score ?? '-' }}</span>
              </div>
              <div v-if="item.sectionPath || item.fusionScore !== null || item.retrievalSources.length" class="result-meta">
                <span v-if="item.sectionPath" class="pill soft">章节 {{ item.sectionPath }}</span>
                <span v-if="item.fusionScore !== null" class="pill soft">fusion {{ item.fusionScore }}</span>
                <span v-if="item.retrievalSources.length" class="pill soft">来源 {{ item.retrievalSources.join(' + ') }}</span>
              </div>
              <p>{{ item.content }}</p>
            </article>
          </div>
        </article>

        <article class="tool-panel">
          <div class="panel-heading">
            <h2>导入知识</h2>
            <span class="pill soft">Docs</span>
          </div>
          <label>
            <span>标题</span>
            <input v-model="docTitle" placeholder="退款补充政策" />
          </label>
          <label>
            <span>内容</span>
            <textarea v-model="docContent" rows="5" placeholder="输入知识库内容"></textarea>
          </label>
          <div class="actions">
            <button @click="submitKnowledge" :disabled="busy || !docTitle.trim() || !docContent.trim()">添加文档</button>
            <label class="file-button">
              上传文件
              <input type="file" accept=".txt,.md,.json" @change="handleUpload" />
            </label>
          </div>
        </article>
      </section>
    </section>
  </main>
</template>

<script setup>
import { computed, nextTick, onMounted, reactive, ref, watch } from 'vue'
import {
  addKnowledge,
  backendMeta,
  createInitialSettings,
  requestChat,
  requestHealth,
  requestKnowledgeStats,
  requestMonitor,
  requestSearch,
  saveSettings,
  uploadKnowledge
} from './lib/backends'

const settings = reactive(createInitialSettings())
const messages = ref([])
const draft = ref('')
const busy = ref(false)
const healthOk = ref(false)
const healthLabel = ref('未检查')
const statusText = ref('')
const knowledgeCount = ref('-')
const searchQuery = ref('退款多久能到账')
const searchTopK = ref(5)
const searchResults = ref([])
const docTitle = ref('退款补充政策')
const docContent = ref('大促期间退款审核时间可能延长到 3-5 个工作日。')
const messageList = ref(null)

const currentBackend = computed(() => backendMeta(settings.backend, settings))
const docsUrl = computed(() => {
  if (settings.backend === 'java') return `${currentBackend.value.baseUrl}/docs`
  return `${currentBackend.value.baseUrl}/docs`
})

watch(
  () => settings.conversationId,
  () => persist()
)

onMounted(() => {
  checkHealth()
  loadStats()
})

function switchBackend(type) {
  settings.backend = type
  persist()
  healthOk.value = false
  healthLabel.value = '未检查'
  statusText.value = ''
  searchResults.value = []
  checkHealth()
}

function persist() {
  saveSettings(settings)
}

async function sendMessage() {
  const content = draft.value.trim()
  if (!content) return
  messages.value.push({ id: crypto.randomUUID(), role: 'user', content })
  draft.value = ''
  busy.value = true
  try {
    const response = await requestChat(settings.backend, settings, content)
    if (response.conversationId && !settings.conversationId) {
      settings.conversationId = response.conversationId
      persist()
    }
    const meta = [
      response.intent,
      response.primaryAgent || response.agentType,
      response.supportingAgents.length ? `协作 ${response.supportingAgents.join(', ')}` : '',
      response.knowledgeStatus === 'used' ? 'RAG' : response.knowledgeStatus,
      response.escalated ? '转人工' : ''
    ].filter(Boolean).join(' · ')
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      content: response.response,
      meta,
      diagnostics: settings.backend === 'python' ? {
        intentGroup: response.intentGroup,
        entities: response.entities,
        intentConfidence: response.intentConfidence,
        intentSourceScores: response.intentSourceScores,
        primaryAgent: response.primaryAgent,
        supportingAgents: response.supportingAgents,
        routingReason: response.routingReason,
        routingConfidence: response.routingConfidence,
        knowledgeStatus: response.knowledgeStatus,
        knowledgeReason: response.knowledgeReason
      } : null
    })
  } catch (error) {
    messages.value.push({
      id: crypto.randomUUID(),
      role: 'assistant',
      content: error.message,
      meta: '请求失败'
    })
  } finally {
    busy.value = false
    await nextTick()
    messageList.value?.scrollTo({ top: messageList.value.scrollHeight, behavior: 'smooth' })
  }
}

function formatConfidence(value) {
  return Number.isFinite(value) ? `${(value * 100).toFixed(1)}%` : '-'
}

function hasValues(value) {
  return value && Object.values(value).some((item) => Array.isArray(item) ? item.length : item !== null && item !== '')
}

function formatJson(value) {
  return JSON.stringify(value, null, 2)
}

async function checkHealth() {
  try {
    const data = await requestHealth(settings.backend, settings)
    healthOk.value = data.status === 'ok'
    healthLabel.value = data.status || 'ok'
    statusText.value = JSON.stringify(data, null, 2)
  } catch (error) {
    healthOk.value = false
    healthLabel.value = '不可用'
    statusText.value = error.message
  }
}

async function loadStats() {
  try {
    const [stats, monitor] = await Promise.allSettled([
      requestKnowledgeStats(settings.backend, settings),
      requestMonitor(settings.backend, settings)
    ])
    if (stats.status === 'fulfilled') {
      knowledgeCount.value = stats.value.total_chunks ?? stats.value.totalChunks ?? '-'
    }
    if (monitor.status === 'fulfilled') {
      statusText.value = JSON.stringify(monitor.value, null, 2)
    }
  } catch (error) {
    statusText.value = error.message
  }
}

async function searchKnowledge() {
  busy.value = true
  try {
    const topK = Number.isInteger(searchTopK.value) ? Math.min(20, Math.max(1, searchTopK.value)) : 5
    const data = await requestSearch(settings.backend, settings, searchQuery.value, topK)
    searchResults.value = data.results || []
    statusText.value = JSON.stringify({
      backend: data.backend,
      requestedTopK: data.requestedTopK ?? topK,
      returned: data.returned,
      reranked: data.reranked
    }, null, 2)
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}

async function submitKnowledge() {
  busy.value = true
  try {
    const data = await addKnowledge(settings.backend, settings, [
      { title: docTitle.value.trim(), content: docContent.value.trim() }
    ])
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}

async function handleUpload(event) {
  const file = event.target.files?.[0]
  event.target.value = ''
  if (!file) return
  busy.value = true
  try {
    const data = await uploadKnowledge(settings.backend, settings, file)
    statusText.value = JSON.stringify(data, null, 2)
    await loadStats()
  } catch (error) {
    statusText.value = error.message
  } finally {
    busy.value = false
  }
}
</script>
