// agent.js — 智能体

// ══════════════════════════════════════
// AGENT
// ══════════════════════════════════════
let agentList = [];
let activeAgentId = null;
let selectedAgentType = 'multi_agent';
let editingAgentSubs = [];
const TOOL_NAMES = ['calculator','get_current_time','text_analyzer','unit_converter','word_counter','get_weather','search_knowledge_base'];
const AGENT_TYPE_MAP = {
  multi_agent: { label: '多智能体协同Agent', color: 'var(--green)', bg: 'var(--green-dim)', icon: '⬢' },
  autonomous: { label: '自主规划Agent', color: 'var(--orange)', bg: 'var(--orange-dim)', icon: '◈' },
};

async function loadAgentList() {
  try {
    const res = await fetch(API + '/agent');
    const data = await res.json();
    agentList = data.agents || [];
    renderAgentGrid();
    updateAgentCount();
  } catch (e) { toast('加载智能体失败', 'error'); }
}

function renderAgentGrid() {
  const grid = document.getElementById('agent-grid');
  if (!agentList.length) {
    grid.innerHTML = '<div style="color:var(--text3);font-size:12px;padding:40px;text-align:center">暂无智能体，点击右上角创建</div>';
    return;
  }
  grid.innerHTML = agentList.map(a => {
    const t = AGENT_TYPE_MAP[a.agent_type] || AGENT_TYPE_MAP.multi_agent;
    return `<div class="agent-card" onclick="openAgentConfig('${a.id}')">
      <div class="agent-card-icon" style="background:${t.bg};color:${t.color}">${t.icon}</div>
      <div class="agent-card-name">${escHtml(a.name)}</div>
      <div class="agent-card-desc">${escHtml(a.description || '暂无描述')}</div>
      <div class="agent-card-footer">
        <span class="agent-card-tag" style="background:${t.bg};color:${t.color}">${t.label}</span>
      </div>
    </div>`;
  }).join('');
}

function selectAgentType(type) {
  selectedAgentType = type;
  document.querySelectorAll('.agent-type-card').forEach(c => c.classList.toggle('selected', c.dataset.type === type));
}

function openCreateAgentModal() {
  selectedAgentType = 'multi_agent';
  selectAgentType('multi_agent');
  document.getElementById('agent-name-input').value = '';
  document.getElementById('agent-desc-input').value = '';
  openModal('modal-create-agent');
}

async function createAgentAndOpenConfig() {
  const name = document.getElementById('agent-name-input').value.trim();
  if (!name) { toast('请输入名称', 'error'); return; }
  try {
    const res = await fetch(API + '/agent', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ name, description: document.getElementById('agent-desc-input').value.trim(), agent_type: selectedAgentType })
    });
    const data = await res.json();
    closeModal('modal-create-agent');
    toast('智能体已创建');
    await loadAgentList();
    openAgentConfig(data.agent_id);
  } catch (e) { toast('创建失败', 'error'); }
}

// ── Config page ──
function openAgentConfig(id) {
  activeAgentId = id;
  const a = agentList.find(x => x.id === id);
  if (!a) return;
  const t = AGENT_TYPE_MAP[a.agent_type] || AGENT_TYPE_MAP.multi_agent;

  document.getElementById('agent-list-view').style.display = 'none';
  const cv = document.getElementById('agent-config-view');
  cv.classList.add('active');

  document.getElementById('acfg-name').textContent = a.name;
  document.getElementById('acfg-type-badge').textContent = t.label;
  document.getElementById('acfg-type-badge').style.cssText = `background:${t.bg};color:${t.color}`;

  // Fill form
  document.getElementById('acfg-input-name').value = a.name || '';
  document.getElementById('acfg-input-desc').value = a.description || '';
  document.getElementById('acfg-input-model').value = a.model_name || 'claude-sonnet-4-6';
  document.getElementById('acfg-input-rounds').value = a.context_rounds ?? 2;
  document.getElementById('acfg-rounds-val').value = a.context_rounds ?? 2;
  document.getElementById('acfg-input-prompt').value = a.system_prompt || '';
  document.getElementById('acfg-input-greeting').value = a.greeting || '';
  document.getElementById('acfg-preview-name').textContent = a.name;
  document.getElementById('acfg-preview-greeting').textContent = a.greeting || '保存后可在此测试对话';

  // Type-specific sections
  const isMulti = a.agent_type === 'multi_agent';
  document.getElementById('acfg-sub-section').style.display = isMulti ? '' : 'none';
  document.getElementById('acfg-tool-section').style.display = isMulti ? 'none' : '';

  if (isMulti) {
    editingAgentSubs = [...(a.sub_agents || [])];
    document.getElementById('acfg-input-collab').value = a.collaboration_mode || 'sequential';
    renderConfigSubAgents();
  } else {
    renderConfigTools(a.available_tools || TOOL_NAMES);
    document.getElementById('acfg-input-maxsteps').value = a.max_steps || 10;
  }

  // Suggested questions
  renderSuggestQuestions(a.suggested_questions || []);
  switchCfgTab('settings', document.querySelector('.agent-cfg-nav-item'));
}

function closeAgentConfig() {
  document.getElementById('agent-config-view').classList.remove('active');
  document.getElementById('agent-list-view').style.display = '';
  activeAgentId = null;
}

function switchCfgTab(tab, el) {
  document.querySelectorAll('.agent-cfg-tab').forEach(t => t.style.display = 'none');
  document.getElementById('acfg-tab-' + tab).style.display = '';
  document.querySelectorAll('.agent-cfg-nav-item').forEach(n => n.classList.remove('active'));
  if (el) el.classList.add('active');
}

function renderConfigSubAgents() {
  const list = document.getElementById('acfg-sub-list');
  document.getElementById('acfg-sub-count').textContent = editingAgentSubs.length;
  if (!editingAgentSubs.length) {
    list.innerHTML = '<div style="color:var(--text3);font-size:11px;padding:12px;text-align:center">暂无子 Agent，点击添加</div>';
    return;
  }
  list.innerHTML = editingAgentSubs.map(sid => {
    const sub = agentList.find(x => x.id === sid);
    const name = sub ? sub.name : sid;
    return `<div class="agent-cfg-sub-item">
      <div class="agent-cfg-sub-icon">⬢</div>
      <div class="agent-cfg-sub-info"><div class="agent-cfg-sub-name">${escHtml(name)}</div></div>
      <div class="agent-cfg-sub-del" onclick="removeSubAgent('${sid}')">✕</div>
    </div>`;
  }).join('');
}

function removeSubAgent(sid) {
  editingAgentSubs = editingAgentSubs.filter(s => s !== sid);
  renderConfigSubAgents();
}

function openAddSubAgentModal() {
  const available = agentList.filter(a => a.id !== activeAgentId && !editingAgentSubs.includes(a.id));
  if (!available.length) { toast('没有可添加的 Agent'); return; }
  const html = available.map(a => `<div class="agent-cfg-sub-item" style="cursor:pointer" onclick="addSubAgent('${a.id}')">
    <div class="agent-cfg-sub-icon">⬢</div>
    <div class="agent-cfg-sub-info"><div class="agent-cfg-sub-name">${escHtml(a.name)}</div><div class="agent-cfg-sub-desc">${escHtml(a.description||'')}</div></div>
  </div>`).join('');
  document.getElementById('acfg-sub-list').innerHTML += `<div id="acfg-sub-picker" style="border:1px dashed var(--accent);border-radius:8px;padding:8px;margin-top:8px">${html}</div>`;
}

function addSubAgent(sid) {
  if (editingAgentSubs.length >= 10) { toast('最多添加 10 个子 Agent'); return; }
  editingAgentSubs.push(sid);
  renderConfigSubAgents();
}

function renderConfigTools(selected) {
  const list = document.getElementById('acfg-tool-list');
  list.innerHTML = TOOL_NAMES.map(t => {
    const checked = selected.includes(t);
    return `<label class="agent-check-item ${checked?'checked':''}" onclick="this.classList.toggle('checked')"><input type="checkbox" value="${t}" ${checked?'checked':''}> ${t}</label>`;
  }).join('');
}

function renderSuggestQuestions(questions) {
  const list = document.getElementById('acfg-suggest-list');
  list.innerHTML = (questions.length ? questions : ['']).map((q, i) =>
    `<div style="display:flex;gap:8px;margin-bottom:8px;align-items:center">
      <input class="agent-cfg-input acfg-suggest-input" value="${escHtml(q)}" placeholder="请输入推荐问" />
      <span style="cursor:pointer;color:var(--text3);font-size:14px" onclick="this.parentElement.remove()">✕</span>
    </div>`
  ).join('');
}

function addSuggestQuestion() {
  const list = document.getElementById('acfg-suggest-list');
  list.insertAdjacentHTML('beforeend', `<div style="display:flex;gap:8px;margin-bottom:8px;align-items:center">
    <input class="agent-cfg-input acfg-suggest-input" value="" placeholder="请输入推荐问" />
    <span style="cursor:pointer;color:var(--text3);font-size:14px" onclick="this.parentElement.remove()">✕</span>
  </div>`);
}

async function saveAgentConfig() {
  if (!activeAgentId) return;
  const a = agentList.find(x => x.id === activeAgentId);
  if (!a) return;
  const isMulti = a.agent_type === 'multi_agent';
  const body = {
    name: document.getElementById('acfg-input-name').value.trim() || a.name,
    description: document.getElementById('acfg-input-desc').value.trim(),
    agent_type: a.agent_type,
    model_name: document.getElementById('acfg-input-model').value,
    context_rounds: parseInt(document.getElementById('acfg-input-rounds').value) || 2,
    system_prompt: document.getElementById('acfg-input-prompt').value.trim(),
    greeting: document.getElementById('acfg-input-greeting').value.trim(),
    suggested_questions: [...document.querySelectorAll('.acfg-suggest-input')].map(i => i.value.trim()).filter(Boolean),
    sub_agents: isMulti ? editingAgentSubs : [],
    collaboration_mode: isMulti ? document.getElementById('acfg-input-collab').value : 'sequential',
    available_tools: !isMulti ? [...document.querySelectorAll('#acfg-tool-list .checked input')].map(i => i.value) : [],
    max_steps: !isMulti ? parseInt(document.getElementById('acfg-input-maxsteps').value) || 10 : 10,
  };
  try {
    await fetch(API + '/agent/' + activeAgentId, { method: 'PUT', headers: {'Content-Type':'application/json'}, body: JSON.stringify(body) });
    toast('已保存');
    await loadAgentList();
    // Update topbar name
    document.getElementById('acfg-name').textContent = body.name;
    document.getElementById('acfg-preview-name').textContent = body.name;
    document.getElementById('acfg-preview-greeting').textContent = body.greeting || '保存后可在此测试对话';
  } catch (e) { toast('保存失败', 'error'); }
}

async function deleteAgentFromConfig() {
  if (!activeAgentId || !confirm('确定删除该智能体？')) return;
  try {
    await fetch(API + '/agent/' + activeAgentId, { method: 'DELETE' });
    toast('已删除');
    closeAgentConfig();
    loadAgentList();
  } catch (e) { toast('删除失败', 'error'); }
}

// Preview chat in config page
async function sendAgentPreview() {
  if (!activeAgentId) return;
  const input = document.getElementById('acfg-preview-input');
  const msg = input.value.trim();
  if (!msg) return;
  input.value = '';
  const chat = document.getElementById('acfg-preview-chat');
  chat.innerHTML += `<div style="text-align:right;margin-bottom:10px"><span style="background:var(--accent);color:#fff;padding:6px 12px;border-radius:12px 12px 2px 12px;font-size:12px;display:inline-block;max-width:80%">${escHtml(msg)}</span></div>`;
  chat.innerHTML += `<div id="acfg-preview-loading" style="margin-bottom:10px"><span style="background:var(--surface2);padding:6px 12px;border-radius:12px 12px 12px 2px;font-size:12px;display:inline-block;color:var(--text3)">思考中...</span></div>`;
  chat.scrollTop = chat.scrollHeight;
  try {
    const res = await fetch(API + '/agent/' + activeAgentId + '/chat', {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({ message: msg, session_id: 'preview-' + activeAgentId, ...getCreds() })
    });
    const data = await res.json();
    const loading = document.getElementById('acfg-preview-loading');
    if (loading) loading.remove();
    chat.innerHTML += `<div style="margin-bottom:10px"><span style="background:var(--surface2);padding:6px 12px;border-radius:12px 12px 12px 2px;font-size:12px;display:inline-block;max-width:80%;white-space:pre-wrap">${escHtml(data.response || '无回复')}</span></div>`;
    chat.scrollTop = chat.scrollHeight;
  } catch (e) {
    const loading = document.getElementById('acfg-preview-loading');
    if (loading) loading.innerHTML = `<span style="background:var(--red-dim);color:var(--red);padding:6px 12px;border-radius:12px;font-size:12px;display:inline-block">请求失败</span>`;
  }
}

function updateAgentCount() {
  const c = agentList.length;
  const el = document.getElementById('agent-count');
  if (el) el.textContent = c;
  const st = document.getElementById('stat-agent');
  if (st) st.textContent = c;
}
