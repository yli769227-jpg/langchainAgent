// kb.js — 知识库

// ══════════════════════════════════════
// KNOWLEDGE BASE
// ══════════════════════════════════════
let kbList = [];
let activeKbId = null;

async function loadKbList() {
  try {
    const r = await fetch(API + '/knowledge-base');
    const d = await r.json();
    kbList = d.knowledge_bases || [];
    renderKbGrid();
    updateKbCount();
    loadChatKbSelect();
  } catch(e) {}
}

function updateKbCount() {
  document.getElementById('kb-count').textContent = kbList.length;
  document.getElementById('stat-kb').textContent = kbList.length;
}

function renderKbGrid() {
  const el = document.getElementById('kb-grid');
  if (!kbList.length) {
    el.innerHTML = '<div style="font-size:12px;color:var(--text3);padding:20px 0">暂无知识库，点击右上角新建</div>';
    return;
  }
  el.innerHTML = kbList.map(kb => `
    <div class="kb-card ${activeKbId === kb.kb_id ? 'selected' : ''}" onclick="selectKb('${kb.kb_id}')">
      <div class="kb-card-icon">📚</div>
      <div class="kb-card-name">${escHtml(kb.name)}</div>
      <div class="kb-card-desc">${escHtml(kb.description || '暂无描述')}</div>
      <div class="kb-card-footer">
        <div class="kb-card-count">
          <svg width="11" height="11" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/></svg>
          ${kb.doc_count} 篇文档
        </div>
        <div class="kb-card-actions">
          <button class="kb-action" onclick="event.stopPropagation();openAddDocModal('${kb.kb_id}')">+ 文档</button>
          <button class="kb-action danger" onclick="event.stopPropagation();deleteKb('${kb.kb_id}')">删除</button>
        </div>
      </div>
    </div>
  `).join('');
}

async function selectKb(kbId) {
  activeKbId = kbId;
  renderKbGrid();
  const kb = kbList.find(k => k.kb_id === kbId);
  if (!kb) return;
  document.getElementById('kb-detail').style.display = 'block';
  document.getElementById('kb-detail-name').textContent = kb.name;
  document.getElementById('kb-detail-sub').textContent = kb.doc_count + ' 篇文档';
  // Load docs
  try {
    const r = await fetch(API + '/knowledge-base');
    const d = await r.json();
    const found = (d.knowledge_bases || []).find(k => k.kb_id === kbId);
    // Fetch full docs via a separate endpoint if available, else show count
    renderDocList(found ? [] : []);
  } catch(e) {}
  // Try to get docs
  try {
    const r = await fetch(API + '/knowledge-base/' + kbId + '/documents');
    if (r.ok) {
      const d = await r.json();
      renderDocList(d.documents || []);
    }
  } catch(e) {}
}

function renderDocList(docs) {
  const el = document.getElementById('doc-list');
  if (!docs.length) {
    el.innerHTML = '<div style="font-size:12px;color:var(--text3);padding:16px;text-align:center">暂无文档，点击"添加文档"</div>';
    return;
  }
  el.innerHTML = docs.map((doc, i) => `
    <div class="doc-item">
      <div class="doc-item-header">
        <span class="doc-item-idx">#${i}</span>
        <span class="doc-item-del" onclick="deleteDoc('${activeKbId}',${i})">删除</span>
      </div>
      <div class="doc-item-content">${escHtml(doc.substring(0, 300))}${doc.length > 300 ? '...' : ''}</div>
    </div>
  `).join('');
}

function openCreateKbModal() { openModal('modal-create-kb'); }

async function createKb() {
  const name = document.getElementById('kb-name-input').value.trim();
  if (!name) return;
  const desc = document.getElementById('kb-desc-input').value.trim();
  await fetch(`${API}/knowledge-base?name=${encodeURIComponent(name)}&description=${encodeURIComponent(desc)}`, { method: 'POST' });
  closeModal('modal-create-kb');
  document.getElementById('kb-name-input').value = '';
  document.getElementById('kb-desc-input').value = '';
  await loadKbList();
  toast('知识库已创建', 'success');
}

let addDocTargetKbId = null;
function openAddDocModal(kbId) {
  addDocTargetKbId = kbId || activeKbId;
  document.getElementById('doc-content-input').value = '';
  openModal('modal-add-doc');
}

async function addDocument() {
  const content = document.getElementById('doc-content-input').value.trim();
  if (!content || !addDocTargetKbId) return;
  await fetch(`${API}/knowledge-base/${addDocTargetKbId}/document`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ content })
  });
  closeModal('modal-add-doc');
  await loadKbList();
  if (activeKbId === addDocTargetKbId) selectKb(activeKbId);
  toast('文档已添加', 'success');
}

async function deleteDoc(kbId, idx) {
  if (!confirm('确认删除该文档？')) return;
  await fetch(`${API}/knowledge-base/${kbId}/document/${idx}`, { method: 'DELETE' });
  await loadKbList();
  selectKb(kbId);
  toast('文档已删除');
}

async function deleteKb(kbId) {
  if (!confirm('确认删除该知识库？')) return;
  await fetch(`${API}/knowledge-base/${kbId}`, { method: 'DELETE' });
  if (activeKbId === kbId) {
    activeKbId = null;
    document.getElementById('kb-detail').style.display = 'none';
  }
  await loadKbList();
  toast('知识库已删除');
}

function loadChatKbSelect() {
  const sel = document.getElementById('chat-kb-select');
  sel.innerHTML = '<option value="">不使用知识库</option>' +
    kbList.map(kb => `<option value="${kb.kb_id}">${escHtml(kb.name)}</option>`).join('');
  // 同时加载 agent 列表
  fetch(API + '/agent').then(r => r.json()).then(data => {
    const aSel = document.getElementById('chat-agent-select');
    aSel.innerHTML = '<option value="">不使用智能体</option>' +
      (data.agents || []).map(a => `<option value="${a.id}">${escHtml(a.name)}</option>`).join('');
  }).catch(() => {});
}
