// workflow.js — 工作流画布

// ══════════════════════════════════════
// WORKFLOW CANVAS
// ══════════════════════════════════════
let wfList = [];
let currentWfId = null;
let canvasNodes = [];   // {id, type, label, x, y, config}
let canvasEdges = [];   // {id, source, target}
let selectedNodeId = null;
let draggingNodeId = null;
let dragOffsetX = 0, dragOffsetY = 0;
let connectingFrom = null; // port node id

// ── Zoom & Pan state ──
let canvasZoom = 1;
let canvasPanX = 0, canvasPanY = 0;
let isPanning = false;
let panStartX = 0, panStartY = 0;
let panStartPanX = 0, panStartPanY = 0;
let spacePressed = false;

// ── Connection preview state ──
let connectPreviewMouse = null; // {x, y} in canvas-layer coords
let connectHintEl = null;

const NODE_LABELS = { start:'开始', end:'结束', llm:'LLM 模型', tool:'工具调用', knowledge:'知识库检索', condition:'条件分支' };
const NODE_DESCS  = { start:'工作流入口', end:'工作流出口', llm:'调用大语言模型', tool:'调用内置工具', knowledge:'RAG 检索', condition:'条件判断' };

async function loadWfList() {
  try {
    const r = await fetch(API + '/workflow');
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    wfList = d.workflows || [];
    renderWfList();
    updateWfCount();
  } catch(e) {
    console.warn('加载工作流列表失败:', e.message);
  }
}

function updateWfCount() {
  document.getElementById('wf-count').textContent = wfList.length;
  document.getElementById('stat-wf').textContent = wfList.length;
}

function renderWfList() {
  const el = document.getElementById('wf-list');
  if (!wfList.length) {
    el.innerHTML = '<div style="font-size:11px;color:var(--text3);text-align:center;padding:24px 0">暂无工作流</div>';
    return;
  }
  el.innerHTML = wfList.map(wf => `
    <div class="wf-item ${currentWfId === wf.id ? 'active' : ''}" onclick="loadWorkflow('${wf.id}')">
      <div class="wf-item-name">${escHtml(wf.name)}</div>
      <div class="wf-item-meta">
        <span>${(wf.nodes||[]).length} 节点</span>
        <span>${(wf.edges||[]).length} 连线</span>
      </div>
      <button class="wf-item-del" onclick="event.stopPropagation();deleteWorkflow('${wf.id}')" title="删除">×</button>
    </div>
  `).join('');
}

async function loadWorkflow(wfId) {
  try {
    const r = await fetch(API + '/workflow/' + wfId);
    const wf = await r.json();
    currentWfId = wfId;
    canvasNodes = wf.nodes || [];
    canvasEdges = wf.edges || [];
    selectedNodeId = null;
    connectingFrom = null;
    document.getElementById('wf-current-name').textContent = wf.name;
    document.getElementById('wf-save-btn').disabled = false;
    document.getElementById('wf-run-btn').disabled = false;
    document.getElementById('wf-canvas-empty').style.display = 'none';
    renderCanvas();
    renderWfList();
    renderPropsEmpty();
    renderRunInputs();
    // Auto-fit if there are nodes
    if (canvasNodes.length > 0) {
      setTimeout(zoomFit, 50);
    } else {
      zoomReset();
    }
  } catch(e) { toast('加载失败', 'error'); }
}

async function deleteWorkflow(wfId) {
  if (!confirm('确认删除该工作流？')) return;
  await fetch(API + '/workflow/' + wfId, { method: 'DELETE' });
  if (currentWfId === wfId) {
    currentWfId = null;
    canvasNodes = [];
    canvasEdges = [];
    selectedNodeId = null;
    document.getElementById('wf-current-name').textContent = '未选择工作流';
    document.getElementById('wf-save-btn').disabled = true;
    document.getElementById('wf-run-btn').disabled = true;
    document.getElementById('wf-canvas-empty').style.display = 'flex';
    renderCanvas();
    renderPropsEmpty();
    renderRunInputs();
  }
  await loadWfList();
  toast('工作流已删除');
}

function openCreateWfModal() { openModal('modal-create-wf'); }

async function createWorkflow() {
  const name = document.getElementById('wf-name-input').value.trim();
  if (!name) { toast('请输入工作流名称', 'error'); return; }
  const desc = document.getElementById('wf-desc-input').value.trim();
  try {
    const r = await fetch(API + '/workflow', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ name, description: desc, nodes: [], edges: [] })
    });
    if (!r.ok) throw new Error('HTTP ' + r.status);
    const d = await r.json();
    closeModal('modal-create-wf');
    document.getElementById('wf-name-input').value = '';
    document.getElementById('wf-desc-input').value = '';
    await loadWfList();
    loadWorkflow(d.workflow_id);
    toast('工作流已创建', 'success');
  } catch(e) {
    toast('创建失败，请确认后端服务已启动 (' + e.message + ')', 'error');
  }
}

async function saveWorkflow() {
  if (!currentWfId) return;
  const wf = wfList.find(w => w.id === currentWfId);
  await fetch(API + '/workflow/' + currentWfId, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      name: wf ? wf.name : '未命名',
      description: wf ? wf.description : '',
      nodes: canvasNodes,
      edges: canvasEdges
    })
  });
  await loadWfList();
  toast('已保存', 'success');
}

// ══════════════════════════════════════
// ZOOM & PAN
// ══════════════════════════════════════
function updateCanvasTransform() {
  const layer = document.getElementById('wf-canvas-layer');
  if (layer) {
    layer.style.transform = `translate(${canvasPanX}px, ${canvasPanY}px) scale(${canvasZoom})`;
  }
  // Update grid pattern to match zoom
  const grid = document.getElementById('wf-canvas-grid');
  if (grid) {
    const pat = grid.querySelector('pattern');
    if (pat) {
      const size = 28 * canvasZoom;
      pat.setAttribute('width', size);
      pat.setAttribute('height', size);
      pat.setAttribute('patternTransform', `translate(${canvasPanX % size}, ${canvasPanY % size})`);
      const circle = pat.querySelector('circle');
      if (circle) {
        circle.setAttribute('cx', size / 2);
        circle.setAttribute('cy', size / 2);
        circle.setAttribute('r', Math.max(0.5, 0.8 * canvasZoom));
      }
    }
  }
  document.getElementById('wf-zoom-label').textContent = Math.round(canvasZoom * 100) + '%';
}

function zoomCanvas(delta) {
  const canvas = document.getElementById('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  const cx = rect.width / 2;
  const cy = rect.height / 2;
  zoomAtPoint(delta, cx, cy);
}

function zoomAtPoint(delta, px, py) {
  const oldZoom = canvasZoom;
  canvasZoom = Math.min(3, Math.max(0.15, canvasZoom + delta));
  // Adjust pan so zoom centers on the point
  const scale = canvasZoom / oldZoom;
  canvasPanX = px - scale * (px - canvasPanX);
  canvasPanY = py - scale * (py - canvasPanY);
  updateCanvasTransform();
}

function zoomReset() {
  canvasZoom = 1;
  canvasPanX = 0;
  canvasPanY = 0;
  updateCanvasTransform();
}

function zoomFit() {
  if (!canvasNodes.length) { zoomReset(); return; }
  const canvas = document.getElementById('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  const padding = 60;
  // Find bounding box of all nodes
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  canvasNodes.forEach(node => {
    const el = document.getElementById('node-' + node.id);
    const w = el ? el.offsetWidth : 140;
    const h = el ? el.offsetHeight : 80;
    minX = Math.min(minX, node.x);
    minY = Math.min(minY, node.y);
    maxX = Math.max(maxX, node.x + w);
    maxY = Math.max(maxY, node.y + h);
  });
  const bw = maxX - minX;
  const bh = maxY - minY;
  if (bw <= 0 || bh <= 0) { zoomReset(); return; }
  const zx = (rect.width - padding * 2) / bw;
  const zy = (rect.height - padding * 2) / bh;
  canvasZoom = Math.min(2, Math.max(0.15, Math.min(zx, zy)));
  canvasPanX = (rect.width - bw * canvasZoom) / 2 - minX * canvasZoom;
  canvasPanY = (rect.height - bh * canvasZoom) / 2 - minY * canvasZoom;
  updateCanvasTransform();
}

// Canvas mouse events for pan
(function initCanvasInteractions() {
  const canvas = document.getElementById('wf-canvas');

  // Wheel zoom
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const rect = canvas.getBoundingClientRect();
    const px = e.clientX - rect.left;
    const py = e.clientY - rect.top;
    const delta = e.deltaY > 0 ? -0.08 : 0.08;
    zoomAtPoint(delta, px, py);
  }, { passive: false });

  // Middle-click or space+click or left-click on empty canvas = pan
  canvas.addEventListener('mousedown', (e) => {
    // Middle button or space pressed
    if (e.button === 1 || (e.button === 0 && spacePressed)) {
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      panStartPanX = canvasPanX;
      panStartPanY = canvasPanY;
      canvas.classList.add('panning');
      e.preventDefault();
      return;
    }
    // Left click on canvas background (not on node/port)
    if (e.button === 0 && !e.target.closest('.wf-node') && !e.target.closest('.wf-port')) {
      if (connectingFrom) {
        cancelConnection();
        return;
      }
      // Start panning by dragging empty canvas area
      isPanning = true;
      panStartX = e.clientX;
      panStartY = e.clientY;
      panStartPanX = canvasPanX;
      panStartPanY = canvasPanY;
      canvas.classList.add('panning');
      e.preventDefault();
    }
  });

  document.addEventListener('mousemove', (e) => {
    if (isPanning) {
      canvasPanX = panStartPanX + (e.clientX - panStartX);
      canvasPanY = panStartPanY + (e.clientY - panStartY);
      updateCanvasTransform();
      return;
    }
    // Connection preview line
    if (connectingFrom) {
      const rect = canvas.getBoundingClientRect();
      connectPreviewMouse = {
        x: (e.clientX - rect.left - canvasPanX) / canvasZoom,
        y: (e.clientY - rect.top - canvasPanY) / canvasZoom
      };
      renderEdges();
      // Move hint
      if (connectHintEl) {
        connectHintEl.style.left = (e.clientX + 16) + 'px';
        connectHintEl.style.top = (e.clientY - 12) + 'px';
      }
    }
  });

  document.addEventListener('mouseup', (e) => {
    if (isPanning) {
      // If barely moved, treat as a click to deselect
      const moved = Math.abs(e.clientX - panStartX) + Math.abs(e.clientY - panStartY);
      if (moved < 4 && !spacePressed && e.button === 0) {
        selectedNodeId = null;
        document.querySelectorAll('.wf-node').forEach(el => el.classList.remove('selected'));
        renderPropsEmpty();
      }
      isPanning = false;
      canvas.classList.remove('panning');
    }
    draggingNodeId = null;
  });

  // Space key for pan mode
  document.addEventListener('keydown', (e) => {
    if (e.code === 'Space' && e.target === document.body) {
      e.preventDefault();
      spacePressed = true;
      canvas.classList.add('panning');
    }
    // Delete key to remove selected node or edge
    if ((e.key === 'Delete' || e.key === 'Backspace') && e.target === document.body) {
      if (selectedNodeId) deleteNode(selectedNodeId);
    }
    // Escape to cancel connection
    if (e.key === 'Escape') {
      if (connectingFrom) cancelConnection();
    }
  });

  document.addEventListener('keyup', (e) => {
    if (e.code === 'Space') {
      spacePressed = false;
      if (!isPanning) canvas.classList.remove('panning');
    }
  });
})();

function cancelConnection() {
  connectingFrom = null;
  connectPreviewMouse = null;
  document.querySelectorAll('.wf-port.active').forEach(p => p.classList.remove('active'));
  if (connectHintEl) { connectHintEl.remove(); connectHintEl = null; }
  document.getElementById('wf-canvas').classList.remove('connecting');
  renderEdges();
}

// Helper: convert page coords to canvas-layer coords
function pageToCanvas(pageX, pageY) {
  const canvas = document.getElementById('wf-canvas');
  const rect = canvas.getBoundingClientRect();
  return {
    x: (pageX - rect.left - canvasPanX) / canvasZoom,
    y: (pageY - rect.top - canvasPanY) / canvasZoom
  };
}

// ── Drag from palette ──
let dragType = null;
function onPaletteDrag(e, type) {
  dragType = type;
  e.dataTransfer.effectAllowed = 'copy';
}

async function onCanvasDrop(e) {
  e.preventDefault();
  if (!dragType) return;
  if (!currentWfId) {
    // 自动创建默认工作流，无需用户手动新建
    try {
      const r = await fetch(API + '/workflow', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: '新工作流', description: '', nodes: [], edges: [] })
      });
      if (!r.ok) throw new Error('HTTP ' + r.status);
      const d = await r.json();
      currentWfId = d.workflow_id;
      document.getElementById('wf-current-name').textContent = '新工作流';
      document.getElementById('wf-save-btn').disabled = false;
      document.getElementById('wf-run-btn').disabled = false;
      await loadWfList();
    } catch(err) {
      toast('请先新建或选择一个工作流', 'error');
      dragType = null;
      return;
    }
  }
  const pos = pageToCanvas(e.clientX, e.clientY);
  const x = pos.x - 70;
  const y = pos.y - 30;
  const id = 'node_' + Date.now();
  canvasNodes.push({
    id, type: dragType,
    label: NODE_LABELS[dragType] || dragType,
    x, y, config: {}
  });
  dragType = null;
  document.getElementById('wf-canvas-empty').style.display = 'none';
  renderCanvas();
}

// ── Canvas render ──
function renderCanvas() {
  const layer = document.getElementById('wf-canvas-layer');
  // Remove old nodes
  layer.querySelectorAll('.wf-node').forEach(n => n.remove());
  // Render nodes
  canvasNodes.forEach(node => {
    const el = createNodeEl(node);
    layer.appendChild(el);
  });
  renderEdges();
  updateCanvasTransform();
}

function buildEndExpandedBody(node) {
  const cfg = node.config || {};
  const nid = node.id;
  const replyMode = cfg.reply_mode || 'template'; // 'direct' | 'template'
  const outputs = cfg.end_outputs || [{ name: 'output', type: '引用', ref: '' }];
  const template = cfg.reply_template || '';
  const streamOn = cfg.stream !== false;

  const outputRowsHtml = outputs.map((o, i) => `
    <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;align-items:center;margin-bottom:5px">
      <input class="node-end-field" placeholder="参数名" value="${escHtml(o.name||'')}"
        oninput="updateEndOutput('${nid}',${i},'name',this.value)" />
      <select class="node-end-sel" onchange="updateEndOutput('${nid}',${i},'type',this.value)">
        <option value="引用" ${(o.type||'引用')==='引用'?'selected':''}>引用</option>
        <option value="String" ${o.type==='String'?'selected':''}>String</option>
        <option value="Number" ${o.type==='Number'?'selected':''}>Number</option>
      </select>
      <select class="node-end-sel" onchange="updateEndOutput('${nid}',${i},'ref',this.value)">
        ${refParamOptions(o.ref)}
      </select>
      <span style="cursor:pointer;color:var(--text3);font-size:13px;text-align:center" onclick="removeEndOutput('${nid}',${i})">×</span>
    </div>`).join('');

  const varChips = outputs.filter(o => o.name).map(o => `
    <span class="node-end-var-chip" onclick="insertEndVar('${nid}','${escHtml(o.name)}')">
      <span class="node-end-var-chip-plus">+</span> {{${escHtml(o.name)}}}
    </span>`).join('');

  return `
    <div class="node-end-expanded-body" onmousedown="event.stopPropagation()">
      <div class="node-end-desc">工作流的最终节点，输出工作流运行后的最终结果。<a>了解更多 ›</a></div>

      <div class="node-end-section-title">回复模式</div>
      <div class="node-end-radio-group">
        <div class="node-end-radio-opt ${replyMode==='direct'?'active':''}" onclick="setEndReplyMode('${nid}','direct')">
          <div class="node-end-radio-dot"></div>
          <span>直接返回参数值</span>
        </div>
        <div class="node-end-radio-opt ${replyMode==='template'?'active':''}" onclick="setEndReplyMode('${nid}','template')">
          <div class="node-end-radio-dot"></div>
          <span>按模版配置格式返回文本</span>
        </div>
      </div>

      <div class="node-end-divider"></div>

      <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px">
        <div class="node-end-section-title" style="margin-bottom:0">输出 <span style="width:13px;height:13px;border-radius:50%;border:1px solid var(--border2);color:var(--text3);font-size:9px;display:inline-flex;align-items:center;justify-content:center">?</span></div>
        <span style="cursor:pointer;color:var(--accent);font-size:18px;line-height:1;font-weight:300" onclick="addEndOutput('${nid}')" title="添加输出">+</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;margin-bottom:5px">
        <div class="node-end-col-hdr">参数名</div>
        <div class="node-end-col-hdr">类型</div>
        <div class="node-end-col-hdr">值</div>
        <div></div>
      </div>
      <div id="node-end-outputs-${nid}">${outputRowsHtml}</div>

      <div class="node-end-divider"></div>

      <div class="node-end-template-wrap">
        <div class="node-end-template-header">
          <div class="node-end-template-label">
            <span class="req">*</span> 回复模板
          </div>
          <div class="node-end-stream-toggle">
            流式输出
            <span style="width:13px;height:13px;border-radius:50%;border:1px solid var(--border2);color:var(--text3);font-size:9px;display:inline-flex;align-items:center;justify-content:center">?</span>
            <div class="node-end-toggle-switch ${streamOn?'on':''}" onclick="toggleEndStream('${nid}',this)"></div>
          </div>
        </div>
        <textarea class="node-end-textarea" id="node-end-tpl-${nid}"
          placeholder="可以根据参数名，在此定义返回结果的格式。例如: 今天 {{output_location}} 的温度为 {{output_temperature}}"
          oninput="updateNodeCfg('${nid}','reply_template',this.value);updateEndCharCount('${nid}')">${escHtml(template)}</textarea>
        <div class="node-end-char-count" id="node-end-cc-${nid}">${template.length}字</div>
        <div class="node-end-var-chips" id="node-end-chips-${nid}">
          ${varChips || '<span style="font-size:11px;color:var(--text3)">添加输出参数后可在此插入变量</span>'}
        </div>
      </div>
    </div>`;
}

function buildLlmExpandedBody(node) {
  const cfg = node.config || {};
  const nid = node.id;
  const modelVal = cfg.model || 'claude-sonnet-4-6';
  const modelName = modelVal === 'claude-opus-4-6' ? 'Claude Opus 4.6' : modelVal === 'deepseek-v3' ? 'DeepSeek-V3.2' : modelVal === 'claude-haiku-4-5-20251001' ? 'Claude Haiku 4.5' : 'Claude Sonnet 4.6';
  const inputs = cfg.inputs || [];
  const sysPrompt = cfg.system_prompt || '';
  const userPrompt = cfg.user_prompt || '';
  const sysOpen = cfg._sysOpen !== false;
  const outputName = cfg.output_name || 'output';
  const outputType = cfg.output_type || 'String';
  const outputDesc = cfg.output_desc || '';

  const inputRowsHtml = inputs.map((inp, i) => `
    <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;align-items:center;margin-bottom:5px">
      <input class="node-llm-field" placeholder="参数名" value="${escHtml(inp.name||'')}"
        oninput="updateLlmInput('${nid}',${i},'name',this.value)" />
      <select class="node-llm-sel" onchange="updateLlmInput('${nid}',${i},'type',this.value)">
        <option value="引用" ${(inp.type||'引用')==='引用'?'selected':''}>引用</option>
        <option value="String" ${inp.type==='String'?'selected':''}>String</option>
        <option value="Number" ${inp.type==='Number'?'selected':''}>Number</option>
      </select>
      <select class="node-llm-sel" onchange="updateLlmInput('${nid}',${i},'ref',this.value)">
        ${refParamOptions(inp.ref)}
      </select>
      <span style="cursor:pointer;color:var(--text3);font-size:13px;text-align:center" onclick="removeLlmInput('${nid}',${i})">×</span>
    </div>`).join('');

  return `
    <div class="node-llm-expanded-body" onmousedown="event.stopPropagation()">
      <div class="llm-desc">调用大语言模型，根据输入参数和提示词生成回复。<a>了解更多 ›</a></div>

      <button class="node-llm-tpl-btn">
        <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        大模型配置模板
      </button>

      <div class="node-llm-model-row">
        <div class="node-llm-model-label">
          <span class="req">*</span> 模型
          <span class="help">?</span>
        </div>
        <button class="node-llm-adv-btn">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          高级
        </button>
      </div>
      <div class="node-llm-model-wrap">
        <div class="node-llm-model-icon">✦</div>
        <select class="node-llm-model-sel" onchange="updateNodeCfg('${nid}','model',this.value);refreshNodeCard('${nid}')">
          <option value="claude-sonnet-4-6" ${modelVal==='claude-sonnet-4-6'?'selected':''}>Claude Sonnet 4.6</option>
          <option value="claude-opus-4-6" ${modelVal==='claude-opus-4-6'?'selected':''}>Claude Opus 4.6</option>
          <option value="claude-haiku-4-5-20251001" ${modelVal==='claude-haiku-4-5-20251001'?'selected':''}>Claude Haiku 4.5</option>
          <option value="deepseek-v3" ${modelVal==='deepseek-v3'?'selected':''}>DeepSeek-V3.2</option>
        </select>
        <div class="node-llm-model-tune">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/></svg>
        </div>
      </div>

      <div class="node-llm-divider"></div>

      <div class="node-llm-section-title">输入 <span class="help">?</span></div>
      <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;margin-bottom:5px">
        <div class="node-llm-col-hdr">参数名</div>
        <div class="node-llm-col-hdr">类型</div>
        <div class="node-llm-col-hdr">值</div>
        <div></div>
      </div>
      <div id="node-llm-inputs-${nid}">${inputRowsHtml}</div>
      <button style="width:100%;padding:6px;border:1px dashed var(--border2);color:var(--text2);background:transparent;border-radius:6px;font-size:11px;cursor:pointer;margin-top:3px"
        onclick="addLlmInput('${nid}')">+ 添加参数</button>

      <div class="node-llm-divider"></div>

      <div class="node-llm-section-title">提示词</div>

      <div>
        <div class="node-llm-prompt-row" onclick="toggleLlmSection('${nid}','sys')">
          <svg class="node-llm-prompt-arrow ${sysOpen?'open':''}" id="node-llm-sys-arrow-${nid}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
          <div class="node-llm-prompt-label">系统提示词 <span class="help">?</span></div>
          <div class="node-llm-prompt-acts">
            <button class="node-llm-act-btn" onclick="event.stopPropagation()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>优化
            </button>
            <button class="node-llm-act-btn" onclick="event.stopPropagation()">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>模板
            </button>
          </div>
        </div>
        <div id="node-llm-sys-body-${nid}" style="display:${sysOpen?'block':'none'}">
          <textarea class="node-llm-textarea" placeholder="设置大模型的角色、行为规范和背景知识..."
            oninput="updateNodeCfg('${nid}','system_prompt',this.value)">${escHtml(sysPrompt)}</textarea>
        </div>
      </div>

      <div style="margin-top:6px">
        <div class="node-llm-prompt-row" style="cursor:default">
          <svg style="width:12px;height:12px;color:var(--text3);transform:rotate(90deg);flex-shrink:0" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
          <div class="node-llm-prompt-label">用户提示词 <span class="req">*</span> <span class="help">?</span></div>
          <div class="node-llm-prompt-acts">
            <button class="node-llm-act-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>优化
            </button>
            <button class="node-llm-act-btn">
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>模板
            </button>
          </div>
        </div>
        <textarea class="node-llm-textarea" style="min-height:110px"
          placeholder="输入大模型的用户提示词，使大模型实现对应功能，通过插入{{参数名}}可以引用对应的参数值"
          oninput="updateNodeCfg('${nid}','user_prompt',this.value)">${escHtml(userPrompt)}</textarea>
      </div>

      <div class="node-llm-divider"></div>

      <div class="node-llm-section-title">输出 <span class="help">?</span></div>
      <div style="display:grid;grid-template-columns:1fr 70px 1fr;gap:5px;margin-bottom:5px">
        <div class="node-llm-col-hdr">参数名</div>
        <div class="node-llm-col-hdr">类型</div>
        <div class="node-llm-col-hdr">描述</div>
      </div>
      <div style="display:grid;grid-template-columns:1fr 70px 1fr;gap:5px">
        <input class="node-llm-field" value="${escHtml(outputName)}" oninput="updateNodeCfg('${nid}','output_name',this.value)" />
        <select class="node-llm-sel" onchange="updateNodeCfg('${nid}','output_type',this.value)">
          <option value="String" ${outputType==='String'?'selected':''}>String</option>
          <option value="Number" ${outputType==='Number'?'selected':''}>Number</option>
          <option value="Object" ${outputType==='Object'?'selected':''}>Object</option>
        </select>
        <input class="node-llm-field" placeholder="请输入描述" value="${escHtml(outputDesc)}" oninput="updateNodeCfg('${nid}','output_desc',this.value)" />
      </div>
    </div>`;
}

function createNodeEl(node) {
  const el = document.createElement('div');
  const isLlmExpanded = node.type === 'llm' && node._expanded !== false;
  const isEndExpanded = node.type === 'end' && node._expanded !== false;
  const isExpanded = isLlmExpanded || isEndExpanded;
  el.className = `wf-node node-${node.type}${isExpanded ? ' expanded' : ''} ${selectedNodeId === node.id ? 'selected' : ''}`;
  el.id = 'node-' + node.id;
  el.style.left = node.x + 'px';
  el.style.top = node.y + 'px';
  const cfg = node.config || {};

  const NODE_ICONS = { start: '▶', end: '⏹', llm: '✦', tool: '⚙', knowledge: '📚', condition: '◆' };
  const typeLabel = { llm:'大模型', start:'开始', end:'结束', tool:'工具', knowledge:'知识库', condition:'条件' }[node.type] || node.type;
  const icon = NODE_ICONS[node.type] || '●';

  let bodyHtml = '';
  if (node.type === 'llm') {
    if (isLlmExpanded) {
      bodyHtml = buildLlmExpandedBody(node);
    } else {
      const modelName = cfg.model === 'claude-opus-4-6' ? 'Claude Opus 4.6' : cfg.model === 'deepseek-v3' ? 'DeepSeek-V3.2' : cfg.model || 'Claude Sonnet 4.6';
      const inputCount = (cfg.inputs || []).length;
      const hasPrompt = !!(cfg.user_prompt || cfg.system_prompt);
      bodyHtml = `
        <div class="wf-node-body">
          <div style="display:flex;align-items:center;gap:6px;padding:6px 8px;background:var(--surface2);border:1px solid var(--border2);border-radius:7px;margin-bottom:8px">
            <div style="width:16px;height:16px;border-radius:3px;background:linear-gradient(135deg,#4f8ef7,#1a6bff);display:flex;align-items:center;justify-content:center;flex-shrink:0;font-size:8px;color:#fff">✦</div>
            <span style="font-size:11px;color:var(--text);font-family:var(--mono);flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${escHtml(modelName)}</span>
          </div>
          <div style="display:flex;gap:5px">
            <div style="flex:1;padding:4px 6px;background:var(--surface2);border:1px solid var(--border2);border-radius:5px;font-size:9px;color:var(--text3);text-align:center">
              <span style="color:var(--text2)">${inputCount}</span> 输入
            </div>
            <div style="flex:1;padding:4px 6px;background:var(--surface2);border:1px solid var(--border2);border-radius:5px;font-size:9px;color:var(--text3);text-align:center">
              <span style="color:${hasPrompt ? 'var(--green)' : 'var(--text2)'}">●</span> 提示词
            </div>
            <div style="flex:1;padding:4px 6px;background:var(--surface2);border:1px solid var(--border2);border-radius:5px;font-size:9px;color:var(--text3);text-align:center">
              <span style="color:var(--text2)">output</span>
            </div>
          </div>
        </div>`;
    }
  } else if (node.type === 'end') {
    if (isEndExpanded) {
      bodyHtml = buildEndExpandedBody(node);
    } else {
      bodyHtml = `<div class="wf-node-body"><div class="wf-node-desc">${escHtml(getNodeDesc(node))}</div></div>`;
    }
  } else {
    bodyHtml = `<div class="wf-node-body"><div class="wf-node-desc">${escHtml(getNodeDesc(node))}</div></div>`;
  }

  const expandBtn = (node.type === 'llm' || node.type === 'end') ? `
    <button class="node-llm-expand-toggle" title="${isExpanded ? '收起' : '展开'}" onclick="event.stopPropagation();toggleNodeExpand('${node.id}')">
      ${isExpanded ? '⊟' : '⊞'}
    </button>` : '';

  el.innerHTML = `
    <div class="wf-node-header">
      <div class="wf-node-icon">${icon}</div>
      <div class="wf-node-title-wrap">
        <div class="wf-node-type">${typeLabel}</div>
        <div class="wf-node-label">${escHtml(node.label)}</div>
      </div>
      ${expandBtn}
    </div>
    ${bodyHtml}
    <div class="wf-node-ports">
      <div class="wf-port in ${connectingFrom ? 'target-hint' : ''}" data-node="${node.id}" data-port="in" title="输入端口"></div>
      <div class="wf-port out" data-node="${node.id}" data-port="out" title="输出端口（点击开始连线）"></div>
    </div>
  `;

  if (isExpanded) {
    el.querySelector('.wf-node-header').addEventListener('mousedown', onNodeMouseDown);
    el.querySelector('.wf-node-header').style.cursor = 'move';
  } else {
    el.addEventListener('mousedown', onNodeMouseDown);
  }
  el.addEventListener('click', (e) => {
    e.stopPropagation();
    if (!spacePressed && !isPanning) selectNode(node.id);
  });
  el.querySelectorAll('.wf-port').forEach(port => {
    port.addEventListener('mousedown', (e) => { e.stopPropagation(); });
    port.addEventListener('click', (e) => { e.stopPropagation(); onPortClick(e); });
  });
  return el;
}

function toggleLlmNodeExpand(nodeId) { toggleNodeExpand(nodeId); }

function toggleNodeExpand(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node._expanded = node._expanded === false ? true : false;
  const el = document.getElementById('node-' + nodeId);
  if (!el) return;
  const newEl = createNodeEl(node);
  el.parentNode.replaceChild(newEl, el);
  renderEdges();
}

function setEndReplyMode(nodeId, mode) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.reply_mode = mode;
  const el = document.getElementById('node-' + nodeId);
  if (!el) return;
  el.querySelectorAll('.node-end-radio-opt').forEach(opt => opt.classList.remove('active'));
  const opts = el.querySelectorAll('.node-end-radio-opt');
  if (mode === 'direct' && opts[0]) opts[0].classList.add('active');
  if (mode === 'template' && opts[1]) opts[1].classList.add('active');
}

function addEndOutput(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.end_outputs = node.config.end_outputs || [];
  node.config.end_outputs.push({ name: '', type: '引用', ref: '' });
  refreshEndOutputs(nodeId);
}

function removeEndOutput(nodeId, idx) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  (node.config.end_outputs || []).splice(idx, 1);
  refreshEndOutputs(nodeId);
}

function updateEndOutput(nodeId, idx, field, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.end_outputs = node.config.end_outputs || [];
  if (node.config.end_outputs[idx]) node.config.end_outputs[idx][field] = val;
  if (field === 'name') refreshEndVarChips(nodeId);
}

function refreshEndOutputs(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  const container = document.getElementById(`node-end-outputs-${nodeId}`);
  if (!container) return;
  const outputs = (node.config || {}).end_outputs || [];
  container.innerHTML = outputs.map((o, i) => `
    <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;align-items:center;margin-bottom:5px">
      <input class="node-end-field" placeholder="参数名" value="${escHtml(o.name||'')}"
        oninput="updateEndOutput('${nodeId}',${i},'name',this.value)" />
      <select class="node-end-sel" onchange="updateEndOutput('${nodeId}',${i},'type',this.value)">
        <option value="引用" ${(o.type||'引用')==='引用'?'selected':''}>引用</option>
        <option value="String" ${o.type==='String'?'selected':''}>String</option>
        <option value="Number" ${o.type==='Number'?'selected':''}>Number</option>
      </select>
      <select class="node-end-sel" onchange="updateEndOutput('${nodeId}',${i},'ref',this.value)">
        ${refParamOptions(o.ref)}
      </select>
      <span style="cursor:pointer;color:var(--text3);font-size:13px;text-align:center" onclick="removeEndOutput('${nodeId}',${i})">×</span>
    </div>`).join('');
  refreshEndVarChips(nodeId);
}

function refreshEndVarChips(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  const chips = document.getElementById(`node-end-chips-${nodeId}`);
  if (!chips) return;
  const outputs = (node.config || {}).end_outputs || [];
  const named = outputs.filter(o => o.name);
  chips.innerHTML = named.length
    ? named.map(o => `<span class="node-end-var-chip" onclick="insertEndVar('${nodeId}','${escHtml(o.name)}')"><span class="node-end-var-chip-plus">+</span> {{${escHtml(o.name)}}}</span>`).join('')
    : '<span style="font-size:11px;color:var(--text3)">添加输出参数后可在此插入变量</span>';
}

function insertEndVar(nodeId, varName) {
  const ta = document.getElementById(`node-end-tpl-${nodeId}`);
  if (!ta) return;
  const pos = ta.selectionStart;
  const val = ta.value;
  const insert = `{{${varName}}}`;
  ta.value = val.slice(0, pos) + insert + val.slice(pos);
  ta.selectionStart = ta.selectionEnd = pos + insert.length;
  ta.focus();
  updateNodeCfg(nodeId, 'reply_template', ta.value);
  updateEndCharCount(nodeId);
}

function updateEndCharCount(nodeId) {
  const ta = document.getElementById(`node-end-tpl-${nodeId}`);
  const cc = document.getElementById(`node-end-cc-${nodeId}`);
  if (ta && cc) cc.textContent = ta.value.length + '字';
}

function toggleEndStream(nodeId, el) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.stream = !node.config.stream;
  el.classList.toggle('on', node.config.stream !== false);
}

function getNodeDesc(node) {
  const cfg = node.config || {};
  if (node.type === 'tool') return cfg.tool_name || NODE_DESCS[node.type];
  if (node.type === 'knowledge') return cfg.kb_id ? '已关联知识库' : NODE_DESCS[node.type];
  if (node.type === 'llm') return cfg.model || 'claude-sonnet-4-6';
  return NODE_DESCS[node.type] || '';
}

// ── Node drag ──
function onNodeMouseDown(e) {
  if (e.target.classList.contains('wf-port')) return;
  e.stopPropagation(); // Prevent canvas mousedown from cancelling active connection
  if (spacePressed) return; // pan mode
  const nodeEl = e.currentTarget.closest('.wf-node') || e.currentTarget;
  const nodeId = nodeEl.id.replace('node-', '');
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  draggingNodeId = nodeId;
  dragOffsetX = (e.clientX - document.getElementById('wf-canvas').getBoundingClientRect().left - canvasPanX) / canvasZoom - node.x;
  dragOffsetY = (e.clientY - document.getElementById('wf-canvas').getBoundingClientRect().top - canvasPanY) / canvasZoom - node.y;
  e.preventDefault();
}

document.addEventListener('mousemove', (e) => {
  if (!draggingNodeId || isPanning) return;
  const node = canvasNodes.find(n => n.id === draggingNodeId);
  if (!node) return;
  const pos = pageToCanvas(e.clientX, e.clientY);
  node.x = pos.x - dragOffsetX;
  node.y = pos.y - dragOffsetY;
  const el = document.getElementById('node-' + draggingNodeId);
  if (el) { el.style.left = node.x + 'px'; el.style.top = node.y + 'px'; }
  renderEdges();
});

// ── Port connecting ──
function onPortClick(e) {
  const port = e.currentTarget;
  const nodeId = port.dataset.node;
  const portType = port.dataset.port;

  if (portType === 'out') {
    // Start connection from output port
    if (connectingFrom) cancelConnection();
    connectingFrom = nodeId;
    port.classList.add('active');
    document.getElementById('wf-canvas').classList.add('connecting');
    // Show hint
    connectHintEl = document.createElement('div');
    connectHintEl.className = 'wf-connect-hint';
    connectHintEl.textContent = '点击目标节点的输入端口 · Esc 取消';
    connectHintEl.style.left = (e.clientX + 16) + 'px';
    connectHintEl.style.top = (e.clientY - 12) + 'px';
    document.body.appendChild(connectHintEl);
  } else if (portType === 'in' && connectingFrom && connectingFrom !== nodeId) {
    // Check for duplicate edge
    const exists = canvasEdges.some(edge => edge.source === connectingFrom && edge.target === nodeId);
    if (exists) {
      toast('该连线已存在', 'error');
      cancelConnection();
      return;
    }
    // Create edge
    const edgeId = 'edge_' + Date.now();
    canvasEdges.push({ id: edgeId, source: connectingFrom, target: nodeId });
    cancelConnection();
    renderEdges();
    toast('连线成功', 'success');
  } else if (portType === 'in' && !connectingFrom) {
    // Clicking input port without connection = start reverse connection? No, just ignore.
  } else {
    cancelConnection();
  }
}

// ── Edges SVG ──
function renderEdges() {
  const svg = document.getElementById('wf-svg');
  // Remove old edges and previews
  svg.querySelectorAll('.wf-edge, .wf-edge-hit, .wf-edge-preview').forEach(e => e.remove());

  canvasEdges.forEach(edge => {
    const src = canvasNodes.find(n => n.id === edge.source);
    const tgt = canvasNodes.find(n => n.id === edge.target);
    if (!src || !tgt) return;
    const srcEl = document.getElementById('node-' + src.id);
    const tgtEl = document.getElementById('node-' + tgt.id);
    if (!srcEl || !tgtEl) return;
    const x1 = src.x + srcEl.offsetWidth;
    const y1 = src.y + srcEl.offsetHeight / 2;
    const x2 = tgt.x;
    const y2 = tgt.y + tgtEl.offsetHeight / 2;
    const dx = Math.max(Math.abs(x2 - x1) * 0.5, 60);
    const d = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;

    // Invisible hit area for clicking
    const hitPath = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    hitPath.setAttribute('class', 'wf-edge-hit');
    hitPath.setAttribute('d', d);
    hitPath.addEventListener('click', (e) => {
      e.stopPropagation();
      if (confirm('删除此连线？')) {
        canvasEdges = canvasEdges.filter(ed => ed.id !== edge.id);
        renderEdges();
      }
    });
    svg.appendChild(hitPath);

    // Visible edge
    const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');
    path.setAttribute('class', 'wf-edge');
    path.setAttribute('d', d);
    path.setAttribute('marker-end', 'url(#arrowhead)');
    svg.appendChild(path);
  });

  // Connection preview line
  if (connectingFrom && connectPreviewMouse) {
    const src = canvasNodes.find(n => n.id === connectingFrom);
    if (src) {
      const srcEl = document.getElementById('node-' + src.id);
      if (srcEl) {
        const x1 = src.x + srcEl.offsetWidth;
        const y1 = src.y + srcEl.offsetHeight / 2;
        const x2 = connectPreviewMouse.x;
        const y2 = connectPreviewMouse.y;
        const dx = Math.max(Math.abs(x2 - x1) * 0.5, 60);
        const d = `M${x1},${y1} C${x1 + dx},${y1} ${x2 - dx},${y2} ${x2},${y2}`;
        const preview = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        preview.setAttribute('class', 'wf-edge-preview');
        preview.setAttribute('d', d);
        svg.appendChild(preview);
      }
    }
  }
}

// ── Select node ──
function selectNode(nodeId) {
  selectedNodeId = nodeId;
  document.querySelectorAll('.wf-node').forEach(el => el.classList.remove('selected'));
  const el = document.getElementById('node-' + nodeId);
  if (el) el.classList.add('selected');
  renderNodeProps(nodeId);
}

function renderPropsEmpty() {
  document.getElementById('wf-props-node-type').textContent = '';
  document.getElementById('wf-props-body').innerHTML = `
    <div class="wf-props-empty">
      <div class="wf-props-empty-icon">⬡</div>
      <div class="wf-props-empty-text">点击节点查看属性</div>
    </div>`;
}

function renderNodeProps(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  const typeLabels = { start:'开始', end:'结束', llm:'大模型', tool:'工具调用', knowledge:'知识库检索', condition:'条件分支' };
  document.getElementById('wf-props-node-type').textContent = typeLabels[node.type] || node.type.toUpperCase();
  const cfg = node.config || {};
  let html = '';
  if (node.type === 'llm') {
    html = `
    <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
      <div style="width:28px;height:28px;border-radius:7px;background:var(--accent-dim);border:1px solid rgba(26,107,255,0.3);display:flex;align-items:center;justify-content:center;font-size:14px;flex-shrink:0">&#10022;</div>
      <input class="prop-input" style="flex:1;font-size:13px;font-weight:600" value="${escHtml(node.label)}" onchange="updateNodeProp('${nodeId}','label',this.value)" />
      <div style="display:flex;gap:6px">
        <button style="background:none;border:none;cursor:pointer;color:var(--text3);font-size:16px;padding:2px 4px" title="运行节点">&#9654;</button>
        <button style="background:none;border:none;cursor:pointer;color:var(--text3);font-size:16px;padding:2px 4px" title="更多">&#8943;</button>
      </div>
    </div>
    <div style="font-size:11px;color:var(--text3);margin-bottom:14px">调用大语言模型，根据输入参数和提示词生成回复。<span style="color:var(--accent);cursor:pointer">了解更多</span></div>`;
  } else {
    html = `
    <div class="prop-group">
      <div class="prop-label">节点名称</div>
      <input class="prop-input" value="${escHtml(node.label)}" onchange="updateNodeProp('${nodeId}','label',this.value)" />
    </div>`;
  }
  if (node.type === 'start') {
    const vars = cfg.variables || [];
    const sysParams = [
      { name: 'rawQuery', type: 'String', desc: '用户输入的原始内容' },
      { name: 'chatHistory', type: 'String', desc: '用户与应用的对话历史（轮数...' },
      { name: 'fileUrls', type: 'Array<String>', desc: '用户在应用对话中上传的文件...' },
      { name: 'fileNames', type: 'Array<String>', desc: '用户在对话中上传的文件名' },
      { name: 'end_user_id', type: 'String', desc: '终端用户的唯一id，应用调用...' },
      { name: 'conversation_id', type: 'String', desc: '会话id，是会话的唯一标识。' },
      { name: 'request_id', type: 'String', desc: '本次请求的id，便于记录与调试' },
      { name: 'fileIds', type: 'Array<String>', desc: '用户在对话中上传的文件id' },
    ];
    const sysParamsHtml = sysParams.map(p => `
      <div style="display:flex;align-items:center;gap:8px;padding:6px 0;border-bottom:1px solid var(--border)">
        <span style="font-size:12px;color:var(--text);font-family:var(--mono);min-width:110px;flex-shrink:0">${p.name}</span>
        <span style="font-size:11px;color:var(--text2);background:var(--surface2);border:1px solid var(--border2);border-radius:4px;padding:1px 6px;flex-shrink:0">${p.type}</span>
        <span style="font-size:11px;color:var(--text2);overflow:hidden;text-overflow:ellipsis;white-space:nowrap">${p.desc}</span>
      </div>`).join('');
    const customVarsHtml = vars.map((v, i) => `
      <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:12px;margin-bottom:10px;position:relative">
        <button class="btn" style="position:absolute;top:8px;right:8px;padding:2px 7px;color:var(--text2);font-size:16px;line-height:1" onclick="removeStartVar('${nodeId}',${i})" title="删除">⊖</button>
        <div style="margin-bottom:8px">
          <div style="font-size:11px;color:var(--text2);margin-bottom:4px">参数名 <span style="color:var(--red)">*</span></div>
          <div style="position:relative">
            <input class="prop-input" style="padding-right:50px" value="${escHtml(v.name)}" placeholder="请输入参数名" maxlength="50" oninput="updateStartVar('${nodeId}',${i},'name',this.value)" />
            <span style="position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:10px;color:var(--text3)">${(v.name||'').length} / 50</span>
          </div>
        </div>
        <div style="margin-bottom:8px">
          <div style="font-size:11px;color:var(--text2);margin-bottom:4px">变量类型 <span style="color:var(--red)">*</span></div>
          <select class="prop-select" onchange="updateStartVar('${nodeId}',${i},'type',this.value)">
            <option value="String" ${v.type==='String'?'selected':''}>String</option>
            <option value="Number" ${v.type==='Number'?'selected':''}>Number</option>
            <option value="Boolean" ${v.type==='Boolean'?'selected':''}>Boolean</option>
            <option value="Array<String>" ${v.type==='Array<String>'?'selected':''}>Array&lt;String&gt;</option>
            <option value="Object" ${v.type==='Object'?'selected':''}>Object</option>
          </select>
        </div>
        <div>
          <div style="font-size:11px;color:var(--text2);margin-bottom:4px">参数描述 <span style="color:var(--red)">*</span></div>
          <div style="position:relative">
            <textarea class="prop-input prop-textarea" style="min-height:72px;padding-bottom:20px;resize:vertical" placeholder="请准确描述该参数所代表的含义，这将帮助大模型更好得理解用户意图" maxlength="256" oninput="updateStartVar('${nodeId}',${i},'description',this.value)">${escHtml(v.description||'')}</textarea>
            <span style="position:absolute;right:8px;bottom:6px;font-size:10px;color:var(--text3)">${(v.description||'').length} / 256</span>
          </div>
        </div>
      </div>`).join('');
    html += `
      <div class="prop-group">
        <div class="prop-label">输入 <span style="font-size:11px;color:var(--text3);font-weight:400;margin-left:4px">ⓘ</span></div>
        <div style="background:var(--surface2);border:1px solid var(--border2);border-radius:8px;padding:10px 12px;margin-bottom:12px">
          <div style="display:flex;align-items:center;gap:6px;margin-bottom:6px;cursor:pointer">
            <span style="color:var(--orange);font-size:12px;font-weight:600">▼ 系统参数</span>
            <span style="font-size:11px;color:var(--text3)">ⓘ</span>
          </div>
          ${sysParamsHtml}
        </div>
        ${customVarsHtml}
        <button class="btn" style="width:100%;padding:8px;border:1px dashed var(--border2);color:var(--text2);background:transparent;border-radius:8px;font-size:13px" onclick="addStartVar('${nodeId}')">+ 添加参数</button>
      </div>`;
  } else if (node.type === 'llm') {
    const inputs = cfg.inputs || [];
    const modelVal = cfg.model || 'claude-sonnet-4-6';
    const sysPrompt = cfg.system_prompt || '';
    const userPrompt = cfg.user_prompt || '';
    const sysOpen = cfg._sysOpen !== false;
    const outputName = cfg.output_name || 'output';
    const outputType = cfg.output_type || 'String';
    const outputDesc = cfg.output_desc || '';

    const inputRowsHtml = inputs.map((inp, i) => `
      <div class="llm-input-row" style="margin-bottom:6px">
        <input class="llm-input-field" placeholder="请输入参数名" value="${escHtml(inp.name||'')}"
          oninput="updateLlmInput('${nodeId}',${i},'name',this.value)" />
        <select class="llm-type-select" onchange="updateLlmInput('${nodeId}',${i},'type',this.value)">
          <option value="引用" ${(inp.type||'引用')==='引用'?'selected':''}>引用</option>
          <option value="String" ${inp.type==='String'?'selected':''}>String</option>
          <option value="Number" ${inp.type==='Number'?'selected':''}>Number</option>
        </select>
        <div style="display:flex;gap:4px;align-items:center">
          <select class="llm-ref-select" style="flex:1" onchange="updateLlmInput('${nodeId}',${i},'ref',this.value)">
            ${refParamOptions(inp.ref)}
          </select>
          <span style="cursor:pointer;color:var(--text3);font-size:14px;padding:0 2px" onclick="removeLlmInput('${nodeId}',${i})" title="删除">×</span>
        </div>
      </div>`).join('');

    html += `
      <!-- 大模型配置模板 -->
      <button class="llm-tpl-btn" onclick="void(0)">
        <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>
        大模型配置模板
      </button>

      <!-- 模型选择 -->
      <div class="llm-model-row">
        <div class="llm-model-label">
          <span class="req">*</span> 模型
          <span class="help" title="选择要调用的大语言模型">?</span>
        </div>
        <button class="llm-advanced-btn" onclick="void(0)">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="3"/><path d="M19.07 4.93a10 10 0 0 1 0 14.14M4.93 4.93a10 10 0 0 0 0 14.14"/></svg>
          高级
        </button>
      </div>
      <div class="llm-model-select-wrap">
        <div class="llm-model-icon">✦</div>
        <select class="llm-model-select" onchange="updateNodeCfg('${nodeId}','model',this.value);refreshNodeCard('${nodeId}')">
          <option value="claude-sonnet-4-6" ${modelVal==='claude-sonnet-4-6'?'selected':''}>Claude Sonnet 4.6</option>
          <option value="claude-opus-4-6" ${modelVal==='claude-opus-4-6'?'selected':''}>Claude Opus 4.6</option>
          <option value="claude-haiku-4-5-20251001" ${modelVal==='claude-haiku-4-5-20251001'?'selected':''}>Claude Haiku 4.5</option>
          <option value="deepseek-v3" ${modelVal==='deepseek-v3'?'selected':''}>DeepSeek-V3.2</option>
        </select>
        <div class="llm-model-tune" title="参数调节">
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="14" height="14"><line x1="4" y1="6" x2="20" y2="6"/><line x1="4" y1="12" x2="20" y2="12"/><line x1="4" y1="18" x2="20" y2="18"/><circle cx="8" cy="6" r="2" fill="currentColor" stroke="none"/><circle cx="16" cy="12" r="2" fill="currentColor" stroke="none"/><circle cx="10" cy="18" r="2" fill="currentColor" stroke="none"/></svg>
        </div>
      </div>

      <div class="llm-divider"></div>

      <!-- 输入 -->
      <div class="llm-section">
        <div class="llm-section-header" style="border-bottom:none;margin-bottom:6px;padding-bottom:4px">
          <div class="llm-section-title">
            输入
            <span class="help" title="配置传入大模型的参数">?</span>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 80px 1fr;gap:6px;margin-bottom:6px">
          <div class="llm-input-col-label">参数名</div>
          <div class="llm-input-col-label">类型</div>
          <div class="llm-input-col-label">值</div>
        </div>
        <div id="llm-inputs-${nodeId}">${inputRowsHtml}</div>
        <button class="btn" style="width:100%;padding:7px;border:1px dashed var(--border2);color:var(--text2);background:transparent;border-radius:7px;font-size:12px;margin-top:4px"
          onclick="addLlmInput('${nodeId}')">+ 添加参数</button>
      </div>

      <div class="llm-divider"></div>

      <!-- 提示词 -->
      <div class="llm-section">
        <div style="font-size:13px;font-weight:600;color:var(--text);margin-bottom:10px">提示词</div>

        <!-- 系统提示词 -->
        <div>
          <div class="llm-prompt-toggle" onclick="toggleLlmSection('${nodeId}','sys')">
            <svg class="llm-prompt-toggle-arrow ${sysOpen?'open':''}" id="llm-sys-arrow-${nodeId}" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polyline points="9 18 15 12 9 6"/></svg>
            <div class="llm-prompt-label">
              系统提示词
              <span class="help" title="设置模型的角色和行为规范">?</span>
            </div>
            <div class="llm-prompt-actions">
              <button class="llm-action-btn" onclick="event.stopPropagation()" title="AI优化">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>
                优化
              </button>
              <button class="llm-action-btn" onclick="event.stopPropagation()" title="导入模板">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                模板
              </button>
            </div>
          </div>
          <div id="llm-sys-body-${nodeId}" style="display:${sysOpen?'block':'none'}">
            <textarea class="llm-prompt-textarea" placeholder="设置大模型的角色、行为规范和背景知识..."
              oninput="updateNodeCfg('${nodeId}','system_prompt',this.value)">${escHtml(sysPrompt)}</textarea>
          </div>
        </div>

        <!-- 用户提示词 -->
        <div style="margin-top:8px">
          <div class="llm-prompt-toggle" style="cursor:default">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="width:14px;height:14px;color:var(--text3);transform:rotate(90deg)"><polyline points="9 18 15 12 9 6"/></svg>
            <div class="llm-prompt-label">
              用户提示词 <span class="req">*</span>
              <span class="help" title="用户输入的提示词，可用{{参数名}}引用参数">?</span>
            </div>
            <div class="llm-prompt-actions">
              <button class="llm-action-btn" onclick="void(0)" title="AI优化">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M12 2l2 7h7l-5.5 4 2 7L12 16l-5.5 4 2-7L3 9h7z"/></svg>
                优化
              </button>
              <button class="llm-action-btn" onclick="void(0)" title="导入模板">
                <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>
                模板
              </button>
            </div>
          </div>
          <textarea class="llm-prompt-textarea" style="min-height:120px"
            placeholder="输入大模型的用户提示词，使大模型实现对应功能，通过插入{{参数名}}可以引用对应的参数值"
            oninput="updateNodeCfg('${nodeId}','user_prompt',this.value)">${escHtml(userPrompt)}</textarea>
        </div>
      </div>

      <div class="llm-divider"></div>

      <!-- 输出 -->
      <div class="llm-section">
        <div class="llm-section-header" style="border-bottom:none;margin-bottom:6px;padding-bottom:4px">
          <div class="llm-section-title">
            输出
            <span class="help" title="大模型的输出参数">?</span>
          </div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 70px 1fr;gap:6px;margin-bottom:6px">
          <div class="llm-input-col-label">参数名</div>
          <div class="llm-input-col-label">类型</div>
          <div class="llm-input-col-label">描述</div>
        </div>
        <div style="display:grid;grid-template-columns:1fr 70px 1fr;gap:6px">
          <input class="llm-input-field" value="${escHtml(outputName)}"
            oninput="updateNodeCfg('${nodeId}','output_name',this.value)" />
          <select class="llm-type-select" onchange="updateNodeCfg('${nodeId}','output_type',this.value)">
            <option value="String" ${outputType==='String'?'selected':''}>String</option>
            <option value="Number" ${outputType==='Number'?'selected':''}>Number</option>
            <option value="Object" ${outputType==='Object'?'selected':''}>Object</option>
          </select>
          <input class="llm-input-field" placeholder="请输入描述" value="${escHtml(outputDesc)}"
            oninput="updateNodeCfg('${nodeId}','output_desc',this.value)" />
        </div>
      </div>
    `;
  } else if (node.type === 'tool') {
    const toolArgDefs = {
      calculator: [{ key: 'expression', label: '表达式', placeholder: '如: 2+3*4 或 {{input}}' }],
      get_current_time: [],
      get_weather: [{ key: 'city', label: '城市', placeholder: '如: 北京 或 {{input}}' }],
      text_analyzer: [{ key: 'text', label: '文本', placeholder: '{{input}}' }],
      unit_converter: [
        { key: 'value', label: '数值', placeholder: '如: 100 或 {{input}}' },
        { key: 'from_unit', label: '原单位', placeholder: '如: km' },
        { key: 'to_unit', label: '目标单位', placeholder: '如: mile' },
      ],
      word_counter: [
        { key: 'text', label: '文本', placeholder: '{{input}}' },
        { key: 'target_word', label: '目标词语', placeholder: '要统计的词' },
      ],
    };
    const args = cfg.args || {};
    const argFields = toolArgDefs[cfg.tool_name] || [];
    const argsHtml = argFields.map(f => `
      <div class="prop-group">
        <div class="prop-label">${f.label}</div>
        <input class="prop-input" value="${escHtml(args[f.key]||'')}" placeholder="${escHtml(f.placeholder||'')}" onchange="updateNodeArg('${nodeId}','${f.key}',this.value)" />
      </div>`).join('');
    html += `
      <div class="prop-group">
        <div class="prop-label">工具名称</div>
        <select class="prop-select" onchange="updateNodeCfg('${nodeId}','tool_name',this.value);renderNodeProps('${nodeId}')">
          <option value="">选择工具</option>
          <option value="calculator" ${cfg.tool_name==='calculator'?'selected':''}>计算器</option>
          <option value="get_current_time" ${cfg.tool_name==='get_current_time'?'selected':''}>获取时间</option>
          <option value="get_weather" ${cfg.tool_name==='get_weather'?'selected':''}>天气查询</option>
          <option value="text_analyzer" ${cfg.tool_name==='text_analyzer'?'selected':''}>文本分析</option>
          <option value="unit_converter" ${cfg.tool_name==='unit_converter'?'selected':''}>单位换算</option>
          <option value="word_counter" ${cfg.tool_name==='word_counter'?'selected':''}>词语统计</option>
        </select>
      </div>
      ${argsHtml}
      <div style="font-size:10px;color:var(--text3);margin-top:-8px;margin-bottom:8px">提示：参数值中可用 <code style="color:var(--accent)">{{input}}</code> 引用上一节点输出</div>`;
  } else if (node.type === 'knowledge') {
    html += `
      <div class="prop-group">
        <div class="prop-label">关联知识库</div>
        <select class="prop-select" onchange="updateNodeCfg('${nodeId}','kb_id',this.value)">
          <option value="">选择知识库</option>
          ${kbList.map(kb => `<option value="${kb.kb_id}" ${cfg.kb_id===kb.kb_id?'selected':''}>${escHtml(kb.name)}</option>`).join('')}
        </select>
      </div>`;
  } else if (node.type === 'condition') {
    html += `
      <div class="prop-group">
        <div class="prop-label">条件表达式</div>
        <input class="prop-input" value="${escHtml(cfg.condition||'')}" placeholder="如: contains:成功 或 equals:1" oninput="updateNodeCfg('${nodeId}','condition',this.value)" />
        <div style="font-size:10px;color:var(--text3);margin-top:4px">
          支持：<code style="color:var(--accent)">contains:关键词</code>、<code style="color:var(--accent)">equals:值</code>、<code style="color:var(--accent)">startswith:前缀</code><br>
          匹配走 <strong style="color:var(--green)">true</strong> 分支，不匹配走 <strong style="color:var(--red)">false</strong> 分支
        </div>
      </div>`;
  }
  html += `
    <div class="prop-group" style="margin-top:20px">
      <button class="btn btn-danger" style="width:100%" onclick="deleteNode('${nodeId}')">删除节点</button>
    </div>`;
  document.getElementById('wf-props-body').innerHTML = html;
}

function updateNodeProp(nodeId, key, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (node) {
    node[key] = val;
    const el = document.getElementById('node-' + nodeId);
    if (el) {
      const labelEl = el.querySelector('.wf-node-label');
      if (labelEl) labelEl.textContent = val;
    }
  }
}

function updateNodeArg(nodeId, key, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (node) {
    node.config = node.config || {};
    node.config.args = node.config.args || {};
    node.config.args[key] = val;
  }
}

function addStartVar(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.variables = node.config.variables || [];
  node.config.variables.push({ name: '', type: 'String', description: '' });
  renderNodeProps(nodeId);
  renderRunInputs();
}

function removeStartVar(nodeId, idx) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config.variables.splice(idx, 1);
  renderNodeProps(nodeId);
  renderRunInputs();
}

function updateStartVar(nodeId, idx, key, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config.variables[idx][key] = val;
}

function updateNodeCfg(nodeId, key, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (node) {
    node.config = node.config || {};
    if (key === 'tool_name') node.config.args = {}; // 切换工具时清空参数
    node.config[key] = val;
    const el = document.getElementById('node-' + nodeId);
    if (el) {
      const descEl = el.querySelector('.wf-node-desc');
      if (descEl) descEl.textContent = getNodeDesc(node);
    }
  }
}

function refreshNodeCard(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  const el = document.getElementById('node-' + nodeId);
  if (!el) return;
  const newEl = createNodeEl(node);
  el.parentNode.replaceChild(newEl, el);
}

function addLlmInput(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.inputs = node.config.inputs || [];
  node.config.inputs.push({ name: '', type: '引用', ref: '' });
  renderNodeProps(nodeId);
  refreshLlmNodeInputs(nodeId);
}

function removeLlmInput(nodeId, idx) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  (node.config.inputs || []).splice(idx, 1);
  renderNodeProps(nodeId);
  refreshLlmNodeInputs(nodeId);
}

function refreshLlmNodeInputs(nodeId) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node || node._expanded === false) return;
  const container = document.getElementById(`node-llm-inputs-${nodeId}`);
  if (!container) return;
  const inputs = (node.config || {}).inputs || [];
  container.innerHTML = inputs.map((inp, i) => `
    <div style="display:grid;grid-template-columns:1fr 72px 1fr 16px;gap:5px;align-items:center;margin-bottom:5px">
      <input class="node-llm-field" placeholder="参数名" value="${escHtml(inp.name||'')}"
        oninput="updateLlmInput('${nodeId}',${i},'name',this.value)" />
      <select class="node-llm-sel" onchange="updateLlmInput('${nodeId}',${i},'type',this.value)">
        <option value="引用" ${(inp.type||'引用')==='引用'?'selected':''}>引用</option>
        <option value="String" ${inp.type==='String'?'selected':''}>String</option>
        <option value="Number" ${inp.type==='Number'?'selected':''}>Number</option>
      </select>
      <select class="node-llm-sel" onchange="updateLlmInput('${nodeId}',${i},'ref',this.value)">
        ${refParamOptions(inp.ref)}
      </select>
      <span style="cursor:pointer;color:var(--text3);font-size:13px;text-align:center" onclick="removeLlmInput('${nodeId}',${i})">×</span>
    </div>`).join('');
}

function updateLlmInput(nodeId, idx, field, val) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  node.config.inputs = node.config.inputs || [];
  if (node.config.inputs[idx]) node.config.inputs[idx][field] = val;
}

function toggleLlmSection(nodeId, section) {
  const node = canvasNodes.find(n => n.id === nodeId);
  if (!node) return;
  node.config = node.config || {};
  if (section === 'sys') {
    node.config._sysOpen = node.config._sysOpen === false ? true : false;
    const isOpen = node.config._sysOpen !== false;
    // Handle both props panel and expanded node card
    ['llm-sys-body-', 'node-llm-sys-body-'].forEach(prefix => {
      const body = document.getElementById(`${prefix}${nodeId}`);
      if (body) body.style.display = isOpen ? 'block' : 'none';
    });
    ['llm-sys-arrow-', 'node-llm-sys-arrow-'].forEach(prefix => {
      const arrow = document.getElementById(`${prefix}${nodeId}`);
      if (arrow) arrow.classList.toggle('open', isOpen);
    });
  }
}

function deleteNode(nodeId) {
  canvasNodes = canvasNodes.filter(n => n.id !== nodeId);
  canvasEdges = canvasEdges.filter(e => e.source !== nodeId && e.target !== nodeId);
  selectedNodeId = null;
  renderCanvas();
  renderPropsEmpty();
}

function deleteSelectedNode() {
  if (selectedNodeId) deleteNode(selectedNodeId);
}

function clearCanvas() {
  if (!confirm('确认清空画布？')) return;
  canvasNodes = [];
  canvasEdges = [];
  selectedNodeId = null;
  connectingFrom = null;
  renderCanvas();
  renderPropsEmpty();
  zoomReset();
  document.getElementById('wf-canvas-empty').style.display = 'flex';
}

// ── Run workflow ──
function renderRunInputs() {
  const startNode = canvasNodes.find(n => n.type === 'start');
  const vars = startNode && startNode.config && startNode.config.variables || [];
  const container = document.getElementById('wf-run-inputs');
  if (!container) return;
  if (vars.length === 0) {
    container.innerHTML = `<input class="wf-run-input" id="wf-run-input" placeholder="输入测试内容，运行工作流..." style="flex:1" />`;
  } else {
    container.innerHTML = vars.map(v => `
      <div style="display:flex;flex-direction:column;gap:2px;flex:1;min-width:80px">
        <span style="font-size:10px;color:var(--text3)">${escHtml(v.name||'变量')}</span>
        <input class="wf-run-input" id="wf-var-${escHtml(v.name)}" placeholder="${escHtml(v.default||v.name||'')}" style="width:100%" />
      </div>`).join('');
  }
}

async function runWorkflow() {
  if (!currentWfId) return;
  // 收集变量值
  const startNode = canvasNodes.find(n => n.type === 'start');
  const vars = startNode && startNode.config && startNode.config.variables || [];
  let input, variables = {};
  if (vars.length === 0) {
    input = (document.getElementById('wf-run-input') || {}).value?.trim() || '';
    if (!input) return;
  } else {
    for (const v of vars) {
      const el = document.getElementById('wf-var-' + v.name);
      variables[v.name] = el ? el.value.trim() || v.default || '' : v.default || '';
    }
    // 第一个变量作为主 input
    input = variables[vars[0].name] || '';
    if (!input) return;
  }
  const btn = document.getElementById('wf-run-btn');
  btn.disabled = true;
  btn.textContent = '运行中...';
  const logEl = document.getElementById('wf-run-log');
  logEl.classList.add('open');
  logEl.innerHTML = '<div style="font-size:11px;color:var(--text3);padding:4px 0">执行中...</div>';
  try {
    // 运行前先同步保存最新节点/连线到后端
    const wf = wfList.find(w => w.id === currentWfId);
    await fetch(API + '/workflow/' + currentWfId, {
      method: 'PUT',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        name: wf ? wf.name : '未命名',
        description: wf ? wf.description : '',
        nodes: canvasNodes,
        edges: canvasEdges
      })
    });
    const r = await fetch(API + '/workflow/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ workflow_id: currentWfId, input, variables, session_id: 'wf_' + currentWfId, ...getCreds() })
    });
    const d = await r.json();
    const typeColors = { start:'var(--green)', end:'var(--red)', llm:'var(--accent)', tool:'var(--orange)', knowledge:'var(--blue)', condition:'#fbbf24' };
    logEl.innerHTML = (d.execution_log || []).map(step => `
      <div class="log-step">
        <span class="log-step-type" style="background:${typeColors[step.type]||'var(--border2)'}22;color:${typeColors[step.type]||'var(--text3)'};">${step.type}</span>
        <div class="log-step-output"><strong style="color:var(--text)">${escHtml(step.node)}</strong><br>${escHtml((step.output||'').substring(0,200))}</div>
      </div>
    `).join('') + `<div style="padding:8px 0;font-size:12px;color:var(--green);border-top:1px solid var(--border);margin-top:6px">✓ 最终输出：${escHtml(d.response||'')}</div>`;
  } catch(e) {
    logEl.innerHTML = `<div style="color:var(--red);font-size:11px">运行失败: ${e.message}</div>`;
  }
  btn.disabled = false;
  btn.innerHTML = '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" width="13" height="13"><polygon points="5 3 19 12 5 21 5 3"/></svg> 运行';
}

function toggleRunLog() {
  document.getElementById('wf-run-log').classList.toggle('open');
}
