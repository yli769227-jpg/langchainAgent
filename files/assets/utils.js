// utils.js — API const / settings / nav / 通用工具


const API = 'http://localhost:8000';

// ══════════════════════════════════════
// SETTINGS: API Key + Base URL
// 仅存在浏览器 localStorage，不落盘后端
// ══════════════════════════════════════
let apiKey = localStorage.getItem('af_api_key') || '';
let baseUrl = localStorage.getItem('af_base_url') || '';

function getCreds() {
  return { api_key: apiKey, base_url: baseUrl };
}

function ensureCreds() {
  if (!apiKey || !baseUrl) {
    if (typeof toast === 'function') toast('请先在「设置」里填写 API Key 和 Base URL', 'error');
    openSettings();
    return false;
  }
  return true;
}

function openSettings() {
  document.getElementById('settings-api-key').value = apiKey;
  document.getElementById('settings-base-url').value = baseUrl || 'https://api.openai.com/v1';
  document.getElementById('settings-error').style.display = 'none';
  document.getElementById('settings-modal').style.display = 'flex';
}

function closeSettings() {
  document.getElementById('settings-modal').style.display = 'none';
}

function saveSettings() {
  const k = document.getElementById('settings-api-key').value.trim();
  const u = document.getElementById('settings-base-url').value.trim();
  const err = document.getElementById('settings-error');
  if (!k) { err.textContent = '请填写 API Key'; err.style.display = 'block'; return; }
  if (!u || !/^https?:\/\//.test(u)) { err.textContent = 'Base URL 必须以 http(s):// 开头'; err.style.display = 'block'; return; }
  apiKey = k; baseUrl = u;
  localStorage.setItem('af_api_key', k);
  localStorage.setItem('af_base_url', u);
  closeSettings();
  if (typeof toast === 'function') toast('已保存', 'success');
}

// ══════════════════════════════════════
// NAVIGATION
// ══════════════════════════════════════
function showPage(id, el) {
  document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
  document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
  document.getElementById('page-' + id).classList.add('active');
  if (el) el.classList.add('active');
  if (id === 'kb') loadKbList();
  if (id === 'workflow') loadWfList();
  if (id === 'tools') loadTools();
  if (id === 'chat') loadChatKbSelect();
  if (id === 'agent') loadAgentList();
}

function toast(msg, type = '') {
  const existing = document.querySelector('.toast');
  if (existing) existing.remove();
  const t = document.createElement('div');
  t.className = 'toast ' + type;
  t.textContent = msg;
  document.body.appendChild(t);
  setTimeout(() => t.remove(), 3000);
}

function openModal(id) { document.getElementById(id).classList.add('open'); }
function closeModal(id) { document.getElementById(id).classList.remove('open'); }
// ══════════════════════════════════════
// UTILS
// ══════════════════════════════════════
function escHtml(s) {
  return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

function renderMd(text) {
  let s = escHtml(text);
  // code blocks
  s = s.replace(/```(\w*)\n([\s\S]*?)```/g, '<pre style="background:var(--bg);border:1px solid var(--border);border-radius:6px;padding:10px;margin:6px 0;overflow-x:auto;font-family:var(--mono);font-size:12px;line-height:1.5"><code>$2</code></pre>');
  // inline code
  s = s.replace(/`([^`]+)`/g, '<code style="background:var(--bg);padding:1px 5px;border-radius:3px;font-family:var(--mono);font-size:12px">$1</code>');
  // bold
  s = s.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  // unordered lists
  s = s.replace(/^- (.+)$/gm, '<div style="padding-left:12px">• $1</div>');
  // ordered lists
  s = s.replace(/^(\d+)\. (.+)$/gm, '<div style="padding-left:12px">$1. $2</div>');
  return s;
}

// 获取开始节点的所有可引用参数（系统参数 + 自定义变量）
function getStartNodeParams() {
  const sysParams = ['rawQuery','chatHistory','fileUrls','fileNames','end_user_id','conversation_id','request_id','fileIds'];
  const startNode = (canvasNodes||[]).find(n => n.type === 'start');
  const customVars = (startNode && startNode.config && startNode.config.variables || []).map(v => v.name).filter(Boolean);
  return [...sysParams, ...customVars];
}

// 生成引用参数 <option> 列表
function refParamOptions(currentRef) {
  const params = getStartNodeParams();
  return `<option value="">请选择引用参数</option>` +
    params.map(p => `<option value="${escHtml(p)}" ${currentRef===p?'selected':''}>${escHtml(p)}</option>`).join('');
}
