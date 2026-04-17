// chat.js — 对话

// ══════════════════════════════════════
// CHAT
// ══════════════════════════════════════
let chatSessionId = 'session_' + Math.random().toString(36).substr(2,8);

const chatInput = document.getElementById('chat-input');
chatInput.addEventListener('input', () => {
  chatInput.style.height = 'auto';
  chatInput.style.height = Math.min(chatInput.scrollHeight, 100) + 'px';
  document.getElementById('chat-char-count').textContent = chatInput.value.length + ' 字符';
});
chatInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); sendChatMessage(); }
});

async function sendChatMessage() {
  const input = chatInput.value.trim();
  if (!input) return;
  if (!ensureCreds()) return;
  const welcome = document.getElementById('chat-welcome');
  if (welcome) welcome.remove();
  appendMsg('user', input);
  chatInput.value = '';
  chatInput.style.height = 'auto';
  document.getElementById('chat-char-count').textContent = '0 字符';
  document.getElementById('chat-send-btn').disabled = true;
  const kbId = document.getElementById('chat-kb-select').value;
  const agentId = document.getElementById('chat-agent-select').value;

  // Create agent message bubble for streaming
  const container = document.getElementById('chat-messages');
  const msgDiv = document.createElement('div');
  msgDiv.className = 'msg agent';
  const time = new Date().toLocaleTimeString('zh-CN', { hour:'2-digit', minute:'2-digit' });
  msgDiv.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-body">
      <div class="msg-meta">AGENT · ${time}</div>
      <div class="msg-bubble"><span class="dots"><span></span><span></span><span></span></span></div>
      <div class="msg-steps" style="display:none"></div>
    </div>`;
  container.appendChild(msgDiv);
  msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });

  const bubble = msgDiv.querySelector('.msg-bubble');
  const stepsDiv = msgDiv.querySelector('.msg-steps');
  let fullText = '';

  try {
    // 如果选择了智能体，走 agent chat 端点（非流式）
    if (agentId) {
      const resp = await fetch(API + '/agent/' + agentId + '/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: input, session_id: chatSessionId, ...getCreds() })
      });
      const data = await resp.json();
      fullText = data.response || '处理完成';
      bubble.innerHTML = renderMd(fullText);
      // 工具调用步骤
      if (data.steps && data.steps.length) {
        stepsDiv.style.display = '';
        data.steps.forEach(s => {
          const inputStr = Object.entries(s.input||{}).map(([k,v])=>`${k}: ${v}`).join(', ');
          stepsDiv.innerHTML += `<div class="msg-step" onclick="this.classList.toggle('open')">
            <div class="msg-step-name">🔧 ${escHtml(s.tool)}(${escHtml(inputStr)})</div>
            <div class="msg-step-detail">${escHtml(typeof s.output === 'string' ? s.output : JSON.stringify(s.output, null, 2))}</div>
          </div>`;
        });
      }
      // 工作流执行日志
      if (data.execution_log) {
        stepsDiv.style.display = '';
        data.execution_log.forEach(l => {
          stepsDiv.innerHTML += `<div class="msg-step" onclick="this.classList.toggle('open')">
            <div class="msg-step-name">📋 ${escHtml(l.node || l.phase)} (${l.type || l.phase})</div>
            <div class="msg-step-detail">${escHtml(typeof l.output === 'string' ? l.output : JSON.stringify(l.output, null, 2))}</div>
          </div>`;
        });
      }
      // 多智能体协同日志
      if (data.multi_agent_log && data.multi_agent_log.length) {
        stepsDiv.style.display = '';
        data.multi_agent_log.forEach(l => {
          const round = l.round ? ` (第${l.round}轮)` : '';
          stepsDiv.innerHTML += `<div class="msg-step" onclick="this.classList.toggle('open')">
            <div class="msg-step-name">🤝 ${escHtml(l.agent_name)}${round}</div>
            <div class="msg-step-detail">${escHtml(l.output || '')}</div>
          </div>`;
        });
      }
      // 自主规划日志
      if (data.autonomous_log && data.autonomous_log.length) {
        stepsDiv.style.display = '';
        data.autonomous_log.forEach(l => {
          let icon = '📋', title = '', detail = '';
          if (l.phase === 'plan') {
            icon = '📝'; title = '任务规划';
            detail = (l.content.steps || []).map((s,i) => `${i+1}. ${s.description}${s.tool ? ' ['+s.tool+']' : ''}`).join('\n');
          } else if (l.phase === 'execute') {
            icon = '⚡'; title = `步骤${l.step}: ${l.description}`;
            detail = l.result || '';
            if (l.tool_steps && l.tool_steps.length) {
              detail += '\n\n工具调用:\n' + l.tool_steps.map(ts => `  ${ts.tool}(${JSON.stringify(ts.input)}) → ${ts.output}`).join('\n');
            }
          } else if (l.phase === 'reflect') {
            icon = '🔍'; title = `反思 (步骤${l.step})`;
            const c = l.content || {};
            detail = `评估: ${c.assessment || ''}\n需要重规划: ${c.need_replan ? '是' : '否'}${c.reason ? '\n原因: '+c.reason : ''}`;
          } else if (l.phase === 'replan') {
            icon = '🔄'; title = '重新规划';
            detail = (l.content.steps || []).map((s,i) => `${i+1}. ${s.description}`).join('\n');
          }
          stepsDiv.innerHTML += `<div class="msg-step" onclick="this.classList.toggle('open')">
            <div class="msg-step-name">${icon} ${escHtml(title)}</div>
            <div class="msg-step-detail">${escHtml(detail)}</div>
          </div>`;
        });
      }
    } else {
      // 普通流式 chat
      const resp = await fetch(API + '/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ message: input, session_id: chatSessionId, kb_id: kbId || null, ...getCreds() })
    });
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.startsWith('data: ')) continue;
        const data = JSON.parse(line.slice(6));
        if (data.type === 'token') {
          fullText += data.content;
          bubble.innerHTML = renderMd(fullText) + '<span class="stream-cursor">▍</span>';
          msgDiv.scrollIntoView({ behavior: 'smooth', block: 'end' });
        } else if (data.type === 'tool') {
          stepsDiv.style.display = '';
          const s = data.step;
          const inputStr = Object.entries(s.input||{}).map(([k,v])=>`${k}: ${v}`).join(', ');
          stepsDiv.innerHTML += `<div class="msg-step" onclick="this.classList.toggle('open')">
            <div class="msg-step-name">🔧 ${escHtml(s.tool)}(${escHtml(inputStr)})</div>
            <div class="msg-step-detail">${escHtml(typeof s.output === 'string' ? s.output : JSON.stringify(s.output, null, 2))}</div>
          </div>`;
        } else if (data.type === 'done') {
          bubble.innerHTML = renderMd(fullText || '处理完成');
        }
      }
    }
    // Final render without cursor
    bubble.innerHTML = renderMd(fullText || '处理完成');
    } // end else (streaming)
  } catch(e) {
    bubble.innerHTML = renderMd('⚠️ 错误: ' + e.message);
  }
  document.getElementById('chat-send-btn').disabled = false;
  document.getElementById('chat-messages').lastElementChild?.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function sendChip(el) {
  chatInput.value = el.textContent;
  sendChatMessage();
}

function appendMsg(role, text, steps) {
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.className = 'msg ' + role;
  const time = new Date().toLocaleTimeString('zh-CN', { hour:'2-digit', minute:'2-digit' });
  const avatar = role === 'user' ? '👤' : '⬡';
  const label = role === 'user' ? 'YOU' : 'AGENT';
  let stepsHtml = '';
  if (steps && steps.length) {
    stepsHtml = '<div class="msg-steps">' + steps.map(s => {
      const inputStr = Object.entries(s.input||{}).map(([k,v])=>`${k}: ${v}`).join(', ');
      return `<div class="msg-step" onclick="this.classList.toggle('open')">
        <div class="msg-step-name">🔧 ${escHtml(s.tool)}(${escHtml(inputStr)})</div>
        <div class="msg-step-detail">${escHtml(typeof s.output === 'string' ? s.output : JSON.stringify(s.output, null, 2))}</div>
      </div>`;
    }).join('') + '</div>';
  }
  const bubbleContent = role === 'user' ? escHtml(text) : renderMd(text);
  div.innerHTML = `
    <div class="msg-avatar">${avatar}</div>
    <div class="msg-body">
      <div class="msg-meta">${label} · ${time}</div>
      <div class="msg-bubble">${bubbleContent}</div>
      ${stepsHtml}
    </div>`;
  container.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
}

function appendThinking() {
  const id = 'think-' + Date.now();
  const container = document.getElementById('chat-messages');
  const div = document.createElement('div');
  div.id = id; div.className = 'msg agent';
  div.innerHTML = `
    <div class="msg-avatar">⬡</div>
    <div class="msg-body">
      <div class="msg-meta">AGENT · 思考中</div>
      <div class="thinking-bubble">处理中 <div class="dots"><span></span><span></span><span></span></div></div>
    </div>`;
  container.appendChild(div);
  div.scrollIntoView({ behavior: 'smooth', block: 'end' });
  return id;
}

function removeThinking(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

function newChatSession() {
  chatSessionId = 'session_' + Math.random().toString(36).substr(2,8);
  const container = document.getElementById('chat-messages');
  container.innerHTML = `
    <div class="chat-welcome" id="chat-welcome">
      <div class="chat-welcome-icon">⬡</div>
      <div class="chat-welcome-title">AgentFlow 智能助手</div>
      <div class="chat-welcome-sub">基于 LangGraph 构建 · 多工具自动调用</div>
      <div class="chat-chips">
        <div class="chat-chip" onclick="sendChip(this)">现在几点了？</div>
        <div class="chat-chip" onclick="sendChip(this)">计算 sin(45°) × √2</div>
        <div class="chat-chip" onclick="sendChip(this)">北京今天天气</div>
        <div class="chat-chip" onclick="sendChip(this)">100公里等于多少英里</div>
      </div>
    </div>`;
}
