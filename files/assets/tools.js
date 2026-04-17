// tools.js — 工具列表页

// ══════════════════════════════════════
// TOOLS PAGE
// ══════════════════════════════════════
async function loadTools() {
  try {
    const r = await fetch(API + '/tools');
    const d = await r.json();
    const grid = document.getElementById('tools-grid');
    const icons = { calculator:'🧮', get_current_time:'🕐', get_weather:'🌤', text_analyzer:'📊', unit_converter:'📐', word_counter:'🔢', search_knowledge_base:'🔍' };
    grid.innerHTML = (d.tools || []).map(t => `
      <div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--radius);padding:18px;transition:border-color 0.2s" onmouseover="this.style.borderColor='var(--border2)'" onmouseout="this.style.borderColor='var(--border)'">
        <div style="font-size:24px;margin-bottom:10px">${icons[t.name]||'🔧'}</div>
        <div style="font-family:var(--display);font-size:13px;font-weight:700;color:var(--text);margin-bottom:6px">${escHtml(t.name)}</div>
        <div style="font-size:12px;color:var(--text3);line-height:1.5">${escHtml(t.description)}</div>
      </div>
    `).join('');
  } catch(e) {}
}
