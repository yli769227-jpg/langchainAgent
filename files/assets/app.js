// app.js — 登录 + 启动

// ── Login ──
async function doLogin() {
  const pwd = document.getElementById('login-pwd').value;
  const errEl = document.getElementById('login-error');
  const input = document.getElementById('login-pwd');
  const btn = document.querySelector('.login-btn');
  if (btn) btn.disabled = true;
  errEl.textContent = '';
  try {
    const resp = await fetch(API + '/auth/login', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({password: pwd})
    });
    if (resp.ok) {
      sessionStorage.setItem('af_auth', '1');
      document.getElementById('login-page').style.display = 'none';
      input.classList.remove('error');
      if (!apiKey || !baseUrl) openSettings();
    } else if (resp.status === 401) {
      errEl.textContent = '密码错误，请重试';
      input.classList.add('error');
      input.value = '';
      input.focus();
    } else {
      errEl.textContent = `登录接口错误：HTTP ${resp.status}`;
    }
  } catch (e) {
    errEl.textContent = '无法连接后端，请确认服务已启动 (' + API + ')';
  } finally {
    if (btn) btn.disabled = false;
  }
}

// ── Init ──
if (sessionStorage.getItem('af_auth') === '1') {
  document.getElementById('login-page').style.display = 'none';
}
loadKbList();
loadWfList();
loadAgentList();
