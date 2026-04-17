// app.js — 登录 + 启动

// ── Login ──
function doLogin() {
  const pwd = document.getElementById('login-pwd').value;
  const errEl = document.getElementById('login-error');
  const input = document.getElementById('login-pwd');
  if (pwd === 'agent2024') {
    sessionStorage.setItem('af_auth', '1');
    document.getElementById('login-page').style.display = 'none';
    errEl.textContent = '';
    input.classList.remove('error');
    if (!apiKey || !baseUrl) openSettings();
  } else {
    errEl.textContent = '密码错误，请重试';
    input.classList.add('error');
    input.value = '';
    input.focus();
  }
}

// ── Init ──
if (sessionStorage.getItem('af_auth') === '1') {
  document.getElementById('login-page').style.display = 'none';
}
loadKbList();
loadWfList();
loadAgentList();
