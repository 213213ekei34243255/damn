(function(){

  document.addEventListener('DOMContentLoaded', () => {
    try {
      console.log('[VERONICA WIDGET] init');

      const thisScript = document.currentScript ||
        document.querySelector('script[src*="widget.js"]') ||
        Array.from(document.getElementsByTagName('script')).reverse().find(s=>s.src && s.src.includes('widget.js'));
      const BASE_API = (thisScript && (thisScript.dataset && thisScript.dataset.api)) ||
                       (thisScript && thisScript.getAttribute('data-api')) ||
                       'https://www.byncai.net';
      console.log('[VERONICA WIDGET] BASE_API =', BASE_API);

      // --- persistent session id (works with Redis backend) ---
      function getOrCreateSessionId() {
        try {
          const key = 'veronica_session_id_v1';
          let id = localStorage.getItem(key);

          // if you want a *fresh* chat on every page refresh instead of persisting:
          //   - replace localStorage with sessionStorage
          //   - or just always generate a new id here
          if (!id) {
            id = (window.crypto && crypto.randomUUID)
              ? crypto.randomUUID()
              : 'sess-' + Date.now() + '-' + Math.random().toString(36).slice(2,10);
            localStorage.setItem(key, id);
          }
          return id;
        } catch (e) {
          return 'sess-fallback-' + Date.now();
        }
      }
      const SESSION_ID = getOrCreateSessionId();
      // --- end session id ---

      // --- create host element and attach shadow root to isolate styles ---
      const host = document.createElement('div');
      host.id = 'vai-widget-host';
      // keep host out of visual flow (container inside shadow will be positioned)
      host.style.all = 'initial'; // reset host's own styling influence
      // attach shadow root
      const shadow = host.attachShadow({ mode: 'open' });
      // append host to body
      document.body.appendChild(host);

      // CSS moved into shadow (note: replaced :root with :host to scope CSS variables)
      const css = `
:host{
  --primary:#ff2fa6;
  --primaryLight:#0b42a7;
  --text-color:black;
  --neon:#ff2fa6;
  --primaryGradient: linear-gradient(135deg,#0b0f1a 0%,#17f38c 100%);
  --secondaryGradient: linear-gradient(180deg,#ffffff,#f2f2f2);
  --primaryBoxShadow: 0 0 25px rgba(243, 16, 148, 0.84);
  --secondaryBoxShadow:0 10px 30px rgba(0,0,0,0.45);
  --header-bg: url("${BASE_API}/static/charlie.jpeg");
}

.vai-chatbox{
  position: fixed;
  bottom: 20px;
  right: 20px;
  z-index: 2147483647 !important;
  font-family: 'Inter', Arial, Helvetica, sans-serif;
}

/* ================= BUTTON ================= */
.vai-chatbox .chatbox__button{
  text-align:right;
}

.vai-chatbox .chatbox__button button{
  padding:10px;
  border:none;
  outline:none;
  cursor:pointer;
  border-radius:50%;
  width:64px;
  height:64px;
  background: radial-gradient(circle at top,var(--neon),#0b0f1a);
  color:#000;
  display:flex;
  align-items:center;
  justify-content:center;
  box-shadow:0 0 25px rgba(243, 16, 148, 0.84);
  transition: transform .25s ease, box-shadow .25s ease;
}

.vai-chatbox .chatbox__button button:hover{
  transform: scale(1.08);
  box-shadow:0 0 40px rgba(243, 16, 148, 0.84);
}

/* ================= CONTAINER ================= */
.vai-chatbox .chatbox__support{
  display:flex;
  flex-direction:column;
  position: fixed;
  bottom: 95px;
  right: 20px;
  width:360px;
  height:470px;
  border-radius:18px;
  overflow:hidden;
  backdrop-filter: blur(20px);
  background: rgba(15,18,30,0.85);
  transform: translateY(12px);
  opacity:0;
  pointer-events:none;
  transition: all .3s ease-in-out;
  box-shadow: var(--secondaryBoxShadow);
  border: 1px solid rgba(255,255,255,0.12);
}

.vai-chatbox .chatbox--active{
  transform: translateY(0);
  opacity:1;
  pointer-events:auto;
}

/* ================= HEADER ================= */
.vai-chatbox .chatbox__header{
  background-image: var(--header-bg);
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;


  color:black;
  padding:14px;
  display:flex;
  align-items:center;
  gap:12px;
  box-shadow: inset 0 -1px 0 rgba(255,255,255,0.2);
}

.vai-chatbox .chatbox__image--header img{
  width:130px;
  height:auto;
  object-fit:contain;
  background:transparent;
  filter: drop-shadow(0 0 12px rgba(243, 16, 148, 0.84));
}

.vai-chatbox .chatbox__heading--header{
  font-size:1.05rem;
  margin:0;
  color:black;
}

.vai-chatbox .chatbox__description--header{
  font-size:0.85rem;
  color:black;
  margin:0;
}

/* ================= MESSAGES ================= */
.vai-chatbox .chatbox__messages{
  padding:14px;
  flex:1;
  overflow:auto;
  display:flex;
  flex-direction:column-reverse;
  gap:10px;
  background:
    radial-gradient(circle at top,rgba(23,243,140,0.08),transparent),
    #ffffff;
}

.vai-chatbox .messages__item{
  max-width:72%;
  padding:10px 14px;
  border-radius:16px;
  background: linear-gradient(145deg,#f0f6ff,#ffffff);
  color:#000;
  align-self:flex-start;
  word-wrap:break-word;
  white-space:pre-wrap;
  box-shadow:0 6px 16px rgba(0,0,0,0.08);
}

.vai-chatbox .messages__item--operator{
  background: linear-gradient(135deg,var(--primary),#0b0f1a);
  color:#000;
  align-self:flex-end;
  box-shadow:0 0 18px rgba(23,243,140,0.45);
}

/* ================= FOOTER ================= */
.vai-chatbox .chatbox__footer{
  padding:12px;
  background: linear-gradient(180deg,#ffffff,#f3f3f3);
  display:flex;
  gap:8px;
  align-items:center;
  border-top:1px solid rgba(0,0,0,0.06);
}

.vai-chatbox .chatbox__footer input{
  flex:1;
  padding:10px 14px;
  border-radius:999px;
  border:1px solid #ddd;
  outline:none;
  font-size:14px;
}

.vai-chatbox .chatbox__footer button{
  padding:9px 14px;
  border-radius:10px;
  background: linear-gradient(135deg,var(--primary),#0b0f1a);
  color:#000;
  border:none;
  cursor:pointer;
  box-shadow:0 0 12px rgba(243, 16, 148, 0.84);
}

/* ================= STATUS & LINKS ================= */
.vai-chatbox .status{
  padding:8px 12px;
  color:#777;
  font-size:12px;
  text-align:center;
}

.vai-chatbox a.vai-link{
  color:var(--primary);
  text-decoration:underline;
  word-break:break-all;
}

/* ================= COPYRIGHT ================= */
.vai-chatbox .chatbox__copyright{
  background:#000;
  color:#bbb;
  font-size:11px;
  text-align:center;
  padding:4px 0;
  font-style:italic;
  letter-spacing:0.3px;
}

.vai-chatbox .chatbox__copyright a{
  color:#bbb;
  text-decoration:none;
  transition:color .2s ease;
}

.vai-chatbox .chatbox__copyright a:hover{
  color:#fff;
  text-decoration:underline;
}

.vai-chatbox .chatbox__button i{
  font-size:30px;
  line-height:1;
}
`;

      // create style element and append to shadow root (isolated)
      const styleEl = document.createElement('style');
      styleEl.setAttribute('type','text/css');
      styleEl.appendChild(document.createTextNode(css));
      shadow.appendChild(styleEl);

      // Create widget container inside the shadow root
      const container = document.createElement('div');
      container.className = 'vai-chatbox';
      container.innerHTML = `
        <div class="chatbox__support" id="vai_support" aria-hidden="true">
          <div class="chatbox__header">
            <div class="chatbox__image--header"><img src="${BASE_API}/static/logo.png" alt="VAI" /></div>
            <div class="chatbox__content--header">
              <h4 class="chatbox__heading--header">Noah</h4>
              <p class="chatbox__description--header">Institutional Language Model</p>
            </div>
          </div>
          <div class="chatbox__messages" id="vai_messages" aria-live="polite"></div>
          <div class="chatbox__copyright">
            © 2026 <a href="https://www.cogniaistudios.com" target="_blank" rel="noopener noreferrer">CogniAI Studios</a>. All rights reserved.
          </div>
          <div class="chatbox__footer">
            <input id="vai_input" type="text" placeholder="Write a message..." aria-label="Message" />
            <button id="vai_send" type="button">Send</button>
          </div>
          <div class="status" id="vai_status" style="display:none"></div>
        </div>
        <div class="chatbox__button">
          <button id="vai_toggle" title="Chat with Veronica">
            <i class='bx bxs-message'></i>
          </button>
        </div>
      `;
      // append container to shadow root (so DOM + styles are encapsulated)
      shadow.appendChild(container);

      // --- START: external floating button (kept exactly as you had it) ---
      const originalToggleBtn = shadow.getElementById('vai_toggle');
      if (originalToggleBtn) {
        originalToggleBtn.style.display = 'none';
      }

      const externalToggle = document.createElement('button');
      externalToggle.id = 'vai_external_toggle';
      externalToggle.innerHTML = '💬';
      
      Object.assign(externalToggle.style, {
        position: 'fixed',
        right: '20px',
        bottom: '20px',
        width: '60px',
        height: '60px',
        borderRadius: '50%',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        boxShadow: '0 6px 14px rgba(0,0,0,0.25)',
        border: '0',
      
        /* 🔥 IMAGE BACKGROUND */
        background: '#17f38c',
      
        color: '#fff',
        cursor: 'pointer',
        zIndex: '2147483001',
        fontSize: '22px',
        outline: 'none'
      });
      
      document.body.appendChild(externalToggle);

      // --- END replacement ---

      // Query elements inside shadow root (isolation preserved)
      const support = shadow.getElementById('vai_support');
      const toggle = shadow.getElementById('vai_toggle'); // original toggle inside shadow (hidden)
      const messagesEl = shadow.getElementById('vai_messages');
      const inputEl = shadow.getElementById('vai_input');
      const sendBtn = shadow.getElementById('vai_send');
      const statusEl = shadow.getElementById('vai_status');

      function setStatus(msg, show=true) {
        if (!msg) {
          statusEl.style.display='none';
          statusEl.textContent='';
          return;
        }
        statusEl.style.display = show ? 'block' : 'none';
        statusEl.textContent = msg;
      }

      // toggle behavior (shadow button + external button)
      if (toggle) {
        try {
          toggle.addEventListener('click', () => {
            support.classList.toggle('chatbox--active');
            if (support.classList.contains('chatbox--active')) inputEl.focus();
          });
        } catch(e) { console.warn('Failed to attach listener to original toggle', e); }
      }

      externalToggle.addEventListener('click', () => {
        support.classList.toggle('chatbox--active');
        if (support.classList.contains('chatbox--active')) inputEl.focus();
      });

      function appendMessage(text, who='veronica') {
        const div = document.createElement('div');
        div.className = 'messages__item ' + (who === 'you' ? 'messages__item--operator' : 'messages__item--visitor');
        div.textContent = text;

        messagesEl.insertBefore(div, messagesEl.firstChild);

        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
      }

      function appendHtmlMessage(htmlContent, who='veronica') {
        const div = document.createElement('div');
        div.className = 'messages__item ' + (who === 'you' ? 'messages__item--operator' : 'messages__item--visitor');
        div.innerHTML = htmlContent;
        messagesEl.insertBefore(div, messagesEl.firstChild);
        messagesEl.scrollTop = messagesEl.scrollHeight;
        return div;
      }

      function openUrlInNewTab(url) {
        try {
          const newWin = window.open(url, '_blank', 'noopener,noreferrer');
          if (!newWin) {
            appendHtmlMessage(`<a class="vai-link" href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`, 'veronica');
          }
        } catch (e) {
          console.error('[VERONICA] open url failed', e);
          appendHtmlMessage(`<a class="vai-link" href="${url}" target="_blank" rel="noopener noreferrer">${url}</a>`, 'veronica');
        }
      }

      async function sendMessage() {
        const val = inputEl.value.trim();
        if (!val) return;
        const userDiv = appendMessage(val, 'you');
        inputEl.value = '';
        setStatus('Sending...');
        try {

          const typingDiv = appendMessage('Just a moment...', 'veronica');

          const doFetch = async () => {
            const payload = {
              message: val,
              session_id: SESSION_ID,   // <--- this is what your Redis backend uses
              url: location.href,
              user_agent: navigator.userAgent || '',
              language: navigator.language || '',
              timestamp: (new Date()).toISOString()
            };
            console.log('[VERONICA] POST', BASE_API + '/predict', 'payload:', payload);
            const res = await fetch(BASE_API + '/predict', {
              method: 'POST',
              headers: {'Content-Type':'application/json'},
              body: JSON.stringify(payload),
              keepalive: true
            });
            return res;
          };

          let res = await doFetch();

          if (typingDiv && typingDiv.parentNode) messagesEl.removeChild(typingDiv);

          if (!res.ok) {
            console.warn('[VERONICA] first fetch not ok, status=', res.status);
            setStatus('Veronica is waking up — retrying in a few seconds...', true);
            await new Promise(r=>setTimeout(r, 3500));
            res = await doFetch();
            if (!res.ok) {
              console.error('[VERONICA] retry failed, status=', res.status);
              setStatus(`Server error (${res.status}). Try again later.`);
              return;
            }
          }

          let data;
          try {
            data = await res.json();
          } catch (e) {
            console.error('[VERONICA] failed to parse JSON', e);
            setStatus('Invalid response from server.');
            return;
          }

          if (data.url) {
            appendMessage(data.answer || (`Opening: ${data.url}`), 'veronica');
            openUrlInNewTab(data.url);
            setStatus('');
            if (data.reply_id) console.log('[VERONICA] reply_id:', data.reply_id);
            return;
          }

          const botDiv = appendMessage(data.answer || 'Sorry, no reply', 'veronica');
          if (data.reply_id) botDiv.dataset.replyId = data.reply_id;
          if (data.reply_id) console.log('[VERONICA] reply_id:', data.reply_id);

          setStatus('');
        } catch (err) {
          console.error('Widget fetch error', err);

          if (err instanceof TypeError) {
            setStatus('Network error or CORS blocked. Check console and server CORS settings.');
          } else {
            setStatus('Error connecting to Veronica. Try again later.');
          }
        }
      }

      sendBtn.addEventListener('click', sendMessage);
      inputEl.addEventListener('keyup', (e) => { if (e.key === 'Enter') sendMessage(); });

      // expose API that operates on elements inside shadow root
      window.VERONICA_WIDGET = {
        baseUrl: BASE_API,
        sessionId: SESSION_ID,
        open: () => support.classList.add('chatbox--active'),
        close: () => support.classList.remove('chatbox--active'),
        send: (msg) => { inputEl.value = msg; sendMessage(); }
      };

      console.log('[VERONICA WIDGET] ready (shadow DOM enabled)');
    } catch (e) {
      console.error('[VERONICA WIDGET] init failed', e);
    }
  });
})();

















