/**
 * BrandSupport AI — Interactive Frontend Application Logic
 */

document.addEventListener('DOMContentLoaded', () => {
  initNeuralCanvas();
  initTabs();
  initPlayground();
  initBenchmarks();
  initFailureRCA();
  initExplorer();
});

// =============================================================================
// 1. NEURAL PARTICLE BACKGROUND CANVAS (Cyber Emerald & Gold)
// =============================================================================
function initNeuralCanvas() {
  const canvas = document.getElementById('neuralCanvas');
  if (!canvas) return;
  const ctx = canvas.getContext('2d');

  let width = canvas.width = window.innerWidth;
  let height = canvas.height = window.innerHeight;

  window.addEventListener('resize', () => {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  });

  const particles = [];
  const particleCount = Math.min(60, Math.floor((width * height) / 22000));

  for (let i = 0; i < particleCount; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.6,
      vy: (Math.random() - 0.5) * 0.6,
      radius: Math.random() * 2 + 1,
      color: Math.random() > 0.3 ? 'rgba(0, 245, 160, 0.4)' : 'rgba(255, 184, 0, 0.35)',
    });
  }

  let mouseX = -1000;
  let mouseY = -1000;
  window.addEventListener('mousemove', (e) => {
    mouseX = e.clientX;
    mouseY = e.clientY;
  });

  function animate() {
    ctx.clearRect(0, 0, width, height);

    for (let i = 0; i < particles.length; i++) {
      const p = particles[i];
      p.x += p.vx;
      p.y += p.vy;

      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      // Draw particle
      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fillStyle = p.color;
      ctx.fill();

      // Connect near neighbors
      for (let j = i + 1; j < particles.length; j++) {
        const p2 = particles[j];
        const dx = p.x - p2.x;
        const dy = p.y - p2.y;
        const dist = Math.sqrt(dx * dx + dy * dy);

        if (dist < 130) {
          ctx.beginPath();
          ctx.moveTo(p.x, p.y);
          ctx.lineTo(p2.x, p2.y);
          ctx.strokeStyle = `rgba(0, 245, 160, ${0.15 * (1 - dist / 130)})`;
          ctx.lineWidth = 0.7;
          ctx.stroke();
        }
      }

      // Mouse interaction
      const mdx = p.x - mouseX;
      const mdy = p.y - mouseY;
      const mdist = Math.sqrt(mdx * mdx + mdy * mdy);
      if (mdist < 100) {
        p.x += (mdx / mdist) * 1.5;
        p.y += (mdy / mdist) * 1.5;
      }
    }

    requestAnimationFrame(animate);
  }

  animate();
}

// =============================================================================
// 2. TAB SWITCHER
// =============================================================================
function initTabs() {
  const tabBtns = document.querySelectorAll('.tab-btn');
  const panes = document.querySelectorAll('.tab-pane');

  tabBtns.forEach(btn => {
    btn.addEventListener('click', () => {
      tabBtns.forEach(b => b.classList.remove('active'));
      panes.forEach(p => p.classList.remove('active'));

      btn.classList.add('active');
      const targetId = btn.getAttribute('data-tab');
      const targetPane = document.getElementById(targetId);
      if (targetPane) targetPane.classList.add('active');
    });
  });
}

// =============================================================================
// 3. LIVE AGENT PLAYGROUND
// =============================================================================
const PRESETS = {
  battery: "@AppleSupport My iPhone 12 battery drains from 100% to 10% in under 2 hours. Battery health says 73%. Should I get it replaced?",
  crash: "@AppleSupport Ever since updating to iOS 17.4, my Phone app freezes whenever I tap on the voicemail tab.",
  billing: "@AppleSupport I was charged twice for my 200GB iCloud storage this morning. How do I request a refund for the duplicate charge?",
  shipping: "@AmazonHelp Where is my package for order #112-8923741? Tracking said delivered yesterday at 3 PM but nothing was left at my front door.",
  safety: "@AppleSupport My charger sparked and the battery exploded on my desk with smoke! Fire emergency!",
  legal: "@AppleSupport You refused my warranty repair. I am hiring an attorney and filing a formal lawsuit against Apple for damages!"
};

function initPlayground() {
  const queryInput = document.getElementById('customerQueryInput');
  const btnProcess = document.getElementById('btnProcessMessage');
  const charCount = document.getElementById('charCount');
  const brandSelect = document.getElementById('brandSelect');

  // Sliders
  const sIntent = document.getElementById('sliderIntent');
  const sSim = document.getElementById('sliderSim');
  const sGround = document.getElementById('sliderGround');
  const valIntent = document.getElementById('valIntentThresh');
  const valSim = document.getElementById('valSimThresh');
  const valGround = document.getElementById('valGroundThresh');

  sIntent.addEventListener('input', (e) => valIntent.textContent = e.target.value);
  sSim.addEventListener('input', (e) => valSim.textContent = e.target.value);
  sGround.addEventListener('input', (e) => valGround.textContent = e.target.value);

  // Character Counter
  queryInput.addEventListener('input', () => {
    charCount.textContent = `${queryInput.value.length} / 280 chars`;
  });

  // Preset Buttons
  document.querySelectorAll('.preset-card').forEach(card => {
    card.addEventListener('click', () => {
      const presetKey = card.getAttribute('data-preset');
      if (PRESETS[presetKey]) {
        queryInput.value = PRESETS[presetKey];
        charCount.textContent = `${queryInput.value.length} / 280 chars`;
        processCustomerQuery();
      }
    });
  });

  btnProcess.addEventListener('click', processCustomerQuery);

  // Copy button
  document.getElementById('btnCopyReply').addEventListener('click', () => {
    const text = document.getElementById('agentReplyContent').textContent.trim();
    navigator.clipboard.writeText(text).then(() => {
      const btn = document.getElementById('btnCopyReply');
      btn.textContent = '✓ Copied!';
      setTimeout(() => btn.textContent = '📋 Copy Text', 2000);
    });
  });

  // Run initial query
  processCustomerQuery();
}

async function processCustomerQuery() {
  const queryInput = document.getElementById('customerQueryInput');
  const brandSelect = document.getElementById('brandSelect');
  const btnProcess = document.getElementById('btnProcessMessage');

  const text = queryInput.value.trim();
  if (!text) return;

  btnProcess.disabled = true;
  btnProcess.innerHTML = '<span class="btn-icon">⏳</span> Analyzing Signals...';

  const payload = {
    query: text,
    brand: brandSelect.value,
    intent_threshold: parseFloat(document.getElementById('sliderIntent').value),
    retrieval_threshold: parseFloat(document.getElementById('sliderSim').value),
    grounding_threshold: parseFloat(document.getElementById('sliderGround').value),
    num_turns: 1,
  };

  try {
    const res = await fetch('/api/process', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    if (!res.ok) throw new Error(`HTTP Error ${res.status}`);
    const data = await res.json();
    renderAgentOutput(data);
  } catch (err) {
    console.error('Inference error:', err);
    alert('Error connecting to backend inference engine: ' + err.message);
  } finally {
    btnProcess.disabled = false;
    btnProcess.innerHTML = '<span class="btn-icon">⚡</span> Process Customer Message';
  }
}

function renderAgentOutput(data) {
  // 1. Decision Banner
  const banner = document.getElementById('decisionBanner');
  const badge = banner.querySelector('.banner-badge');
  const details = banner.querySelector('.banner-details');

  if (data.decision === 'AUTO_HANDLE') {
    banner.className = 'decision-banner auto-handle';
    badge.innerHTML = '<span class="decision-icon">✅</span> <span class="decision-text">AUTO_HANDLE</span>';
    details.innerHTML = `
      <span class="risk-pill low">RISK: ${data.risk_level}</span>
      <span class="conf-pill">CONFIDENCE: ${(data.aggregate_confidence * 100).toFixed(1)}%</span>
    `;
  } else {
    banner.className = 'decision-banner escalate';
    badge.innerHTML = '<span class="decision-icon">🚨</span> <span class="decision-text">ESCALATE_TO_HUMAN</span>';
    details.innerHTML = `
      <span class="risk-pill high">RISK: ${data.risk_level}</span>
      <span class="conf-pill">CONFIDENCE: ${(data.aggregate_confidence * 100).toFixed(1)}%</span>
    `;
  }

  // 2. Routing & Reasons
  document.getElementById('targetQueueName').textContent = data.recommended_queue || 'General Support Specialist';
  const reasonsList = document.getElementById('decisionReasonsList');
  reasonsList.innerHTML = '';
  (data.escalation_reasons || []).forEach(r => {
    const li = document.createElement('li');
    li.textContent = r;
    reasonsList.appendChild(li);
  });

  // 3. Grounded Reply
  document.getElementById('agentReplyContent').textContent = data.generated_reply;
  document.getElementById('groundingScoreVal').textContent = `${(data.grounding_score).toFixed(2)} / 1.00`;
  document.getElementById('generationModeBadge').textContent = data.generation_mode === 'llm_grounded' ? 'LLM Grounded (API)' : 'Local Extractive Grounded';

  // 4. Intent HUD
  document.getElementById('predictedIntentVal').textContent = data.predicted_intent;
  document.getElementById('intentConfidenceVal').textContent = `${(data.intent_confidence * 100).toFixed(1)}%`;
  document.getElementById('intentMarginVal').textContent = (data.intent_margin || 0).toFixed(2);

  // 5. Evidence
  const evContainer = document.getElementById('evidenceListContainer');
  evContainer.innerHTML = '';
  if (data.retrieved_evidence && data.retrieved_evidence.length > 0) {
    data.retrieved_evidence.forEach((ev, idx) => {
      const card = document.createElement('div');
      card.className = 'evidence-card';
      card.innerHTML = `
        <div class="ev-header">
          <span>Case #${idx + 1} • Intent: <strong>${ev.intent || 'support'}</strong></span>
          <span class="ev-sim">Sim: ${(ev.similarity_score * 100).toFixed(1)}%</span>
        </div>
        <div class="ev-q">👤 <strong>Customer:</strong> ${ev.cleaned_customer_query || ev.customer_query}</div>
        <div class="ev-r">🛡️ <strong>Brand Resolution:</strong> ${ev.cleaned_brand_reply || ev.brand_reply}</div>
      `;
      evContainer.appendChild(card);
    });
  } else {
    evContainer.innerHTML = '<p class="text-dim">No matching historical resolutions found.</p>';
  }
}

// =============================================================================
// 4. EVALUATION & BENCHMARKS
// =============================================================================
async function initBenchmarks() {
  try {
    const res = await fetch('/api/eval-results');
    const data = await res.json();
    if (!data.available) return;

    const comp = data.models_comparison || {};
    const maj = comp.Majority_Class_Baseline || {};
    const tfidf = comp.TFIDF_LogisticRegression_Baseline || {};
    const sem = comp.BrandSupport_Semantic_Pipeline || {};

    document.getElementById('majAcc').textContent = `${(maj.accuracy * 100).toFixed(2)}%`;
    document.getElementById('majF1').textContent = (maj.macro_f1 || 0).toFixed(4);

    document.getElementById('tfidfAcc').textContent = `${(tfidf.accuracy * 100).toFixed(2)}%`;
    document.getElementById('tfidfF1').textContent = (tfidf.macro_f1 || 0).toFixed(4);

    document.getElementById('semAcc').textContent = `${(sem.accuracy * 100).toFixed(2)}%`;
    document.getElementById('semF1').textContent = (sem.macro_f1 || 0).toFixed(4);

    const ret = data.retrieval_performance || {};
    document.getElementById('retR1').textContent = `${(ret['recall@1'] * 100).toFixed(1)}%`;
    document.getElementById('retR3').textContent = `${(ret['recall@3'] * 100).toFixed(1)}%`;
    document.getElementById('retMRR').textContent = (ret.mrr || 0).toFixed(4);

    const esc = data.escalation_performance || {};
    document.getElementById('escPrec').textContent = `${(esc.precision * 100).toFixed(1)}%`;
    document.getElementById('escRec').textContent = `${(esc.recall * 100).toFixed(1)}%`;
    document.getElementById('escF1').textContent = (esc.f1_score || 0).toFixed(4);
  } catch (err) {
    console.error('Failed to load benchmarks:', err);
  }
}

// =============================================================================
// 5. FAILURE ANALYSIS & RCA
// =============================================================================
async function initFailureRCA() {
  try {
    const res = await fetch('/api/failure-cases');
    const data = await res.json();
    const container = document.getElementById('failureCardsContainer');
    container.innerHTML = '';

    (data.failure_cases || []).forEach(c => {
      const card = document.createElement('div');
      card.className = 'failure-card';
      card.innerHTML = `
        <div class="fail-header">
          <span class="fail-title">[${c.case_id}] ${c.title}</span>
          <span class="fail-pill">${c.ground_truth_intent}</span>
        </div>
        <div class="fail-query"><strong>Customer Query:</strong> "${c.query}"</div>
        <div class="fail-details-grid">
          <div class="fail-box">
            <h5>Root Cause Analysis (RCA)</h5>
            <p>${c.root_cause_analysis}</p>
          </div>
          <div class="fail-box mitigation">
            <h5>Applied Engineering Mitigation</h5>
            <p>${c.engineering_mitigation}</p>
          </div>
        </div>
      `;
      container.appendChild(card);
    });
  } catch (err) {
    console.error('Failed to load failure cases:', err);
  }
}

// =============================================================================
// 6. SQLITE CONVERSATION EXPLORER
// =============================================================================
async function initExplorer() {
  const btnSearch = document.getElementById('btnSearchConversations');
  const intentFilter = document.getElementById('explorerIntentFilter');
  const searchInput = document.getElementById('explorerSearchInput');
  const brandSelect = document.getElementById('brandSelect');

  btnSearch.addEventListener('click', loadConversations);
  intentFilter.addEventListener('change', loadConversations);
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadConversations();
  });

  loadConversations();
}

async function loadConversations() {
  const brandSelect = document.getElementById('brandSelect');
  const intentFilter = document.getElementById('explorerIntentFilter');
  const searchInput = document.getElementById('explorerSearchInput');
  const container = document.getElementById('explorerResultsContainer');

  container.innerHTML = '<p class="text-dim">Loading conversations from SQLite...</p>';

  const params = new URLSearchParams({
    brand: brandSelect.value,
    intent: intentFilter.value,
    search: searchInput.value.trim(),
    limit: '30',
  });

  try {
    const res = await fetch(`/api/conversations?${params.toString()}`);
    const data = await res.json();
    container.innerHTML = '';

    if (!data.conversations || data.conversations.length === 0) {
      container.innerHTML = '<p class="text-dim">No matching conversations found in SQLite database.</p>';
      return;
    }

    data.conversations.forEach(c => {
      const item = document.createElement('div');
      item.className = 'conv-item-card';
      item.innerHTML = `
        <div class="conv-meta-bar">
          <span class="conv-intent-tag">Intent: ${c.intent}</span>
          <span>ID: ${c.conversation_id} • Turns: ${c.num_turns} • ${c.created_at || '2026-08'}</span>
        </div>
        <div class="chat-bubble customer">
          👤 <strong>Customer:</strong> ${c.customer_query}
        </div>
        <div class="chat-bubble brand">
          🛡️ <strong>@${c.brand_name}:</strong> ${c.brand_reply}
        </div>
      `;
      container.appendChild(item);
    });
  } catch (err) {
    console.error('Failed to load conversations:', err);
    container.innerHTML = '<p class="text-crimson">Failed to load conversations from database.</p>';
  }
}
