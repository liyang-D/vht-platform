const $ = (id) => document.getElementById(id);

async function api(path, options = {}) {
  const response = await fetch(path, options);
  if (!response.ok) {
    let detail = response.statusText;
    try { detail = (await response.json()).detail || detail; } catch (_) {}
    throw new Error(`${response.status} ${detail}`);
  }
  return response.json();
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function domainSummary(domains) {
  const row = element('div', 'history-domains');
  for (const [name, value] of Object.entries(domains || {})) {
    const completed = value.completed_questions ?? value.questions?.filter(Boolean).length ?? 0;
    const total = value.total_questions ?? value.questions?.length ?? 0;
    row.append(element('span', '', `${name} ${completed}/${total}`));
  }
  return row;
}

async function activate(sessionId) {
  if (!confirm('Save the current draft and open this session?')) return;
  await api(`api/sessions/history/${sessionId}/activate`, {method: 'POST'});
  location.href = '/';
}

async function showDetail(sessionId) {
  const data = await api(`api/sessions/history/${sessionId}`);
  const root = $('history-detail-content');
  root.replaceChildren();
  root.append(element('h2', '', data.participant || 'Untitled session'));
  root.append(element('p', 'session-meta', `${data.status} · ${data.updated_at} · ${data.session_id}`));

  const summary = element('div', 'detail-summary');
  summary.append(domainSummary(data.domains));
  summary.append(element('p', '', `${data.completed_question_count} / ${data.total_question_count} questions completed`));
  root.append(summary);

  const open = element('button', '', data.status === 'complete' ? 'Reopen and overwrite questions' : 'Continue this session');
  open.onclick = () => activate(data.session_id);
  root.append(open);

  root.append(element('h3', '', 'Question attempts and review'));
  const questions = element('div', 'question-review-list');
  for (const question of data.questions || []) {
    const card = element('article', 'question-review');
    card.append(element('h4', '', `${question.domain} · Question ${Number(question.question_index) + 1}`));
    card.append(element('p', '', question.question_text || ''));
    card.append(element('p', 'score-line', `Score ${question.score} / ${question.score_cap} · ${question.rule_processing?.match_type || question.match_type}`));
    const attempts = element('ol', 'attempt-list');
    for (const attempt of question.attempts || (question.response || []).map((response, index) => ({attempt: index + 1, response}))) {
      attempts.append(element('li', '', `${attempt.response}${attempt.input_metadata ? ` · ${JSON.stringify(attempt.input_metadata)}` : ''}`));
    }
    card.append(attempts);
    const note = document.createElement('textarea');
    note.placeholder = 'Manual review note';
    note.value = question.manual_review_note || '';
    const save = element('button', 'secondary', 'Save note');
    save.onclick = async () => {
      await api(`api/sessions/history/${data.session_id}/review-note`, {
        method: 'PUT', headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({domain: question.domain, question_index: question.question_index, note: note.value}),
      });
      save.textContent = 'Saved';
    };
    card.append(note, save);
    questions.append(card);
  }
  root.append(questions);

  root.append(element('h3', '', `ASR transcripts (${data.asr?.length || 0})`));
  const asr = element('pre', 'artifact-data');
  asr.textContent = JSON.stringify(data.asr || [], null, 2);
  root.append(asr);
  root.append(element('h3', '', 'Raw media'));
  const media = element('pre', 'artifact-data');
  media.textContent = JSON.stringify(data.media || {}, null, 2);
  root.append(media);
  root.append(element('h3', '', `Model responses (${data.model_responses?.length || 0})`));
  const models = element('pre', 'artifact-data');
  models.textContent = JSON.stringify(data.model_responses || [], null, 2);
  root.append(models);
  $('history-detail').showModal();
}

async function loadHistory() {
  const data = await api('api/sessions/history');
  $('history-count').textContent = `${data.sessions.length} sessions`;
  const list = $('history-list');
  list.replaceChildren();
  for (const session of data.sessions) {
    const card = element('article', 'history-session');
    const top = element('div', 'history-session-top');
    const title = element('div');
    title.append(element('h2', '', session.participant || 'Untitled session'));
    title.append(element('p', 'session-meta', `${session.status}${session.is_current ? ' · current' : ''} · ${new Date(session.updated_at).toLocaleString()}`));
    const actions = element('div', 'history-actions');
    const view = element('button', 'secondary', 'View details');
    view.onclick = () => showDetail(session.session_id);
    const resume = element('button', '', session.status === 'complete' ? 'Reopen' : 'Continue');
    resume.onclick = () => activate(session.session_id);
    actions.append(view, resume);
    top.append(title, actions);
    card.append(top, domainSummary(session.domains));
    list.append(card);
  }
}

$('refresh-history').onclick = loadHistory;
$('close-history-detail').onclick = () => $('history-detail').close();
loadHistory().catch((error) => { $('history-count').textContent = error.message; });
