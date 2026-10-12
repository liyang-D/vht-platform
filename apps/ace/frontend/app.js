const $ = (id) => document.getElementById(id);
let audioRecorder, videoRecorder, audioChunks = [], videoChunks = [], cameraStream, videoStream;
let currentAudioId = null;
let currentStimulus = null;
let lastRuntimeState = '';
let activeSessionId = null;
let navigationSignature = '';
let clickSubmitTimer = null;
let interactionSignature = '';
const audioClientId=typeof crypto.randomUUID==='function'?crypto.randomUUID():`${Date.now()}-${Math.random()}`;
const playbackChannel=typeof BroadcastChannel==='function'?new BroadcastChannel('ace-tts-playback'):null;
const items = ["spoon","book","kangaroo","penguin","anchor","camel","harp","rhinoceros","barrel","crown","crocodile","accordion"];

function debug(message,detail){const output=$('debug-console');const suffix=detail===undefined?'':` ${typeof detail==='string'?detail:JSON.stringify(detail)}`;output.textContent+=`[${new Date().toLocaleTimeString()}] ${message}${suffix}\n`;const lines=output.textContent.split('\n');if(lines.length>201)output.textContent=lines.slice(-201).join('\n');output.scrollTop=output.scrollHeight}
$('clear-console').onclick=()=>{$('debug-console').textContent='';debug('Console cleared')};
async function api(path, options={}) { const r=await fetch(path,options); if(!r.ok){let detail=r.statusText;try{detail=(await r.json()).detail||detail}catch(e){}throw new Error(`${r.status} ${detail}`)} return r.json(); }
async function loadDefaults(){try{const config=await api('api/config');for(const [name,value] of Object.entries(config.leader_defaults||{})){const input=document.querySelector(`[name="${name}"]`);if(input&&!input.value)input.value=value}}catch(e){}}
debug('ACE page loaded');loadDefaults().then(()=>debug('System defaults loaded'));
$('start-form').onsubmit = async (event) => { event.preventDefault(); const f=new FormData(event.target); try {
  await api('api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    patient:{name:f.get('patient_name'),dob:f.get('dob')}, assessor:f.get('assessor'),
    location:{number:f.get('number'),street:f.get('street'),town:f.get('town'),county:f.get('county'),country:f.get('country')},
    current_uk_pm:f.get('uk_pm'),current_us_president:f.get('us_president'),previous_uk_pm:f.get('previous_uk_pm')||null,
    previous_us_president:f.get('previous_us_president')||null,synthesize_audio:!!f.get('synthesize')})});
  currentAudioId=null;$('audio').removeAttribute('src');$('setup').classList.add('hidden'); $('assessment-layout').classList.remove('hidden');debug('Session started');
} catch(e){debug('Session start failed',e.message);alert(e.message)} };

function show(id, yes){$(id).classList.toggle('hidden',!yes)}
function stopTtsPlayback(reason){const player=$('audio');if(!player.paused||player.getAttribute('src')){player.pause();player.removeAttribute('src');player.load();debug('TTS playback stopped',{reason})}}
playbackChannel?.addEventListener('message',event=>{if(event.data?.client_id!==audioClientId)stopTtsPlayback('another ACE page took playback')});
window.addEventListener('storage',event=>{if(event.key==='ace-active-audio-client'&&event.newValue!==audioClientId)stopTtsPlayback('another ACE page took playback')});
document.addEventListener('visibilitychange',()=>{if(document.hidden)stopTtsPlayback('page hidden')});
async function playTtsOnce(payload){
  const markerKey='ace-last-played-audio-id';
  const play=async()=>{
    if(localStorage.getItem(markerKey)===payload.id){debug('Duplicate TTS playback suppressed',{id:payload.id});return}
    localStorage.setItem(markerKey,payload.id);
    localStorage.setItem('ace-active-audio-client',audioClientId);
    playbackChannel?.postMessage({client_id:audioClientId,audio_id:payload.id});
    const player=$('audio');
    player.pause();player.removeAttribute('src');player.load();
    player.src=`data:${payload.mime_type};base64,${payload.data_base64}`;
    try{await player.play()}catch(e){localStorage.removeItem(markerKey);debug('TTS autoplay blocked',e.message)}
  };
  if(navigator.locks?.request){
    await navigator.locks.request('ace-tts-playback',play);
  }else{
    await play();
  }
}
function renderNavigation(navigation,currentDomain){const domains=navigation?.domains||[];const signature=JSON.stringify([navigation?.can_jump,currentDomain,domains]);if(signature===navigationSignature)return;navigationSignature=signature;const list=$('domain-list');list.replaceChildren();for(const domain of domains){const item=document.createElement('li');const button=document.createElement('button');const name=document.createElement('span');const count=document.createElement('small');button.type='button';button.className=`domain-link ${domain.status}`;button.disabled=!navigation.can_jump;button.setAttribute('aria-current',domain.name===currentDomain?'step':'false');name.textContent=domain.name;count.textContent=`${domain.completed}/${domain.total}`;button.append(name,count);if(navigation.can_jump)button.onclick=async()=>{if(domain.status==='complete'&&!confirm(`Restart ${domain.name}? Existing results for this module will be replaced.`))return;try{debug('Module jump requested',{domain:domain.name});await api('api/sessions/jump',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({domain:domain.name})})}catch(e){debug('Module jump failed',e.message);$('error').textContent=e.message}};item.append(button);list.append(item)}$('navigation-hint').textContent=navigation?.can_jump?'Debug navigation enabled. Select a module to jump to it.':'Progress is view only for this session.'}
const renderDomainDirectory=renderNavigation;renderNavigation=(navigation,currentDomain)=>{$('previous-question').disabled=!navigation?.can_jump;$('next-question').disabled=!navigation?.can_jump;return renderDomainDirectory(navigation,currentDomain)};
async function jumpQuestion(direction){try{debug('Question jump requested',{direction});const result=await api('api/sessions/jump-question',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({direction})});debug('Question jump accepted',result)}catch(e){debug('Question jump failed',e.message);$('error').textContent=e.message}}
$('previous-question').onclick=()=>jumpQuestion('previous');$('next-question').onclick=()=>jumpQuestion('next');
async function poll(){try{const audioParams=new URLSearchParams();if(currentAudioId)audioParams.set('known_audio_id',currentAudioId);if(!document.hidden)audioParams.set('audio_client_id',audioClientId);const audioQuery=audioParams.size?`?${audioParams}`:'';const s=await api(`api/sessions/current${audioQuery}`);if(s.session_id!==activeSessionId){activeSessionId=s.session_id;$('setup').classList.add('hidden');$('assessment-layout').classList.remove('hidden');debug('Attached to session',{session_id:s.session_id})}$('status-text').textContent=s.status; $('prompt').textContent=s.prompt||'Processing…'; $('error').textContent=s.error||'';const progress=s.progress||{};const total=Number(progress.total)||0;const current=Number(progress.current)||0;const percent=total?Math.min(100,Math.round(current/total*100)):0;$('progress-label').textContent=total?`${current} of ${total} completed`:'Preparing…';$('progress-bar').style.width=`${percent}%`;$('progress-track').setAttribute('aria-valuenow',String(percent));renderNavigation(s.navigation,progress.domain);const runtimeState=`${s.status}:${s.awaiting||'-'}`;if(runtimeState!==lastRuntimeState){lastRuntimeState=runtimeState;debug('Runtime state',{status:s.status,awaiting:s.awaiting})}
  if(s.audio&&s.audio.id!==currentAudioId){currentAudioId=s.audio.id;debug('New TTS audio',{id:s.audio.id,mime_type:s.audio.mime_type,source:s.audio.source,base64_chars:s.audio.data_base64.length});void playTtsOnce(s.audio)}
  show('stimulus-stage',!!s.stimulus);if(s.stimulus!==currentStimulus){currentStimulus=s.stimulus;if(s.stimulus){$('stimulus').src=s.stimulus;debug('Stimulus changed',{src:s.stimulus})}else{$('stimulus').removeAttribute('src')}}show('click-overlay',s.awaiting==='click');
  const interaction=s.interaction||null;const nextInteractionSignature=JSON.stringify(interaction);if(nextInteractionSignature!==interactionSignature){interactionSignature=nextInteractionSignature;debug('Interaction step changed',interaction||{kind:'none'})}const naming=interaction?.kind==='picture_naming';show('naming-overlay',naming);if(naming){for(const [index,cell] of [...$('naming-overlay').children].entries())cell.classList.toggle('active',index===interaction.active_grid_index)}
  show('audio-controls',s.awaiting==='audio'); show('image-controls',s.awaiting==='image'); show('video-controls',s.awaiting==='video');
  $('capture').textContent=interaction?.kind==='clock'&&interaction.phase==='clock_face'?'Clock face complete':interaction?.kind==='clock'&&interaction.phase==='clock_hands'?'Capture final clock':interaction?.kind==='writing'&&interaction.phase==='one_sentence_followup'?'Capture updated writing':'Capture drawing';
  if(s.awaiting==='image'&&!cameraStream){cameraStream=await navigator.mediaDevices.getUserMedia({video:true});$('camera').srcObject=cameraStream;debug('Camera preview started')}
  else if(cameraStream&&$('camera').srcObject!==cameraStream){$('camera').srcObject=cameraStream}
  if(s.awaiting==='video'&&!videoStream){videoStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false});$('video-preview').srcObject=videoStream;debug('Video preview started')}
  else if(videoStream&&$('video-preview').srcObject!==videoStream){$('video-preview').srcObject=videoStream}
}catch(e){if(!e.message.startsWith('404 '))debug('Status poll failed',e.message)} setTimeout(poll,900)} poll();

$('audio').onplay=()=>debug('TTS playback started');$('audio').onended=()=>debug('TTS playback ended');$('audio').onerror=()=>debug('TTS playback error',$('audio').error?.message||'unknown media error');
$('record').onclick=async()=>{try{const stream=await navigator.mediaDevices.getUserMedia({audio:true});audioChunks=[];const preferred='audio/webm;codecs=opus';const options=MediaRecorder.isTypeSupported(preferred)?{mimeType:preferred}:undefined;audioRecorder=new MediaRecorder(stream,options);audioRecorder.ondataavailable=e=>{if(e.data.size)audioChunks.push(e.data)};audioRecorder.start();debug('Recording started',{mime_type:audioRecorder.mimeType});$('record').disabled=true;$('stop').disabled=false}catch(e){debug('Microphone start failed',e.message)}};
$('stop').onclick=()=>{debug('Recording stop requested');$('stop').disabled=true;audioRecorder.onstop=async()=>{const blob=new Blob(audioChunks,{type:audioRecorder.mimeType});debug('Recording captured',{mime_type:blob.type,size_bytes:blob.size,chunks:audioChunks.length});const f=new FormData();f.append('audio',blob,'answer.webm');try{debug('Uploading audio');const result=await api('api/answers/audio',{method:'POST',body:f});debug('ASR accepted',{transcript:result.transcript})}catch(e){debug('Audio upload/ASR failed',e.message);$('error').textContent=e.message}finally{audioRecorder.stream.getTracks().forEach(t=>t.stop());$('record').disabled=false}};audioRecorder.stop()};
$('capture').onclick=async()=>{const v=$('camera'),c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(async blob=>{const f=new FormData();f.append('kind','image');f.append('media',blob,'capture.png');await api('api/answers/media',{method:'POST',body:f})},'image/png')};
$('record-video').onclick=()=>{videoChunks=[];videoRecorder=new MediaRecorder(videoStream);videoRecorder.ondataavailable=e=>videoChunks.push(e.data);videoRecorder.start();$('record-video').disabled=true;$('stop-video').disabled=false};
$('stop-video').onclick=()=>{videoRecorder.onstop=async()=>{const f=new FormData();f.append('kind','video');f.append('media',new Blob(videoChunks,{type:videoRecorder.mimeType}),'action.webm');await api('api/answers/media',{method:'POST',body:f})};videoRecorder.stop();$('record-video').disabled=false;$('stop-video').disabled=true};
items.forEach((name,gridIndex)=>{const b=document.createElement('button');b.type='button';b.setAttribute('aria-label',`Image position ${gridIndex+1}`);b.onclick=()=>{for(const cell of $('click-overlay').children)cell.classList.remove('selected');b.classList.add('selected');clearTimeout(clickSubmitTimer);debug('Pointing selection changed',{grid_number:gridIndex+1});clickSubmitTimer=setTimeout(async()=>{try{await api('api/answers/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({grid_index:gridIndex,item:name})});debug('Pointing selection submitted',{grid_number:gridIndex+1,item:name});b.classList.remove('selected')}catch(e){debug('Pointing selection failed',e.message)}},1200)};$('click-overlay').appendChild(b)});
items.forEach((_,gridIndex)=>{const cell=document.createElement('span');cell.setAttribute('aria-label',`Picture position ${gridIndex+1}`);$('naming-overlay').appendChild(cell)});
$('text-form').onsubmit=async e=>{e.preventDefault();await api('api/answers/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('text-answer').value})});$('text-answer').value=''};
$('pause').onclick=async()=>{try{await api('api/sessions/pause',{method:'POST'});debug('Draft saved');alert('Draft saved. Unsubmitted current input was not scored.')}catch(e){alert(e.message)}};
function showPartialReport(report){const domains=Object.entries(report.domains||{}).map(([name,value])=>`<tr><th>${name}</th><td>${value.earned_score} / ${value.attempted_available_score}</td><td>${value.completed_questions} / ${value.total_questions}</td><td>${value.unanswered_question_indices.length}</td></tr>`).join('');$('report-content').innerHTML=`<h2>Partial report</h2><p class="report-warning">${report.score_notice}</p><dl><dt>Completed questions</dt><dd>${report.completed_question_count} / ${report.total_question_count}</dd><dt>Score on completed questions</dt><dd>${report.earned_score_on_completed_questions} / ${report.available_score_on_completed_questions}</dd></dl><table><thead><tr><th>Domain</th><th>Earned / available attempted</th><th>Questions done</th><th>Unanswered</th></tr></thead><tbody>${domains}</tbody></table>`;$('report-dialog').showModal()}
$('finish-partial').onclick=async()=>{if(!confirm('End this test and generate a partial report? The current unsubmitted response will not be scored.'))return;try{const result=await api('api/sessions/finish-partial',{method:'POST'});showPartialReport(result.report);debug('Partial report generated',{completed:result.report.completed_question_count})}catch(e){debug('Partial report failed',e.message);$('error').textContent=e.message}};
$('close-report').onclick=()=>$('report-dialog').close();
