const $ = (id) => document.getElementById(id);
let audioRecorder, videoRecorder, audioChunks = [], videoChunks = [], cameraStream, videoStream;
const items = ["spoon","book","kangaroo","penguin","anchor","camel","harp","rhinoceros","barrel","crown","crocodile","accordion"];

async function api(path, options={}) { const r=await fetch(path,options); if(!r.ok) throw new Error((await r.json()).detail||r.statusText); return r.json(); }
$('start-form').onsubmit = async (event) => { event.preventDefault(); const f=new FormData(event.target); try {
  await api('api/sessions',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({
    patient:{name:f.get('patient_name'),dob:f.get('dob')}, assessor:f.get('assessor'),
    location:{number:f.get('number'),street:f.get('street'),town:f.get('town'),county:f.get('county'),country:f.get('country')},
    current_uk_pm:f.get('uk_pm'),current_us_president:f.get('us_president'),previous_uk_pm:f.get('previous_uk_pm')||null,
    previous_us_president:f.get('previous_us_president')||null,synthesize_audio:!!f.get('synthesize')})});
  $('setup').classList.add('hidden'); $('session').classList.remove('hidden');
} catch(e){alert(e.message)} };
$('resume-form').onsubmit=async event=>{event.preventDefault();try{await api(`api/sessions/${$('resume-id').value.trim()}/resume`,{method:'POST'});$('setup').classList.add('hidden');$('session').classList.remove('hidden')}catch(e){alert(e.message)}};

function show(id, yes){$(id).classList.toggle('hidden',!yes)}
async function poll(){try{const s=await api('api/sessions/current'); $('status-text').textContent=s.status; $('prompt').textContent=s.prompt||'Processing…'; $('error').textContent=s.error||'';
  if(s.audio){$('audio').src=`data:${s.audio.mime_type};base64,${s.audio.data_base64}`}
  show('stimulus',!!s.stimulus); if(s.stimulus)$('stimulus').src=s.stimulus;
  show('audio-controls',s.awaiting==='audio'); show('image-controls',s.awaiting==='image'); show('video-controls',s.awaiting==='video'); show('click-controls',s.awaiting==='click');
  if(s.awaiting==='image'&&!cameraStream) cameraStream=await navigator.mediaDevices.getUserMedia({video:true}); if(cameraStream)$('camera').srcObject=cameraStream;
  if(s.awaiting==='video'&&!videoStream) videoStream=await navigator.mediaDevices.getUserMedia({video:true,audio:false}); if(videoStream)$('video-preview').srcObject=videoStream;
}catch(e){} setTimeout(poll,900)} poll();

$('record').onclick=async()=>{const stream=await navigator.mediaDevices.getUserMedia({audio:true}); audioChunks=[]; audioRecorder=new MediaRecorder(stream); audioRecorder.ondataavailable=e=>audioChunks.push(e.data); audioRecorder.start(); $('record').disabled=true;$('stop').disabled=false};
$('stop').onclick=()=>{audioRecorder.onstop=async()=>{const blob=new Blob(audioChunks,{type:audioRecorder.mimeType});const f=new FormData();f.append('audio',blob,'answer.webm');await api('api/answers/audio',{method:'POST',body:f});audioRecorder.stream.getTracks().forEach(t=>t.stop())};audioRecorder.stop();$('record').disabled=false;$('stop').disabled=true};
$('capture').onclick=async()=>{const v=$('camera'),c=document.createElement('canvas');c.width=v.videoWidth;c.height=v.videoHeight;c.getContext('2d').drawImage(v,0,0);c.toBlob(async blob=>{const f=new FormData();f.append('kind','image');f.append('media',blob,'capture.png');await api('api/answers/media',{method:'POST',body:f})},'image/png')};
$('record-video').onclick=()=>{videoChunks=[];videoRecorder=new MediaRecorder(videoStream);videoRecorder.ondataavailable=e=>videoChunks.push(e.data);videoRecorder.start();$('record-video').disabled=true;$('stop-video').disabled=false};
$('stop-video').onclick=()=>{videoRecorder.onstop=async()=>{const f=new FormData();f.append('kind','video');f.append('media',new Blob(videoChunks,{type:videoRecorder.mimeType}),'action.webm');await api('api/answers/media',{method:'POST',body:f})};videoRecorder.stop();$('record-video').disabled=false;$('stop-video').disabled=true};
items.forEach(name=>{const b=document.createElement('button');b.textContent=name;b.onclick=()=>api('api/answers/click',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:name})});$('click-controls').appendChild(b)});
$('text-form').onsubmit=async e=>{e.preventDefault();await api('api/answers/text',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text:$('text-answer').value})});$('text-answer').value=''};
$('pause').onclick=async()=>{try{const r=await api('api/sessions/pause',{method:'POST'});alert(`Saved ${r.checkpoint}`)}catch(e){alert(e.message)}};
