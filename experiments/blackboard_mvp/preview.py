"""Write a portable, offline preview of per-board visual observations.

The page only references local, relative image files.  It does not run OCR,
infer teaching order, or join observations from different times into one image.
"""

from __future__ import annotations

import json
from pathlib import Path


def write_preview(manifest: dict, output_dir: Path) -> Path:
    """Embed *manifest* into ``review.html`` beside its local image files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    # Escape HTML delimiters, including a possible </script> in a label.  JSON
    # remains data: the browser reads textContent and never evaluates it.
    payload = json.dumps(manifest, ensure_ascii=False, allow_nan=False)
    for value, escaped in (
        ("&", "\\u0026"),
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        payload = payload.replace(value, escaped)
    target = output_dir / "review.html"
    target.write_text(_HTML.replace("__MANIFEST_JSON__", payload), encoding="utf-8")
    return target


_HTML = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src 'self' file:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; base-uri 'none'; form-action 'none'; connect-src 'none'">
<title>多黑板视觉变化 · 离线预览</title>
<style>
:root{color-scheme:light;--ink:#15202b;--muted:#536575;--line:#d9e1e8;--accent:#005d83;--warn:#7b4510}
*{box-sizing:border-box}body{margin:0;background:#f3f6f8;color:var(--ink);font:15px/1.55 system-ui,-apple-system,sans-serif}
header,main{max-width:1500px;margin:auto;padding:20px}header{padding-bottom:0}h1{font-size:24px;margin:0 0 8px}h2{font-size:19px;margin:0 0 12px}h3{font-size:16px;margin:0 0 8px}p{margin:6px 0 12px}.muted,small{color:var(--muted)}
.notice{padding:12px 15px;border-left:4px solid var(--accent);background:#e8f2f7}.warning{background:#fff4de;border-color:#df9b2b;color:var(--warn)}
.layout{display:grid;grid-template-columns:minmax(0,1.6fr) minmax(310px,1fr);gap:18px}.card{background:white;border:1px solid var(--line);border-radius:10px;padding:16px;margin-bottom:18px;min-width:0}
.controls{display:flex;gap:8px;align-items:center;flex-wrap:wrap;margin:10px 0}.controls label{font-weight:600}button,select{font:inherit;border:1px solid #b8c7d1;border-radius:5px;background:white;color:var(--ink);padding:6px 9px;max-width:100%}button{cursor:pointer}button:hover{background:#e9f2f6}button:disabled{cursor:default;opacity:.5}button:focus-visible,select:focus-visible{outline:3px solid #2a98c9;outline-offset:2px}
.image-box{position:relative;background:#101820;overflow:hidden;border-radius:5px}.image-box img{display:block;width:100%;height:auto}.overlay{position:absolute;inset:0;width:100%;height:100%;pointer-events:none}.caption{font-size:13px;overflow-wrap:anywhere;margin:6px 0}.badge{display:inline-block;background:#eaf0f4;border-radius:4px;padding:2px 6px;font-size:12px;margin:2px}.badge.warn{color:var(--warn);background:#fff0d2}
.assembly-group{grid-column:1/-1;border:1px solid var(--line);border-radius:7px;padding:10px;background:#f7fafc}.assembly-group>.board-grid{margin-top:8px}.position-table{border-collapse:collapse;width:100%;font-size:13px}.position-table td,.position-table th{padding:6px;border:1px solid var(--line);text-align:left;overflow-wrap:anywhere}.scroll{overflow:auto;max-height:420px}.board-grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}.board-tile{border:1px solid var(--line);border-radius:6px;padding:9px;min-width:0}.board-tile h3{overflow-wrap:anywhere}.board-tile .image-box{margin-top:6px}.roi-invalid{border:2px solid #ce7415}.timeline{max-height:690px;overflow:auto;padding:2px}.event{display:block;width:100%;text-align:left;margin:0 0 9px;border-left:4px solid #3785a7;padding:10px}.event.active{background:#e8f4fc;border-color:#005d83}.event-title{display:block;font-weight:700;overflow-wrap:anywhere}.event-detail{display:block;color:var(--muted);font-size:13px}.event .badge{white-space:normal}
.evidence-grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}.list{margin:5px 0 12px;padding-left:22px}.empty{padding:20px;border:1px dashed var(--line);color:var(--muted);text-align:center}.facts{display:grid;grid-template-columns:minmax(90px,auto) 1fr;gap:5px 12px;margin:12px 0;font-size:14px}.facts dt{color:var(--muted)}.facts dd{margin:0;overflow-wrap:anywhere}details{margin:10px 0}summary{cursor:pointer;font-weight:600}.history{max-height:300px;overflow:auto}.history button{display:block;text-align:left;width:100%;margin-bottom:6px}.history button.active{background:#e8f4fc}.wide{grid-column:1/-1}#error{display:none}.legend{display:flex;gap:8px;flex-wrap:wrap}.legend span{font-size:13px}.source-button{margin-top:6px;font-size:13px}a{color:var(--accent)}
@media(max-width:900px){.layout{grid-template-columns:1fr}header,main{padding:12px}.evidence-grid{grid-template-columns:1fr}.timeline{max-height:400px}}
</style>
</head>
<body>
<header>
<h1>多黑板视觉变化 · 离线预览</h1>
<p class="notice">这里只记录采样画面的观察顺序。board_id 指物理板身份，屏幕位置可随机械联动改变。位置标签用于定位，不代表书写、讲解或证明顺序；画面变化也不等于已确认书写。未接入语音分析或公式识别。</p>
<p id="summary" class="muted"></p>
<details><summary>源媒体探测元信息</summary><dl id="source-metadata" class="facts"></dl></details>
<p id="error" class="notice warning" role="alert"></p>
</header>
<main class="layout">
<section class="card" aria-labelledby="sample-title">
<h2 id="sample-title">全部采样帧与独立黑板区域</h2>
<p class="muted">全部采样帧均保留。事件筛选仅改变引用列表，不删除帧；每次展示一个实际时刻的原画面。</p>
<div class="controls"><button id="previous">上一帧</button><label for="frame-select">采样帧</label><select id="frame-select"></select><button id="next">下一帧</button></div>
<div id="camera-state" class="notice"></div><div id="original-frame"></div><p id="frame-caption" class="caption"></p>
<div id="roi-warning"></div><div id="board-legend" class="legend"></div>
<div id="board-crops" class="board-grid"></div>
</section>
<section class="card" aria-labelledby="timeline-title">
<h2 id="timeline-title">全局观察时间轴</h2>
<p class="muted">各板事件按观察时间汇总。同一采样区间内的并列变化保留区间和不确定性，列表位置不构成内部先后证据。</p>
<div class="controls"><label for="event-board">筛选黑板</label><select id="event-board"></select><label for="event-kind">事件类型</label><select id="event-kind"><option value="">全部事件</option><option value="motion">机械移动</option><option value="content">板书变化候选</option><option value="visibility">遮挡 / 重现</option><option value="identity">身份待复核</option></select><span id="event-count" class="muted"></span></div>
<div id="timeline" class="timeline"></div>
</section>
<section class="card wide" aria-labelledby="evidence-title">
<h2 id="evidence-title">事件前后证据</h2><div id="evidence"></div>
</section>
<section class="card wide" aria-labelledby="position-title">
<h2 id="position-title">物理黑板位置与遮挡完整历史</h2>
<p class="muted">board_id 是物理板身份，不是固定屏幕位置。外层板在前、内层板在后；反向联动只是运动先验，不能单独证明身份。下方保留全部观察，包括完全遮挡、移动失败及重新出现；身份不确定需人工确认。</p>
<div class="controls"><label for="position-board">物理黑板</label><select id="position-board"></select></div>
<div id="positions" class="history"></div><div id="position-evidence"></div>
</section>
<section class="card" aria-labelledby="version-title">
<h2 id="version-title">单板观察版本</h2>
<p class="muted">换板不删除旧版本。起始画面已有内容仅标为“起始时已存在”；持续可见的旧内容不会标为刚出现。</p>
<div class="controls"><label for="version-board">黑板</label><select id="version-board"></select></div>
<div id="versions" class="history"></div><div id="version-evidence"></div>
</section>
<section class="card" aria-labelledby="runs-title">
<h2 id="runs-title">连续视觉变化候选段</h2>
<p class="muted">同板相邻变化的辅助索引，可与其他板交错。它不表示老师一直在写这块板，也不确定完整证明顺序。</p>
<div id="runs"></div>
</section>
</main>
<script id="manifest-data" type="application/json">__MANIFEST_JSON__</script>
<script>
'use strict';
(() => {
  const $ = id => document.getElementById(id);
  const el = (tag, text, className) => { const n = document.createElement(tag); if (text !== undefined && text !== null) n.textContent = String(text); if (className) n.className = className; return n; };
  const append = (parent, ...nodes) => { nodes.forEach(n => { if (n) parent.appendChild(n); }); return parent; };
  const number = value => Number.isFinite(Number(value)) ? Number(value) : 0;
  const time = value => { const ms = Math.max(0, number(value)); const s = Math.floor(ms / 1000); return `${String(Math.floor(s / 3600)).padStart(2,'0')}:${String(Math.floor(s / 60) % 60).padStart(2,'0')}:${String(s % 60).padStart(2,'0')}.${String(Math.floor(ms % 1000)).padStart(3,'0')}`; };
  const interval = e => Array.isArray(e.interval_ms) ? `${time(e.interval_ms[0])} – ${time(e.interval_ms[1])}` : time(e.observed_at_ms);
  // Never allow a manifest to turn an image reference into a network request.
  const localPath = value => { if (typeof value !== 'string' || !value) return null; let p; try { p = decodeURIComponent(value); } catch (_) { return null; } if (/[:\\?#\u0000-\u001f]/.test(p) || p.startsWith('/') || p.split('/').some(s => s === '..' || s === '.')) return null; return value; };
  let data;
  try { data = JSON.parse($('manifest-data').textContent); } catch (_) { $('error').textContent = '无法读取预览数据。'; $('error').style.display = 'block'; return; }
  const boards = Array.isArray(data.config?.boards) ? data.config.boards : [];
  const frames = Array.isArray(data.frames) ? data.frames : [];
  const assemblies = Array.isArray(data.config?.assemblies) ? data.config.assemblies : [];
  const mechanical = assemblies.length > 0;
  const events = (Array.isArray(data.events) ? data.events : []).map((e,i) => ({...e, _index:i})).sort((a,b) => number(a.observed_at_ms) - number(b.observed_at_ms) || a._index - b._index);
  const boardById = new Map(boards.map((b,i) => [b.board_id,{...b,_color:['#00c5ff','#ffca43','#ad93ff','#72e298','#ff93ba','#e8a468'][i%6]}]));
  const frameById = new Map(frames.map((f,i) => [f.frame_id,{...f,_index:i}]));
  const kindName = {initial_observation:'起始时已存在',visual_change_candidate:'视觉变化候选',roi_needs_reconfirmation:'区域身份需重新确认',board_motion_event:'机械移动事件',board_content_change_event:'板书变化候选',visibility_event:'遮挡 / 重现观察',identity_review_event:'物理身份待复核',geometry_review_event:'相机/场景几何待复核（身份单独记录）',identity_lost:'身份关联丢失',identity_recovered:'身份重新确认',content_baseline_initialized:'新的几何基线（非新板书）'};
  let frameIndex = 0;
  let selectedEvent = null;
  const boardName = id => { const b = boardById.get(id); return b ? `${b.board_id} · ${b.label || '未标注位置'}` : String(id ?? '未知黑板'); };
  const visibilityName = value => ({visible:'可见',partially_occluded:'部分遮挡',occluded:'完全遮挡',identity_unconfirmed:'身份未确认'}[value] || value || '未知');
  const identityUncertain = state => state?.identity_status === 'identity_uncertain' || (!state?.identity_contract && state?.roi_status === 'needs_reconfirmation');
  const visibilityUncertain = state => identityUncertain(state) || ['provisional_geometry_only','unknown_geometry'].includes(state?.visibility_status);
  const visibleLabel = state => visibilityUncertain(state) ? '估计可见区域，未确认可比较' : '机械几何可见区域（未进行人体分割）';
  const priorText = prior => prior ? `${prior.status ?? '未知'}；反向位移残差 ${prior.residual_pixels ?? '未知'} px` : '未提供';
  const ratio = state => state?.visible_fraction == null ? '未知' : `${(Math.max(0,Math.min(1,number(state.visible_fraction)))*100).toFixed(1)}%`;
  const position = value => Array.isArray(value) ? `[${value.join(', ')}]` : '未知';
  const boardState = (frameId, boardId) => (frameById.get(frameId)?.boards || []).find(b=>b.board_id===boardId);
  const category = e => e.kind === 'board_motion_event' ? 'motion' : ['visual_change_candidate','board_content_change_event'].includes(e.kind) ? 'content' : ['visibility_event','visibility_restored_candidate'].includes(e.kind) ? 'visibility' : ['roi_needs_reconfirmation','identity_review_event','identity_lost','identity_recovered'].includes(e.kind) ? 'identity' : 'other';
  const badge = (text, warn=false) => el('span',text,warn?'badge warn':'badge');
  const empty = text => el('div',text,'empty');
  const option = (parent,value,label) => { const n = el('option',label); n.value = value; parent.appendChild(n); };
  const imageBox = (path, alt, rects=[], size=null) => {
    const box = el('div',null,'image-box'); const safe = localPath(path);
    if (!safe) { box.appendChild(empty(path ? '图片路径不安全，已阻止加载' : '无对应画面（起始观察没有前帧）')); return box; }
    const img = el('img'); img.src = safe; img.alt = alt; img.loading = 'lazy'; box.appendChild(img);
    img.addEventListener('error',() => { box.replaceChildren(empty('无法读取本地图片，请保持图片目录与预览页的相对位置')); });
    if (rects.length) {
      const svg = document.createElementNS('http://www.w3.org/2000/svg','svg'); svg.classList.add('overlay'); svg.setAttribute('aria-hidden','true');
      const draw = () => {
        const w = number(size?.[0]) || img.naturalWidth; const h = number(size?.[1]) || img.naturalHeight;
        if (!w || !h) return; svg.setAttribute('viewBox',`0 0 ${w} ${h}`); svg.replaceChildren();
        rects.forEach(r => {
          if (!Array.isArray(r.bbox) || r.bbox.length !== 4) return;
          const [x,y,rw,rh] = r.bbox.map(number); const rect = document.createElementNS(svg.namespaceURI,'rect');
          Object.entries({x,y,width:rw,height:rh,fill:'none',stroke:r.color || '#ffe16a','stroke-width':Math.max(w/400,2),'vector-effect':'non-scaling-stroke'}).forEach(([k,v]) => rect.setAttribute(k,String(v)));
          if (r.invalid) rect.setAttribute('stroke-dasharray','10 5'); svg.appendChild(rect);
          if (r.label) {
            const text = document.createElementNS(svg.namespaceURI,'text'); text.textContent = String(r.label);
            Object.entries({x:x+5,y:Math.max(y+20,20),fill:r.color || '#ffe16a','font-size':Math.max(w/65,14),'font-family':'system-ui, sans-serif',stroke:'#101820','stroke-width':3,'paint-order':'stroke'}).forEach(([k,v]) => text.setAttribute(k,String(v))); svg.appendChild(text);
          }
        });
      };
      img.addEventListener('load',draw); box.appendChild(svg); if (img.complete) draw();
    }
    return box;
  };
  const sourceButton = (id) => {
    const f = frameById.get(id); if (!f) return el('p',id ? `源帧 ${id}（不可用）` : '无前置源帧','caption');
    const b = el('button',`查看原帧 ${id} · ${time(f.timestamp_ms)}`,'source-button'); b.type='button'; b.addEventListener('click',() => { showFrame(f._index); $('sample-title').scrollIntoView({behavior:'smooth',block:'start'}); }); return b;
  };
  const fact = (list,label,value) => { append(list,el('dt',label),el('dd',value ?? '未知')); };
  const strings = (container,title,values) => {
    container.appendChild(el('h3',title));
    if (!Array.isArray(values) || !values.length) { container.appendChild(el('p','未提供','muted')); return; }
    const ul = el('ul',null,'list'); values.forEach(v => ul.appendChild(el('li', typeof v === 'string' ? v : JSON.stringify(v)))); container.appendChild(ul);
  };
  function boardEvidence(state, label, changeRegions=[]) {
    const panel=el('div');
    if (visibilityUncertain(state)) panel.appendChild(el('p','估计可见区域，未确认可比较；身份或机械几何待人工确认。','notice warning'));
    if (state?.visibility === 'occluded') {
      panel.appendChild(empty(visibilityUncertain(state) ? '按待确认几何估计为完全遮挡 / 出框；暂停内容比较，需人工核实。' : '此物理板在机械几何模型中完全遮挡 / 出框；没有可见板书可比较。当前屏幕中的前层板不是该内层板的新内容。'));
    } else {
      panel.appendChild(imageBox(state?.crop_path, label,changeRegions));
      if (state?.mask_path) {
        const details=el('details'); details.appendChild(el('summary',`${visibleLabel(state)}遮罩（白色仅几何可见候选，黑色几何不可见 / 不可用）`));details.appendChild(el('p','此遮罩只依据机械板面、窗口与前后层几何；没有识别老师身体、手或其他临时遮挡。几何可见不等于已确认可比较。','caption'));
        details.appendChild(imageBox(state.mask_path,`${label} 可见区域遮罩`)); panel.appendChild(details);
      }
    }
    return panel;
  }
  function showFrame(index) {
    if (!frames.length) { $('original-frame').replaceChildren(empty('没有采样帧')); return; }
    frameIndex = Math.max(0,Math.min(frames.length-1,index)); const f = frames[frameIndex]; $('frame-select').value = String(frameIndex);
    $('previous').disabled = frameIndex === 0; $('next').disabled = frameIndex === frames.length-1;
    const states = new Map((Array.isArray(f.boards)?f.boards:[]).map(b=>[b.board_id,b]));
    const rects = [];
    assemblies.forEach(a=>rects.push({bbox:a.search_region || a.viewport,label:a.assembly_id+' search',color:'#80949c',invalid:true}));
    boards.forEach(b=>{
      const state=states.get(b.board_id); const color=boardById.get(b.board_id)._color;
      const modern = 'measured_screen_roi' in (state || {});
      if (modern) {
        if (state.measured_screen_roi) rects.push({bbox:state.measured_screen_roi,label:b.board_id+' measured',color,invalid:identityUncertain(state)});
        else if(state.predicted_screen_roi) rects.push({bbox:state.predicted_screen_roi,label:b.board_id+' prediction (NOT measurement)',color,invalid:true});
      } else rects.push({bbox:state?.screen_roi || b.roi,label:b.board_id+' legacy stored ROI',color,invalid:identityUncertain(state)});
      (state?.visible_regions || []).forEach(r=>rects.push({bbox:r,color:visibilityUncertain(state)?'#ff9e40':'#40ed8c',invalid:visibilityUncertain(state)}));
    });
    $('camera-state').textContent = f.camera ? `Camera ${f.camera.reliability} · ${f.camera.disturbance_type} · 搜索 ${f.camera.search_attempted ? '已尝试' : '未尝试'} · 恢复计数 ${f.camera.recovery_count ?? 0} · epoch ${f.camera.geometry_epoch} · ${JSON.stringify(f.camera.reason_codes || [])}。本批内容接纳关闭；时间仅为本地录屏时间。` : '旧版记录没有独立相机契约；stored ROI不是已验证动态测量。';
    $('original-frame').replaceChildren(imageBox(f.image_path,`源帧 ${f.frame_id}`,rects));
    $('frame-caption').textContent = `${f.frame_id} · 采样时间 ${time(f.timestamp_ms)} · 源时间 ${f.pts_time_ms == null ? '未知' : time(f.pts_time_ms)} · 原始 PTS ${f.native_pts ?? '未提供'} · 原始 time_base ${f.native_time_base ?? '未提供'} · ${f.time_status === 'verified_pts' ? '已记录源 PTS（原始值见前述字段）' : (f.time_status === 'best_effort_pts' ? 'FFprobe best_effort 时间，需复核；未按FPS猜测' : '时间来源需复核')} · 第 ${frameIndex+1}/${frames.length} 帧。绿色框为机械几何可见区域；橙色虚线为估计区域，未确认可比较。均未进行人体分割。`;
    const invalid = boards.filter(b=>identityUncertain(states.get(b.board_id)));
    $('roi-warning').replaceChildren(); if (invalid.length) $('roi-warning').appendChild(el('p',`物理身份待人工确认：${invalid.map(b=>b.board_id).join('、')}。当前位置只是待复核的几何候选；反向联动或再次可见均不能单独证明同板身份。`,'notice warning'));
    $('board-legend').replaceChildren(); $('board-crops').replaceChildren();
    const groups = new Map();
    assemblies.forEach(a=>{
      const group=el('div',null,'assembly-group'); group.appendChild(el('h3',`机械组件 ${a.assembly_id}`));
      group.appendChild(el('p',`搜索窗口 ${position(a.search_region || a.viewport)}；真实开口 ${position(a.physical_aperture)}（未标定时未知）；outer 在前，inner 在后。`,'caption'));
      const grid=el('div',null,'board-grid'); group.appendChild(grid); $('board-crops').appendChild(group); groups.set(a.assembly_id,grid);
    });
    boards.forEach(b=>{
      const state=states.get(b.board_id); const legend=el('span',boardName(b.board_id)); legend.style.borderBottom=`3px solid ${boardById.get(b.board_id)._color}`; $('board-legend').appendChild(legend);
      const tile=el('div',null,`board-tile${identityUncertain(state)?' roi-invalid':''}`); tile.appendChild(el('h3',boardName(b.board_id)));
      if(b.assembly_id) tile.appendChild(badge(`${b.assembly_id} / ${b.layer}`));
      tile.appendChild(badge(identityUncertain(state)?'identity_uncertain · 待人工确认':(mechanical?'物理身份证据已记录':'人工指定区域'),identityUncertain(state)));
      tile.appendChild(el('p',`观察版本：${state?.version_id ?? '未知'}；可见性：${visibilityName(state?.visibility)}；${visibilityUncertain(state)?'估计可见比例':'几何可见比例'}：${ratio(state)}；几何不可见（含出框）：${state?.visible_fraction==null?'未知':((1-number(state.visible_fraction))*100).toFixed(1)+'%'}；运动：${state?.motion_status ?? '未提供'}。`,'caption'));
      tile.appendChild(el('p',`实测屏幕位置 ${position(state?.screen_roi)}；预测位置 ${position(state?.predicted_screen_roi)}；最后可信 ${position(state?.last_verified_screen_roi)}；位置证据：${state?.position_basis ?? state?.tracking_source ?? state?.identity_basis ?? '未提供'}。`,'caption'));
      if(state?.mechanical_prior) tile.appendChild(el('p',`联动先验：${priorText(state.mechanical_prior)}`,state.mechanical_prior.status==='consistent'?'caption':'notice warning'));
      tile.appendChild(boardEvidence(state,`${boardName(b.board_id)} 当前采样裁剪图`));
      const historyButton=el('button','查看该物理板完整位置 / 遮挡历史','source-button'); historyButton.addEventListener('click',()=>{$('position-board').value=b.board_id;showPositions();$('position-title').scrollIntoView({behavior:'smooth'});}); tile.appendChild(historyButton);
      (groups.get(b.assembly_id) || $('board-crops')).appendChild(tile);
    });
  }
  function showEvent(e) {
    selectedEvent=e.event_id; renderTimeline(); const container=$('evidence'); container.replaceChildren();
    container.appendChild(el('h3',`${e.event_id} · ${boardName(e.board_id)} · ${kindName[e.kind] || e.kind}`));
    container.appendChild(el('p',`观察区间 ${interval(e)}；记录时刻 ${time(e.observed_at_ms)}。这是视觉证据区间，不是确定的书写起止时间。`));
    if (e.order_status==='undetermined_within_interval' || e.simultaneous_group) container.appendChild(el('p',`同区间多板变化，内部先后未知${e.simultaneous_group ? `（组 ${e.simultaneous_group}）` : ''}。`,'notice warning'));
    if (e.kind==='initial_observation') container.appendChild(el('p','片段起始时已经可见；不能推断内容何时写成。','notice'));
    if (e.kind==='roi_needs_reconfirmation' || e.kind==='identity_review_event') container.appendChild(el('p','区域可靠身份待人工重新确认，此后变化不能作为同一实体黑板的可靠连续证据。','notice warning'));
    const grid=el('div',null,'evidence-grid');
    [['变化前',e.before_crop_path,e.before_frame_id,[]],['变化后',e.after_crop_path,e.after_frame_id,e.change_regions || []]].forEach(([title,path,id,regions])=>{
      const state=boardState(id,e.board_id); let displayRegions=regions;
      if ((e.comparison_coordinate_space || e.change_coordinate_space) === 'physical_board_local' && state?.screen_roi && state?.crop_roi) displayRegions=regions.map(r=>({...r,bbox:[r.bbox[0]+state.screen_roi[0]-state.crop_roi[0],r.bbox[1]+state.screen_roi[1]-state.crop_roi[1],r.bbox[2],r.bbox[3]]}));
      const panel=el('div'); append(panel,el('h3',title),boardEvidence({...state,crop_path:path},`${e.event_id} ${title}裁剪`,displayRegions),sourceButton(id));
      if (regions.length) panel.appendChild(el('p','黄色框：可靠共同可见区域中的变化候选；并非已识别公式。','caption')); grid.appendChild(panel);
    }); container.appendChild(grid);
    const references=Array.isArray(e.comparison_references)?e.comparison_references:[];
    if(references.length){
      const refs=el('details');refs.open=true;refs.appendChild(el('summary',`内容比较的历史参考帧（${references.length} 张）`));
      refs.appendChild(el('p',references.length>1?'不同局部区域分别使用其最后一次已确认可见的历史帧。上方“变化前”只展示最早参考，不能代表整块板的唯一前图；下列各图保持各自时间，不合成跨时黑板图。':'此参考对应相关局部区域最后一次已确认可见的证据，不表示精确书写时间。','notice'));
      references.forEach(r=>{const panel=el('div',null,'board-tile');panel.appendChild(el('h3',`${r.frame_id} · ${time(r.timestamp_ms)}`));const state=boardState(r.frame_id,e.board_id);const regions=(r.board_local_regions || []).map(v=>({bbox:Array.isArray(v)?v:v.bbox})).filter(v=>Array.isArray(v.bbox));const shifted=regions.map(v=>({...v,bbox:state?.screen_roi&&state?.crop_roi?[v.bbox[0]+state.screen_roi[0]-state.crop_roi[0],v.bbox[1]+state.screen_roi[1]-state.crop_roi[1],v.bbox[2],v.bbox[3]]:v.bbox}));panel.appendChild(boardEvidence({...state,crop_path:r.crop_path},`${e.board_id} 历史参考 ${r.frame_id}`,shifted));panel.appendChild(sourceButton(r.frame_id));panel.appendChild(el('p',`对应物理板局部区域 ${JSON.stringify(r.board_local_regions || [])}`,'caption'));refs.appendChild(panel);});container.appendChild(refs);
    }
    const facts=el('dl',null,'facts'); fact(facts,'本次版本',e.version_id); fact(facts,'前一版本',e.previous_version_id ?? '无（首次观察）'); fact(facts,'顺序状态',e.order_status); if(mechanical) {fact(facts,'机械组件 / 层',`${e.assembly_id ?? boardById.get(e.board_id)?.assembly_id ?? '未知'} / ${boardById.get(e.board_id)?.layer ?? '未知'}`);fact(facts,'垂直位移',e.delta_y == null ? '未提供' : `${e.delta_y} px`);fact(facts,'运动补偿状态',e.compensation_status ?? '未提供');fact(facts,'运动证据',e.motion_basis ?? e.tracking_source ?? e.position_basis ?? '未提供');fact(facts,'联动先验',priorText(e.mechanical_prior ?? boardState(e.after_frame_id,e.board_id)?.mechanical_prior));} container.appendChild(facts);
    strings(container,'入选原因',e.selected_reasons); strings(container,'不确定性 / 可能干扰',e.uncertainty);
    if (Array.isArray(e.change_regions) && e.change_regions.length) { const d=el('details'); d.appendChild(el('summary','板内变化区域坐标')); d.appendChild(el('pre',JSON.stringify(e.change_regions,null,2))); container.appendChild(d); }
  }
  function renderTimeline() {
    const filter=$('event-board').value; const type=$('event-kind').value; const visible=events.filter(e=>(!filter || e.board_id===filter) && (!type || category(e)===type)); $('event-count').textContent=`${visible.length} / ${events.length} 条`;
    const box=$('timeline'); box.replaceChildren(); if (!visible.length) { box.appendChild(empty('没有符合筛选条件的事件')); return; }
    visible.forEach(e=>{
      const b=el('button',null,`event${selectedEvent===e.event_id?' active':''}`); b.type='button'; b.setAttribute('aria-pressed',String(selectedEvent===e.event_id));
      append(b,el('span',`${time(e.observed_at_ms)} · ${boardName(e.board_id)}`,'event-title'),el('span',`${kindName[e.kind] || e.kind} · ${interval(e)}`,'event-detail'));
      if (e.order_status==='undetermined_within_interval' || e.simultaneous_group) b.appendChild(badge('并列观察 · 区间内顺序未知',true));
      if (e.kind==='roi_needs_reconfirmation') b.appendChild(badge('ROI 需重新确认',true));
      if (Array.isArray(e.uncertainty) && e.uncertainty.length) b.appendChild(badge('有不确定性 / 可能干扰',true));
      b.addEventListener('click',()=>showEvent(e)); box.appendChild(b);
    });
  }
  function collectVersions(boardId) {
    const map=new Map();
    // Frame snapshots preserve access to a version even if there is no event.
    frames.forEach(f=>(f.boards || []).filter(b=>b.board_id===boardId && b.version_id).forEach(b=>{
      if (!map.has(b.version_id)) map.set(b.version_id,{version_id:b.version_id,timestamp_ms:f.timestamp_ms,crop_path:b.crop_path,frame_id:f.frame_id,roi_status:b.roi_status});
    }));
    (Array.isArray(data.versions)?data.versions:[]).filter(v=>v.board_id===boardId).forEach(v=>{ if(v.version_id) map.set(v.version_id,{...map.get(v.version_id),...v}); });
    events.filter(e=>e.board_id===boardId && e.version_id && ['initial_observation','visual_change_candidate','board_content_change_event'].includes(e.kind)).forEach(e=>{const v=map.get(e.version_id);map.set(e.version_id,{version_id:e.version_id,timestamp_ms:e.observed_at_ms,crop_path:e.after_crop_path,frame_id:e.after_frame_id,previous_version_id:e.previous_version_id,kind:e.kind,event_id:e.event_id,...v});});
    return [...map.values()].sort((a,b)=>number(a.timestamp_ms)-number(b.timestamp_ms));
  }
  function showVersions() {
    const id=$('version-board').value; const versions=collectVersions(id); const list=$('versions'); list.replaceChildren(); $('version-evidence').replaceChildren();
    if (!versions.length) { list.appendChild(empty('没有该板的观察版本')); return; }
    versions.forEach(v=>{
      const b=el('button',`${time(v.timestamp_ms)} · ${v.version_id} · ${kindName[v.kind] || '已保留快照'}`); b.type='button'; b.addEventListener('click',()=>{
        [...list.children].forEach(n=>n.classList.remove('active')); b.classList.add('active'); const view=$('version-evidence'); view.replaceChildren();
        append(view,el('p',`版本 ${v.version_id}；前一版本 ${v.previous_version_id ?? v.parent_version_id ?? '无 / 未提供'}；内容状态 ${v.content_status ?? '未提供'}；可见性 ${v.visibility ?? '未知'}；ROI 状态 ${v.roi_status ?? '未知'}。`,'caption'),boardEvidence({...boardState(v.frame_id,id),crop_path:v.crop_path},`${boardName(id)} 版本 ${v.version_id}`),sourceButton(v.frame_id));
        if(v.event_id) { const event=events.find(e=>e.event_id===v.event_id); const eb=el('button','查看此版本的变化证据','source-button'); eb.addEventListener('click',()=>{ showEvent(event); $('evidence-title').scrollIntoView({behavior:'smooth'}); }); view.appendChild(eb); }
      }); list.appendChild(b);
    });
    list.firstElementChild.click();
  }
  function showPositions() {
    const id=$('position-board').value; const list=$('positions'); list.replaceChildren(); $('position-evidence').replaceChildren();
    const supplied=(Array.isArray(data.position_history)?data.position_history:[]).filter(p=>p.board_id===id);
    const positions=supplied.length ? supplied : frames.flatMap(f=>(f.boards || []).filter(b=>b.board_id===id).map(b=>({...b,frame_id:f.frame_id,timestamp_ms:f.timestamp_ms})));
    if(!positions.length) {list.appendChild(empty('没有该板的位置观察'));return;}
    positions.sort((a,b)=>number(a.timestamp_ms)-number(b.timestamp_ms)).forEach((p,index)=>{
      const button=el('button',`${time(p.timestamp_ms)} · ${visibilityName(p.visibility)} · 可见 ${ratio(p)} · ${p.motion_status || '运动未提供'}${identityUncertain(p)?' · 身份待复核':''}`);button.type='button';
      button.addEventListener('click',()=>{
        [...list.children].forEach(n=>n.classList.remove('active'));button.classList.add('active');const view=$('position-evidence');view.replaceChildren();
        const facts=el('dl',null,'facts');fact(facts,'物理板 / 组件 / 层',`${id} / ${p.assembly_id ?? boardById.get(id)?.assembly_id ?? '未提供'} / ${p.layer ?? boardById.get(id)?.layer ?? '未提供'}`);fact(facts,'观察时刻',time(p.timestamp_ms));fact(facts,'当前屏幕位置',position(p.screen_roi));fact(facts,'预测位置（非测量）',position(p.predicted_screen_roi));fact(facts,'最后可信位置',position(p.last_verified_screen_roi));fact(facts,'相机可靠性',p.camera_reliability);fact(facts,'身份独立证据',JSON.stringify(p.identity_evidence ?? p.identity_confidence ?? {}));fact(facts,'内容准入',JSON.stringify(p.comparison_eligibility ?? p.comparison_status));fact(facts,'comparison epoch',p.comparison_epoch);fact(facts,'追踪证据',JSON.stringify(p.tracking_evidence ?? {}));fact(facts,'前次位置',index ? position(positions[index-1].screen_roi) : '起始观察');fact(facts,visibleLabel(p),JSON.stringify(p.visible_regions ?? []));fact(facts,'可见性 / 几何比例（不含人体分割）',`${visibilityName(p.visibility)} / ${ratio(p)}`);fact(facts,'身份状态',p.identity_status ?? p.roi_status);fact(facts,'内容版本',p.version_id ?? '尚无可见内容版本');fact(facts,'位置证据',p.position_basis ?? p.tracking_source ?? p.identity_basis ?? '未提供');fact(facts,'联动先验',priorText(p.mechanical_prior));view.appendChild(facts);
        if(identityUncertain(p))view.appendChild(el('p','身份未确认：保留 board_id 的历史索引，但本次位置 / 重现不作为可靠身份延续。需人工确认。','notice warning'));
        const grid=el('div',null,'evidence-grid');[index?positions[index-1]:null,p].forEach((entry,i)=>{const panel=el('div');panel.appendChild(el('h3',i?'本次观察':'前次观察'));if(entry){panel.appendChild(boardEvidence(entry,`${id} ${time(entry.timestamp_ms)}`));panel.appendChild(sourceButton(entry.frame_id));}else panel.appendChild(empty('片段起始，无前次观察'));grid.appendChild(panel);});view.appendChild(grid);
        const related=events.filter(e=>e.board_id===id && e.after_frame_id===p.frame_id);related.forEach(e=>{const b=el('button',`查看 ${kindName[e.kind] || e.kind} 证据`,'source-button');b.addEventListener('click',()=>{showEvent(e);$('evidence-title').scrollIntoView({behavior:'smooth'});});view.appendChild(b);});
      });list.appendChild(button);
    });list.firstElementChild.click();
  }
  function showRuns() {
    const box=$('runs'); const runs=Array.isArray(data.writing_runs)?data.writing_runs:[]; if (!runs.length) { box.appendChild(empty(data.aggregation_status === 'not_implemented_in_s0_s2' ? 'S0–S2未实施聚合；空列表不代表没有书写。' : '无连续变化候选段')); return; }
    runs.forEach(r=>{
      const d=el('details'); d.appendChild(el('summary',`${r.run_id} · ${boardName(r.board_id)} · ${interval(r)}`)); d.appendChild(el('p','仅为连续视觉变化候选，不确定写作意图或讲解顺序。','caption'));
      (r.event_ids || []).forEach(id=>{const e=events.find(e=>e.event_id===id); if (!e) return; const b=el('button',`${id} · ${interval(e)}`,'source-button'); b.addEventListener('click',()=>showEvent(e)); d.appendChild(b);}); box.appendChild(d);
    });
  }
  $('summary').textContent=`${data.schema_version || '未注明版本'} · ${frames.length} 个保留采样帧 · ${boards.length} 块物理黑板 / ${assemblies.length} 个机械组件 · ${events.length} 条视觉观察事件 · 源视频时长 ${time(data.source?.duration_ms)}。所有图片来自本地文件，无在线请求。`;
  const source = data.source || {}; const probe = source.probe || source;
  fact($('source-metadata'),'编码',probe.codec_name ?? probe.codec ?? '未提供');
  fact($('source-metadata'),'实际分辨率',probe.width != null && probe.height != null ? `${probe.width} × ${probe.height}` : '未提供');
  fact($('source-metadata'),'实际时长',probe.duration_ms != null ? `${probe.duration_ms} ms (${time(probe.duration_ms)})` : (probe.duration != null ? `${probe.duration} s` : '未提供'));
  fact($('source-metadata'),'源 time_base',probe.time_base ?? '未提供');
  fact($('source-metadata'),'视频流索引',probe.video_stream_index ?? '未提供');
  fact($('source-metadata'),'元信息来源',probe.metadata_source ?? source.metadata_source ?? '未提供探测器标识');
  frames.forEach((f,i)=>option($('frame-select'),String(i),`${time(f.timestamp_ms)} · ${f.frame_id}`));
  option($('event-board'),'','全部黑板（全局时间顺序）'); boards.forEach(b=>{option($('event-board'),b.board_id,boardName(b.board_id));option($('version-board'),b.board_id,boardName(b.board_id));option($('position-board'),b.board_id,boardName(b.board_id));});
  $('frame-select').addEventListener('change',()=>showFrame(Number($('frame-select').value))); $('previous').addEventListener('click',()=>showFrame(frameIndex-1)); $('next').addEventListener('click',()=>showFrame(frameIndex+1)); $('event-board').addEventListener('change',renderTimeline);$('event-kind').addEventListener('change',renderTimeline);$('position-board').addEventListener('change',showPositions); $('version-board').addEventListener('change',showVersions);
  showFrame(0); renderTimeline(); if (events.length) showEvent(events[0]); else $('evidence').appendChild(empty('没有事件；全部采样帧仍可查看'));
  if (boards.length) {showVersions();showPositions();} showRuns();
})();
</script>
</body>
</html>
'''
