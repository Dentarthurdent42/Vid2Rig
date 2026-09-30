"""Self-contained HTML viewer for a fitted rig (no external dependencies).

Drag to orbit, shift/right-drag to pan, scroll to zoom. Click a joint, a
point or a legend entry to copy its label to the clipboard.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..geometry import pad3
from ..rig import Rig
from ..tracks import Tracks

PALETTE = ["#2f6fdf", "#e0592a", "#2a9d6f", "#c43d8b", "#8a63d2", "#c9a227", "#1f9bb4",
           "#d6453d", "#5c8a2e", "#9c5b33", "#4f5bd5", "#b8487a", "#3f8f8f", "#a86ad6"]


def write_viewer(rig: Rig, tracks: Tracks, path: str | Path, max_points: int = 1500,
                 title: str = "vid2rig", seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    N = tracks.N
    sel = np.sort(rng.choice(N, size=min(N, max_points), replace=False))
    X = pad3(tracks.points[:, sel])
    V = tracks.visible[:, sel]
    J = pad3(rig.joint_positions())
    tails = pad3(rig.tail_positions())

    allp = np.concatenate([X[V], J.reshape(-1, 3)])
    center = np.median(allp, 0)
    scale = max(np.percentile(np.linalg.norm(allp - center, axis=1), 98), 1e-9)
    q = lambda a: np.round((a - center) / scale * 1000).astype(int)  # noqa: E731

    XP = q(X)
    XP[~V] = 32767
    bone_of_part = {p: i for i, p in enumerate(rig.order)}
    labels = [bone_of_part.get(int(l), -1) for l in rig.fit.labels[sel]]
    is_leaf = [not np.any(rig.parent == i) for i in range(rig.B)]

    bones = []
    for i in range(rig.B):
        j = rig.joints[i]
        bones.append({
            "name": rig.names[i],
            "parent": int(rig.parent[i]),
            "kind": "root" if j is None else j.kind,
            "rms": None if j is None else float(j.rms),
            "range": None if j is None else round(float(j.range_deg), 1),
            "points": int((rig.fit.labels == rig.order[i]).sum()),
            "leaf": bool(is_leaf[i]),
        })
    data = {
        "title": title, "T": rig.T, "fps": rig.fps, "dims": rig.D, "bones": bones,
        "palette": PALETTE,
        "joints": q(J).transpose(1, 0, 2).reshape(rig.T, -1).tolist(),
        "tails": q(tails).transpose(1, 0, 2).reshape(rig.T, -1).tolist(),
        "points": XP.reshape(rig.T, -1).tolist(),
        "pointIds": sel.tolist(),
        "labels": labels,
    }
    html = _TEMPLATE.replace("__TITLE__", title).replace("__DATA__", json.dumps(data, separators=(",", ":")))
    Path(path).write_text(html)


_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>__TITLE__</title>
<style>
:root{--bg:#f6f7f9;--panel:#ffffff;--ink:#1c2230;--muted:#667085;--line:#d9dee7;--grid:#e7eaf0;--accent:#2f6fdf;--toast:#1c2230;--toastInk:#fff}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#11141a;--panel:#191d25;--ink:#e6e9ef;--muted:#98a2b3;--line:#2a303b;--grid:#1f242d;--accent:#6f9bff;--toast:#e6e9ef;--toastInk:#11141a}}
:root[data-theme="dark"]{--bg:#11141a;--panel:#191d25;--ink:#e6e9ef;--muted:#98a2b3;--line:#2a303b;--grid:#1f242d;--accent:#6f9bff;--toast:#e6e9ef;--toastInk:#11141a}
*{box-sizing:border-box}html,body{margin:0;height:100%;background:var(--bg);color:var(--ink);font:14px/1.4 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
.app{display:grid;grid-template-columns:1fr 260px;height:100%}
@media (max-width:760px){.app{grid-template-columns:1fr;grid-template-rows:62vh auto}}
.stage{position:relative;min-height:0}canvas{width:100%;height:100%;display:block;cursor:grab;touch-action:none}
canvas.drag{cursor:grabbing}canvas.hot{cursor:pointer}
aside{background:var(--panel);border-left:1px solid var(--line);padding:14px 16px;overflow:auto}
@media (max-width:760px){aside{border-left:0;border-top:1px solid var(--line)}}
h1{font-size:15px;margin:0 0 2px}.sub{color:var(--muted);font-size:12px;margin-bottom:12px}
.row{display:flex;gap:8px;align-items:center;margin:8px 0;flex-wrap:wrap}
button,select{font:inherit;color:var(--ink);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:4px 10px;cursor:pointer}
input[type=range]{width:100%}label{color:var(--muted);font-size:12px;display:flex;gap:6px;align-items:center;cursor:pointer}
.legend{list-style:none;padding:0;margin:10px 0 0}.legend li{display:flex;align-items:center;gap:8px;padding:4px 6px;border-radius:6px;cursor:pointer}
.legend li:hover,.legend li.hl{background:var(--bg)}.sw{width:10px;height:10px;border-radius:2px;flex:none}
.legend .k{margin-left:auto;color:var(--muted);font-size:12px}
.tip{position:absolute;pointer-events:none;background:var(--panel);border:1px solid var(--line);border-radius:6px;padding:4px 8px;font-size:12px;white-space:nowrap;display:none;box-shadow:0 2px 8px rgba(0,0,0,.12)}
.toast{position:absolute;left:50%;bottom:16px;transform:translateX(-50%);background:var(--toast);color:var(--toastInk);padding:6px 12px;border-radius:6px;font-size:13px;opacity:0;transition:opacity .2s;pointer-events:none}
.hint{color:var(--muted);font-size:12px;margin-top:12px}
</style></head><body>
<div class="app">
 <div class="stage"><canvas id="c"></canvas><div class="tip" id="tip"></div><div class="toast" id="toast"></div></div>
 <aside>
  <h1 id="title"></h1><div class="sub" id="sub"></div>
  <div class="row"><button id="play">Pause</button><select id="speed"><option value="0.25">0.25×</option><option value="0.5">0.5×</option><option value="1" selected>1×</option><option value="2">2×</option></select><button id="reset">Reset view</button></div>
  <input type="range" id="frame" min="0" value="0">
  <div class="row"><label><input type="checkbox" id="showPts" checked>points</label><label><input type="checkbox" id="showBones" checked>bones</label><label><input type="checkbox" id="showNames">names</label></div>
  <ul class="legend" id="legend"></ul>
  <div class="hint">Click a joint, point or legend row to copy its label. Drag to orbit, shift-drag to pan, scroll to zoom.</div>
 </aside>
</div>
<script id="data" type="application/json">__DATA__</script>
<script>
const D=JSON.parse(document.getElementById('data').textContent);
const B=D.bones.length,T=D.T,S=1000;
const cv=document.getElementById('c'),ctx=cv.getContext('2d'),tip=document.getElementById('tip'),toast=document.getElementById('toast');
const col=i=>i<0?getComputedStyle(document.body).getPropertyValue('--muted'):D.palette[i%D.palette.length];
document.getElementById('title').textContent=D.title;
document.getElementById('sub').textContent=`${B} bones · ${T} frames @ ${D.fps} fps · ${D.dims}D`;
const view0={yaw:D.dims===2?0:0.6,pitch:D.dims===2?0:0.25,zoom:1,px:0,py:0};
let view={...view0},frame=0,playing=true,speed=1,hl=-1,hot=null,last=performance.now(),acc=0;
const fr=document.getElementById('frame');fr.max=T-1;
function resize(){const r=cv.getBoundingClientRect(),d=devicePixelRatio||1;cv.width=r.width*d;cv.height=r.height*d;ctx.setTransform(d,0,0,d,0,0);}
addEventListener('resize',()=>{resize();draw()});resize();
function proj(x,y,z){const cy=Math.cos(view.yaw),sy=Math.sin(view.yaw),cp=Math.cos(view.pitch),sp=Math.sin(view.pitch);
 let X=cy*x+sy*z,Z=-sy*x+cy*z,Y=cp*y-sp*Z;Z=sp*y+cp*Z;
 const r=cv.getBoundingClientRect(),f=Math.min(r.width,r.height)*0.42*view.zoom/S,dist=4*S,k=dist/(dist-Z);
 return [r.width/2+view.px+X*f*k,r.height/2+view.py-Y*f*k,Z];}
let shapes=[];
function draw(){const r=cv.getBoundingClientRect();ctx.clearRect(0,0,r.width,r.height);shapes=[];
 const P=D.points[frame],J=D.joints[frame],TL=D.tails[frame];
 if(document.getElementById('showPts').checked){
  for(let i=0;i<D.labels.length;i++){const x=P[3*i];if(x===32767)continue;const p=proj(x,P[3*i+1],P[3*i+2]);const b=D.labels[i];
   ctx.globalAlpha=hl<0||hl===b?0.75:0.12;ctx.fillStyle=col(b);ctx.fillRect(p[0]-1.5,p[1]-1.5,3,3);shapes.push({t:'p',i,x:p[0],y:p[1]});}
  ctx.globalAlpha=1;}
 const jp=[];for(let b=0;b<B;b++)jp.push(proj(J[3*b],J[3*b+1],J[3*b+2]));
 const tp=[];for(let b=0;b<B;b++)tp.push(proj(TL[3*b],TL[3*b+1],TL[3*b+2]));
 if(document.getElementById('showBones').checked){ctx.lineCap='round';
  for(let b=0;b<B;b++){const bo=D.bones[b];const seg=[];
   if(bo.parent>=0)seg.push([jp[bo.parent],jp[b],b]);if(bo.leaf)seg.push([jp[b],tp[b],b]);
   for(const [a,c,k] of seg){ctx.strokeStyle=col(k);ctx.globalAlpha=hl<0||hl===k?1:0.25;ctx.lineWidth=hl===k?5:3.5;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(c[0],c[1]);ctx.stroke();}}
  ctx.globalAlpha=1;
  const ink=getComputedStyle(document.body).getPropertyValue('--ink'),bg=getComputedStyle(document.body).getPropertyValue('--panel');
  for(let b=0;b<B;b++){const p=jp[b],k=D.bones[b].kind,rad=hl===b||(hot&&hot.t==='j'&&hot.i===b)?8:6;
   ctx.fillStyle=bg;ctx.strokeStyle=col(b);ctx.lineWidth=2.5;ctx.beginPath();
   if(k==='hinge'){ctx.rect(p[0]-rad,p[1]-rad,2*rad,2*rad);}else if(k==='fixed'){ctx.moveTo(p[0],p[1]-rad);ctx.lineTo(p[0]+rad,p[1]);ctx.lineTo(p[0],p[1]+rad);ctx.lineTo(p[0]-rad,p[1]);ctx.closePath();}else{ctx.arc(p[0],p[1],rad,0,7);}
   ctx.fill();ctx.stroke();if(k==='root'){ctx.fillStyle=col(b);ctx.beginPath();ctx.arc(p[0],p[1],rad-3,0,7);ctx.fill();}
   shapes.push({t:'j',i:b,x:p[0],y:p[1]});
   if(document.getElementById('showNames').checked){ctx.fillStyle=ink;ctx.font='12px system-ui';ctx.fillText(D.bones[b].name,p[0]+9,p[1]-8);}}}
 fr.value=frame;}
function label(s){if(s.t==='j')return D.bones[s.i].name;const b=D.labels[s.i];return `point_${D.pointIds[s.i]}`;}
function detail(s){if(s.t==='j'){const b=D.bones[s.i];const par=b.parent>=0?D.bones[b.parent].name:'—';
  return `<b>${b.name}</b> · ${b.kind}${b.range!==null?` · ${b.range}° range`:''} · parent ${par} · ${b.points} pts`;}
 const b=D.labels[s.i];return `<b>point_${D.pointIds[s.i]}</b> · ${b>=0?D.bones[b].name:'outlier'}`;}
function pick(mx,my){let best=null,bd=1e9;for(const s of shapes){const d=Math.hypot(s.x-mx,s.y-my),lim=s.t==='j'?11:5,pen=s.t==='j'?-100:0;if(d<lim&&d+pen<bd){bd=d+pen;best=s;}}return best;}
async function copy(text){try{await navigator.clipboard.writeText(text);}catch(e){const ta=document.createElement('textarea');ta.value=text;document.body.appendChild(ta);ta.select();try{document.execCommand('copy');}catch(_){}ta.remove();}
 toast.textContent=`Copied “${text}”`;toast.style.opacity=1;clearTimeout(copy.t);copy.t=setTimeout(()=>toast.style.opacity=0,1200);}
let drag=null,moved=false;
cv.addEventListener('pointerdown',e=>{drag={x:e.clientX,y:e.clientY,pan:e.shiftKey||e.button===2};moved=false;cv.setPointerCapture(e.pointerId);});
cv.addEventListener('pointermove',e=>{const r=cv.getBoundingClientRect(),mx=e.clientX-r.left,my=e.clientY-r.top;
 if(drag){const dx=e.clientX-drag.x,dy=e.clientY-drag.y;if(Math.abs(dx)+Math.abs(dy)>3)moved=true;
  if(drag.pan){view.px+=dx;view.py+=dy;}else{view.yaw+=dx*0.01;view.pitch=Math.max(-1.5,Math.min(1.5,view.pitch+dy*0.01));}
  drag.x=e.clientX;drag.y=e.clientY;cv.classList.add('drag');tip.style.display='none';if(!playing)draw();return;}
 hot=pick(mx,my);cv.classList.toggle('hot',!!hot);
 if(hot){tip.innerHTML=detail(hot)+' · <i>click to copy</i>';tip.style.display='block';tip.style.left=Math.min(mx+12,r.width-tip.offsetWidth-4)+'px';tip.style.top=(my+14)+'px';}else tip.style.display='none';
 if(!playing)draw();});
cv.addEventListener('pointerup',e=>{cv.classList.remove('drag');if(!moved){const r=cv.getBoundingClientRect();const s=pick(e.clientX-r.left,e.clientY-r.top);if(s)copy(label(s));}drag=null;});
cv.addEventListener('pointerleave',()=>{tip.style.display='none';hot=null;});
cv.addEventListener('contextmenu',e=>e.preventDefault());
cv.addEventListener('wheel',e=>{e.preventDefault();view.zoom*=Math.exp(-e.deltaY*0.0015);if(!playing)draw();},{passive:false});
const pb=document.getElementById('play');pb.onclick=()=>{playing=!playing;pb.textContent=playing?'Pause':'Play';last=performance.now();};
document.getElementById('speed').onchange=e=>speed=+e.target.value;
document.getElementById('reset').onclick=()=>{view={...view0};draw();};
fr.oninput=()=>{frame=+fr.value;draw();};
for(const id of ['showPts','showBones','showNames'])document.getElementById(id).onchange=draw;
const lg=document.getElementById('legend');
D.bones.forEach((b,i)=>{const li=document.createElement('li');li.innerHTML=`<span class="sw" style="background:${col(i)}"></span><span>${b.name}</span><span class="k">${b.kind}</span>`;
 li.title='Click to copy';li.onmouseenter=()=>{hl=i;draw();};li.onmouseleave=()=>{hl=-1;draw();};li.onclick=()=>copy(b.name);lg.appendChild(li);});
function tick(now){if(playing){acc+=(now-last)/1000*D.fps*speed;const n=Math.floor(acc);if(n>0){frame=(frame+n)%T;acc-=n;}draw();}last=now;requestAnimationFrame(tick);}
draw();requestAnimationFrame(tick);
</script></body></html>
"""
