/* RSI explorer — reads <run>_viz.json (or ?data=…); default sample_viz.json.
   Left: 3D PCA scatter of self-clustered failures + arm bars + cluster lessons.
   Right: inference stream (test items) with flip badges; select → detail + 3D highlight. */
"use strict";
const DATA = new URLSearchParams(location.search).get("data") || "sample_viz.json";
const PALETTE = [0x58a6ff,0x3fb950,0xf85149,0xd29922,0xa371f7,0xff7b72,0x39c5cf,0xdb61a2,0x7ee787,0xffa657,0x79c0ff,0xbc8cff];
const clusterColor = i => PALETTE[i % PALETTE.length];
const hex = n => "#" + n.toString(16).padStart(6,"0");
const outcome = it => (!it.base_correct && it.routed_correct) ? "fixed"
                    : (it.base_correct && !it.routed_correct) ? "broke" : "same";
const OUT_COL = {fixed:0x3fb950, broke:0xf85149, same:0x6e7681};

let V, three;

fetch(DATA).then(r=>{ if(!r.ok) throw new Error(r.status); return r.json(); })
  .then(d=>{ V=d; init(); })
  .catch(e=>{ document.getElementById("detail").innerHTML =
    `<span class="bad">Could not load ${DATA} (${e}).</span><br><span class="muted">Serve this dir: <code>python3 -m http.server</code> in site/, then open http://localhost:8000/</span>`; });

function init(){
  document.getElementById("modeltag").textContent = "· " + (V.model||"");
  // build reliable panels first; 3D (fragile WebGL) last, each isolated so one failure can't blank the rest
  safe("bars", buildBars); safe("clusters", buildClusters); safe("list", buildList);
  safe("scene", buildScene);
  window.addEventListener("resize", onResize);
}
function safe(where, fn){
  try { fn(); }
  catch(e){
    console.error("[" + where + "]", e);
    const tgt = document.getElementById(where);
    if(tgt) tgt.insertAdjacentHTML("afterbegin",
      `<div class="bad" style="padding:8px;font-size:12px">⚠ ${where} failed: ${esc(String(e&&e.message||e))}</div>`);
  }
}

/* ---------- 3D scatter (three.js) ---------- */
function buildScene(){
  if (typeof THREE === "undefined") throw new Error("three.js did not load (CDN blocked?)");
  const el = document.getElementById("scene");
  const W = el.clientWidth || el.offsetWidth || 600, H = el.clientHeight || el.offsetHeight || 400;
  const scene = new THREE.Scene(); scene.background = new THREE.Color(0x0d1117);
  const cam = new THREE.PerspectiveCamera(55, W/H, 0.1, 1000);
  cam.position.set(0,0,22);
  const rnd = new THREE.WebGLRenderer({antialias:true}); rnd.setPixelRatio(devicePixelRatio);
  rnd.setSize(W, H); el.appendChild(rnd.domElement);
  let ctrl = null;
  if (THREE.OrbitControls){ ctrl = new THREE.OrbitControls(cam, rnd.domElement); ctrl.enableDamping = true; }
  else { console.warn("OrbitControls unavailable — static 3D view"); }

  // train-failure points, colored by cluster
  const tp = V.train_points;
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(tp.length*3), col = new Float32Array(tp.length*3);
  tp.forEach((p,i)=>{ pos.set(p.xyz,i*3);
    const c = new THREE.Color(clusterColor(p.cluster)); col.set([c.r,c.g,c.b], i*3); });
  g.setAttribute("position", new THREE.BufferAttribute(pos,3));
  g.setAttribute("color", new THREE.BufferAttribute(col,3));
  scene.add(new THREE.Points(g, new THREE.PointsMaterial({size:0.5, vertexColors:true})));

  // cluster centroids (larger)
  V.clusters.forEach(c=>{
    const m = new THREE.Mesh(new THREE.SphereGeometry(0.45,16,16),
      new THREE.MeshBasicMaterial({color:clusterColor(c.id)}));
    m.position.set(...c.centroid); scene.add(m);
  });

  // all test points, faint, colored by outcome
  const test = V.test_items, tg = new THREE.BufferGeometry();
  const tpos = new Float32Array(test.length*3), tcol = new Float32Array(test.length*3);
  test.forEach((it,i)=>{ tpos.set(it.xyz,i*3);
    const c = new THREE.Color(OUT_COL[outcome(it)]); tcol.set([c.r,c.g,c.b], i*3); });
  tg.setAttribute("position", new THREE.BufferAttribute(tpos,3));
  tg.setAttribute("color", new THREE.BufferAttribute(tcol,3));
  scene.add(new THREE.Points(tg, new THREE.PointsMaterial({size:0.32, vertexColors:true, transparent:true, opacity:0.5})));

  // selected-item marker + routing line (updated on select)
  const sel = new THREE.Mesh(new THREE.SphereGeometry(0.6,20,20),
    new THREE.MeshBasicMaterial({color:0xffffff})); sel.visible=false; scene.add(sel);
  const lineMat = new THREE.LineBasicMaterial({color:0xffffff});
  const lineGeo = new THREE.BufferGeometry().setFromPoints([new THREE.Vector3(),new THREE.Vector3()]);
  const line = new THREE.Line(lineGeo, lineMat); line.visible=false; scene.add(line);

  (function loop(){ requestAnimationFrame(loop); if(ctrl) ctrl.update(); rnd.render(scene,cam); })();
  three = {scene,cam,rnd,ctrl,el,sel,line,lineGeo};
  requestAnimationFrame(onResize);   // re-fit once layout has settled (fixes 0-size canvas)
}
function onResize(){ if(!three) return; const {el,cam,rnd}=three;
  const W = el.clientWidth || el.offsetWidth || 600, H = el.clientHeight || el.offsetHeight || 400;
  cam.aspect = W/H; cam.updateProjectionMatrix(); rnd.setSize(W,H); }

function highlight(it){
  if(!three) return; const {sel,line,lineGeo}=three;
  sel.position.set(...it.xyz); sel.visible=true;
  const c = V.clusters.find(c=>c.id===it.routed_cluster);
  if(c){ lineGeo.setFromPoints([new THREE.Vector3(...it.xyz), new THREE.Vector3(...c.centroid)]); line.visible=true; }
}

/* ---------- arm bars (d3) ---------- */
function buildBars(){
  const arms = Object.entries(V.arms);
  const max = Math.max(...arms.map(([,a])=>a.acc), 0.01);
  const box = d3.select("#bars");
  arms.forEach(([name,a])=>{
    const row = box.append("div").attr("class","bar-row");
    row.append("div").attr("class","bar-lab").text(name);
    row.append("div").attr("class","bar-track").append("div").attr("class","bar-fill")
      .style("width", (100*a.acc/max)+"%");
    const sig = (a.p!=null && a.p<0.05) ? " *" : "";
    const bc = (name==="base") ? "" : ` (+${a.b}/-${a.c}${sig})`;
    row.append("div").attr("class","bar-val").text((100*a.acc).toFixed(1)+"%"+bc);
  });
}

/* ---------- cluster lessons ---------- */
function buildClusters(){
  const box = d3.select("#clusters");
  V.clusters.forEach(c=>{
    const div = box.append("div").attr("class","cl").style("border-left-color", hex(clusterColor(c.id)));
    const t = div.append("div").attr("class","ttl");
    const nm = c.name ? ` ${c.name}` : ` cluster ${c.id}`;
    t.append("span").html(`<span class="dot" style="background:${hex(clusterColor(c.id))}"></span><b style="color:var(--fg)">${esc(nm.trim())}</b> <span class="muted">#${c.id}</span>`);
    t.append("span").text(`n=${c.size}`);
    const ul = div.append("ul");
    (c.lessons.length?c.lessons:["(no lesson)"]).forEach(l=>ul.append("li").text(l));
  });
}

/* ---------- inference stream list + detail ---------- */
function buildList(){
  const test = V.test_items;
  const f = test.filter(it=>outcome(it)==="fixed").length, b = test.filter(it=>outcome(it)==="broke").length;
  document.getElementById("counts").innerHTML =
    `${test.length} tasks · <span class="ok">${f} fixed</span> · <span class="bad">${b} broke</span>`;
  const list = d3.select("#list");
  test.forEach((it,i)=>{
    const o = outcome(it);
    const row = list.append("div").attr("class","item").attr("data-i",i).on("click",()=>select(i));
    row.append("span").attr("class","task").text(it.task);
    const lab = {fixed:"✗→✓",broke:"✓→✗",same:(it.base_correct?"✓→✓":"✗→✗")}[o];
    row.append("span").attr("class","badge b-"+(o==="same"?"same":o)).text(lab);
  });
  document.getElementById("play").onclick = play;
}

let selIdx=-1;
function select(i){
  selIdx=i; const it=V.test_items[i];
  document.querySelectorAll(".item").forEach(e=>e.classList.toggle("sel", +e.dataset.i===i));
  const el = document.querySelector(`.item[data-i="${i}"]`); if(el) el.scrollIntoView({block:"nearest"});
  highlight(it);
  const mark = ok => ok?'<span class="ok">✓ correct</span>':'<span class="bad">✗ wrong</span>';
  const cc = hex(clusterColor(it.routed_cluster));
  document.getElementById("detail").innerHTML = `
    <div><span class="tag">${it.task}</span>
      <span class="tag">routed → <span class="dot" style="background:${cc}"></span>${esc(it.routed_cluster_name || ("cluster "+it.routed_cluster))}</span>
      <span class="tag">gold: ${esc(it.gold)}</span></div>
    <h2 style="padding-left:0">prompt</h2><pre>${esc(it.prompt)}</pre>
    <h2 style="padding-left:0">routed lessons</h2>
    <pre>${it.routed_lessons.length?it.routed_lessons.map(l=>"- "+esc(l)).join("\n"):"(none)"}</pre>
    <h2 style="padding-left:0">base output (no lessons) — ${mark(it.base_correct)}</h2><pre>${esc(it.base_output)}</pre>
    <h2 style="padding-left:0">with routed lessons — ${mark(it.routed_correct)}</h2><pre>${esc(it.routed_output)}</pre>`;
}
const esc = s => (s==null?"":String(s)).replace(/[&<>]/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;"}[c]));

let timer=null;
function play(){
  const btn=document.getElementById("play");
  if(timer){ clearInterval(timer); timer=null; btn.textContent="▶ play"; return; }
  btn.textContent="⏸ pause"; let i=Math.max(selIdx,0);
  timer=setInterval(()=>{ if(i>=V.test_items.length){clearInterval(timer);timer=null;btn.textContent="▶ play";return;} select(i++); }, 900);
}

function fit(){ /* leave default camera; OrbitControls handles the rest */ }
