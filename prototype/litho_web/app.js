/* Lithophane Studio — frontend logic + Three.js 3D view + pywebview bridge */

const $ = (id) => document.getElementById(id);

/* ===== Status ===== */
function setStatus(text, kind = '') {
  const el = $('status');
  el.textContent = text;
  el.className = 'status' + (kind ? ' ' + kind : '');
}

/* ===== Tabs ===== */
document.querySelectorAll('.tab').forEach(t => t.addEventListener('click', () => {
  document.querySelectorAll('.tab').forEach(x => x.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(x => x.classList.remove('active'));
  t.classList.add('active');
  $('tab-' + t.dataset.tab).classList.add('active');
  if (t.dataset.tab === 'view3d') requestAnimationFrame(() => onResize3D());
}));

/* ===== Params read ===== */
function readParams() {
  return {
    mode: $('mode').value,
    order: $('order').value,
    width_mm: parseFloat($('width').value) || 144,
    height_mm: parseFloat($('height').value) || 108,
    layer_h: parseFloat($('layerh').value) || 0.2,
    layers: parseInt($('layers').value) || 8,
    pitch: parseFloat($('pitch').value) || 0.3,
    dwhite: parseFloat($('dwhite').value) || 0.3,
    td_c: parseFloat($('tdc').value) || 0.5,
    td_m: parseFloat($('tdm').value) || 0.5,
    td_y: parseFloat($('tdy').value) || 0.5,
    dwhite: parseFloat($('dwhite').value) || 0.8,
    maxthick: parseFloat($('maxthick').value) || 2.8,
    carve: $('carve').value,
    sharpen: parseFloat($('sharpen').value) || 0.5,
    contrast: parseFloat($('contrast').value) || 1.3,
  };
}

window.onModeChange = async () => {
  // INTERLEAVED / OVERLAP / BAMBU lock order to MIXED; LAYERED offers the 6
  // CMY orders.
  const mode = $('mode').value;
  const order = $('order');
  if (mode === 'interleaved' || mode === 'overlap' || mode === 'bambu') {
    order.value = 'MIXED';
    order.disabled = true;
  } else if (mode === 'layered') {
    order.disabled = false;
    if (order.value === 'MIXED') order.value = 'CMY';
  } else {
    order.disabled = true;
    order.value = 'CMY';
  }
  // Auto rebuild if image loaded.
  if (window.hasImage) window.build();
};
window.onParamChange = () => { if (window.hasImage) debouncedBuild(); };

let debounceTimer = null;
function debouncedBuild() {
  clearTimeout(debounceTimer);
  debounceTimer = setTimeout(() => window.build(), 500);
}

/* ===== Image pick ===== */
window.pickImage = async () => {
  const result = await pywebview.api.pick_image();
  if (result && result.ok) {
    window.hasImage = true;
    $('orig-preview').innerHTML = `<img src="data:image/png;base64,${result.thumb}">`;
    $('btn-build').disabled = false;
    $('btn-reverse').disabled = false;
    setStatus(`已加载: ${result.name} (${result.w}×${result.h}px)`, 'ok');
    // Default size = image long side 144mm.
    const scale = 144 / Math.max(result.w, result.h);
    $('width').value = Math.round(result.w * scale);
    $('height').value = Math.round(result.h * scale);
    window.build();
  } else {
    setStatus(result?.error || '取消选择', 'err');
  }
};

/* ===== Build ===== */
window.build = async () => {
  setStatus('构建中…', 'busy');
  $('btn-build').disabled = true;
  try {
    const params = readParams();
    const result = await pywebview.api.build(params);
    if (!result.ok) { setStatus('构建失败: ' + result.error, 'err'); return; }

    // Show WYSIWYG preview.
    if (result.reached_b64) {
      $('reached-preview').innerHTML = `<img src="data:image/png;base64,${result.reached_b64}">`;
    }
    // Show stats.
    const dE = result.dE_med;
    const stats = `dE中位=${dE} · 面数 ${(result.total_faces/1e6).toFixed(2)}M · 耗时 ${result.elapsed}s`;
    $('stats').textContent = stats;
    // Feed 3D view.
    render3D(result.meshes);
    $('btn-build').disabled = false;
    $('btn-export').disabled = false;
    setStatus(`构建完成 · ${stats}`, 'ok');
  } catch (e) {
    setStatus('构建错误: ' + e, 'err');
    $('btn-build').disabled = false;
  }
};

/* ===== Export ===== */
window.exportAll = async () => {
  const fmt = $('fmt').value;
  setStatus('导出中…', 'busy');
  const result = await pywebview.api.export_all(fmt);
  setStatus(result.ok ? `已导出 → ${result.dir}` : '导出失败: ' + result.error, result.ok ? 'ok' : 'err');
};

/* ===== Reverse import ===== */
window.reverseImport = async () => {
  setStatus('反向导入…', 'busy');
  const result = await pywebview.api.reverse_import();
  if (result.ok && result.recon_b64) {
    $('reached-preview').innerHTML = `<img src="data:image/png;base64,${result.recon_b64}">`;
    setStatus('已还原预览（对比上方原图）', 'ok');
  } else {
    setStatus(result.error || '取消', 'err');
  }
};

/* ===================== Three.js 3D view ===================== */
let renderer, scene, camera, controls = { rx: -0.5, ry: 0.6, zoom: 1, px: 0, py: 0 };
let meshGroup = null;
let layerMeshes = {};   // color -> THREE.Mesh
let layerVisible = { W: true, C: true, M: true, Y: true, top: true };
let isDragging = false, dragBtn = 0, lastX = 0, lastY = 0;

const LAYER_COLORS = { W: 0xe2e8f0, C: 0x38bdf8, M: 0xf472b6, Y: 0xfde047, top: 0xffffff };

function init3D() {
  const container = $('three-container');
  // The 3D tab may be hidden (display:none) on first build -> client size is
  // 0. Fall back to a sane size; onResize3D() corrects it once the tab is
  // shown (avoids a 0x0 renderer and NaN aspect at first frame).
  const w = container.clientWidth || 800;
  const h = container.clientHeight || 600;
  renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
  renderer.setPixelRatio(window.devicePixelRatio);
  renderer.setSize(w, h);
  container.appendChild(renderer.domElement);

  scene = new THREE.Scene();
  scene.background = new THREE.Color(0x0f1220);

  camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 1000);
  camera.position.set(0, 0, 80);

  // Lights
  scene.add(new THREE.AmbientLight(0xffffff, 0.7));
  const d1 = new THREE.DirectionalLight(0xffffff, 0.8); d1.position.set(30, 40, 60); scene.add(d1);
  const d2 = new THREE.DirectionalLight(0xffffff, 0.3); d2.position.set(-30, -20, 20); scene.add(d2);

  // Grid helper
  const grid = new THREE.GridHelper(60, 20, 0x2c3560, 0x1b2138);
  grid.rotation.x = Math.PI / 2;  // lay flat
  scene.add(grid);

  // Events
  container.addEventListener('mousedown', e => { isDragging = true; dragBtn = e.button; lastX = e.clientX; lastY = e.clientY; });
  window.addEventListener('mouseup', () => { isDragging = false; });
  window.addEventListener('mousemove', e => {
    if (!isDragging) return;
    const dx = e.clientX - lastX, dy = e.clientY - lastY;
    lastX = e.clientX; lastY = e.clientY;
    if (dragBtn === 0) { controls.ry += dx * 0.01; controls.rx += dy * 0.01; }
    else { controls.px += dx; controls.py -= dy; }
    refreshView();
  });
  container.addEventListener('wheel', e => {
    e.preventDefault();
    controls.zoom *= e.deltaY > 0 ? 0.92 : 1.08;
    refreshView();
  }, { passive: false });
}
window.addEventListener('resize', onResize3D);
function onResize3D() {
  const container = $('three-container');
  if (!renderer) return;
  renderer.setSize(container.clientWidth, container.clientHeight);
  camera.aspect = container.clientWidth / container.clientHeight || 1;
  camera.updateProjectionMatrix();
  refreshView();
}

function render3D(meshes) {
  if (!renderer) init3D();
  // Dispose old layers' geometries.
  for (const key of Object.keys(layerMeshes)) {
    const m = layerMeshes[key];
    if (m) { m.geometry.dispose(); m.material.dispose(); }
  }
  layerMeshes = {};

  if (meshGroup) { scene.remove(meshGroup); }
  meshGroup = new THREE.Group();
  scene.add(meshGroup);

  const built = [];
  for (const [color, data] of Object.entries(meshes)) {
    if (!data || data.verts.length === 0) continue;
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute('position', new THREE.Float32BufferAttribute(data.verts, 3));
    geometry.setIndex(data.faces);
    geometry.computeVertexNormals();
    const mat = new THREE.MeshStandardMaterial({
      color: LAYER_COLORS[color] || 0x888888,
      roughness: 0.6, metalness: 0.1,
      transparent: color === 'top', opacity: color === 'top' ? 0.85 : 1.0,
      side: THREE.DoubleSide,  // decimated preview mesh is a thin shell
    });
    const mesh = new THREE.Mesh(geometry, mat);
    mesh.visible = layerVisible[color] !== false;
    built.push(mesh);
    layerMeshes[color] = mesh;
  }

  // Center the WHOLE stack (all layers) once so the Z offsets between
  // layers survive — per-mesh centering collapses W/C/M/Y/top to z≈0 and
  // makes the layers interpenetrate. Orbit/pivot = union bbox center.
  if (built.length) {
    const bb = new THREE.Box3();
    for (const m of built) bb.expandByObject(m);
    const center = bb.getCenter(new THREE.Vector3());
    for (const m of built) m.position.sub(center);
  }
  for (const m of built) meshGroup.add(m);

  refreshView();
}

// Interaction-only refresh: apply the orbit transform to the EXISTING
// meshGroup and re-render. Never disposes/rebuilds — calling render3D()
// without data used to wipe the model (dispose + empty group rebuild).
function refreshView() {
  if (!meshGroup) return;
  meshGroup.rotation.x = controls.rx;
  meshGroup.rotation.y = controls.ry;
  meshGroup.scale.setScalar(controls.zoom * 0.6);
  meshGroup.position.x = controls.px;
  meshGroup.position.y = controls.py;
  renderer.render(scene, camera);
}

/* Layer visibility toggles (from the sidebar legend) */
function toggleLayer(color, visible) {
  layerVisible[color] = visible;
  if (layerMeshes[color]) layerMeshes[color].visible = visible;
  refreshView();
}

/* ===== Boot ===== */
window.addEventListener('DOMContentLoaded', () => {
  onModeChange();
  setStatus('就绪 — 选择图片开始');
});
