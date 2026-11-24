/**
 * Gestos entrenados (ml5 + KNN) — Extensión didáctica para TurboWarp.
 * Compatible con los JSON gestos-blockly-knn-v1 y gestos-blockly-knn-v2
 * exportados por el entrenador ml5.js del proyecto.
 *
 * IMPORTANTE: cargar en TurboWarp como extensión personalizada SIN SANDBOX.
 * No se ejecuta código incluido en los JSON: solo se admiten muestras numéricas.
 * La cámara solo se activa al ejecutar explícitamente el bloque correspondiente.
 */
(function (Scratch) {
  'use strict';
  if (!Scratch.extensions.unsandboxed) {
    throw new Error('Gestos entrenados debe cargarse con «Ejecutar sin sandbox».');
  }

  const EXT_ID = 'gestosentrenados';
  const P5_CDN = 'https://cdn.jsdelivr.net/npm/p5@2.0.5/lib/p5.min.js';
  const ML5_CDN = 'https://unpkg.com/ml5@1/dist/ml5.min.js';
  const MIN_PER_CLASS = 8;
  const K = 7;

  async function ensureLibrary(url, globalName) {
    if (window[globalName]) return;
    await new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = url;
      script.async = true;
      script.onload = resolve;
      script.onerror = () => reject(new Error('No se ha podido cargar ' + globalName + '. Comprueba la conexión o el bloqueo de scripts.'));
      document.head.appendChild(script);
    });
    if (!window[globalName]) throw new Error('La biblioteca ' + globalName + ' no está disponible.');
  }

  function distance(a, b) {
    return Math.hypot(a.x - b.x, a.y - b.y);
  }

  // MISMA normalización usada en el entrenador HTML. No modificar una parte sin la otra.
  function getFeatures(hand) {
    if (!hand || !hand.keypoints || hand.keypoints.length !== 21) return null;
    const p = hand.keypoints, w = p[0], base = p[9];
    const scale = distance(w, base);
    if (!Number.isFinite(scale) || scale < 8) return null;
    const angle = -Math.PI / 2 - Math.atan2(base.y - w.y, base.x - w.x);
    const cos = Math.cos(angle), sin = Math.sin(angle);
    let coords = p.map(q => {
      const dx = (q.x - w.x) / scale, dy = (q.y - w.y) / scale;
      return [dx * cos - dy * sin, dx * sin + dy * cos];
    });
    if (coords[5][0] > coords[17][0]) coords = coords.map(([x, y]) => [-x, y]);
    return coords.flat().map(v => Math.round(v * 1000) / 1000);
  }

  function validateJSON(data) {
    if (!data || typeof data !== 'object') throw new Error('Archivo JSON vacío o inválido.');
    let classes;
    if (data.format === 'gestos-blockly-knn-v1') {
      classes = [
        {id: 'PIEDRA', name: 'Piedra'},
        {id: 'PAPEL', name: 'Papel'},
        {id: 'TIJERAS', name: 'Tijeras'}
      ];
    } else if (data.format === 'gestos-blockly-knn-v2' && Array.isArray(data.classes)) {
      classes = data.classes;
    } else {
      throw new Error('Formato no reconocido. Exporta el JSON desde el entrenador de gestos.');
    }
    if (!Array.isArray(data.samples) || classes.length > 1000 || data.samples.length > 50000 ||
        !classes.every(c => c && typeof c.id === 'string' && /^[A-Za-z0-9_-]{1,60}$/.test(c.id) &&
          typeof c.name === 'string' && c.name.trim().length > 0 && c.name.length <= 50) ||
        new Set(classes.map(c => c.id)).size !== classes.length ||
        new Set(classes.map(c => c.name.trim().toLocaleLowerCase('es'))).size !== classes.length) {
      throw new Error('La lista de clases o de ejemplos está dañada.');
    }
    const ids = new Set(classes.map(c => c.id));
    if (!data.samples.every(s => ids.has(s.label) && Array.isArray(s.features) &&
        s.features.length === 42 && s.features.every(v => typeof v === 'number' && Number.isFinite(v)))) {
      throw new Error('Las muestras no tienen 42 coordenadas numéricas válidas.');
    }
    return {
      classes: classes.map(c => ({id: c.id, name: c.name.trim()})),
      samples: data.samples.map(s => ({label: s.label, features: s.features.slice()}))
    };
  }

  function askForJSONFile() {
    return new Promise((resolve, reject) => {
      const input = document.createElement('input');
      input.type = 'file';
      input.accept = '.json,application/json';
      input.style.display = 'none';
      let done = false;
      const finish = (value, error = null) => {
        if (done) return;
        done = true;
        input.remove();
        if (error) reject(error); else resolve(value);
      };
      input.addEventListener('change', () => finish(input.files && input.files[0] || null), {once: true});
      input.addEventListener('cancel', () => finish(null), {once: true});
      document.body.appendChild(input);
      // Es preciso pulsar este bloque desde la paleta para conservar la activación del usuario.
      try { input.click(); } catch (e) { finish(null, e); }
    });
  }

  class GestosEntrenados {
    constructor() {
      this.classes = [];
      this.samples = [];
      this.readyIds = new Set();
      this.recognizedId = '';
      this.confidence = 0;
      this.seenId = '';
      this.seenCount = 0;
      this.missingFrames = 0;
      this.statusText = 'Sin modelo. Haz clic en «cargar JSON». ';
      this.loading = false;
      this.running = false;
      this.video = null;
      this.handPose = null;
      this.p5instance = null;
      this.panel = null;
      this.canvas = null;
      this.camHost = null;
      this.ctx = null;
    }

    getInfo() {
      return {
        id: EXT_ID,
        name: 'Gestos entrenados',
        color1: '#167D75', color2: '#11675F', color3: '#0F5551',
        blocks: [
          {opcode:'loadModel', blockType:Scratch.BlockType.COMMAND, text:'cargar modelo JSON de gestos'},
          {opcode:'startCamera', blockType:Scratch.BlockType.COMMAND, text:'activar webcam y reconocer gestos'},
          {opcode:'stopCamera', blockType:Scratch.BlockType.COMMAND, text:'detener webcam'},
          '---',
          {opcode:'recognizedGesture', blockType:Scratch.BlockType.REPORTER, text:'gesto reconocido'},
          {opcode:'isGesture', blockType:Scratch.BlockType.BOOLEAN, text:'¿es el gesto [GESTURE]?', arguments:{GESTURE:{type:Scratch.ArgumentType.STRING, menu:'gestureMenu'}}},
          {opcode:'confidencePercent', blockType:Scratch.BlockType.REPORTER, text:'coincidencia (%)'},
          {opcode:'modelStatus', blockType:Scratch.BlockType.REPORTER, text:'estado del reconocedor'},
          {opcode:'classCount', blockType:Scratch.BlockType.REPORTER, text:'número de clases'},
          '---',
          {opcode:'whenGesture', blockType:Scratch.BlockType.EVENT, isEdgeActivated:false,
           text:'cuando reconozca [GESTURE]',
           arguments:{GESTURE:{type:Scratch.ArgumentType.STRING, menu:'gestureMenu'}}}
        ],
        menus: {
          gestureMenu: {acceptReporters:false, items:'gestureMenuItems'}
        }
      };
    }

    gestureMenuItems() {
      return this.classes.length
        ? this.classes.map(c => ({text:c.name, value:c.id}))
        : [{text:'(primero carga un modelo)', value:'__SIN_MODELO__'}];
    }

    modelStatus() { return this.statusText; }
    classCount() { return this.classes.length; }
    recognizedGesture() { return this.classes.find(c => c.id === this.recognizedId)?.name || ''; }
    isGesture(args) { return Boolean(this.recognizedId && this.recognizedId === String(args.GESTURE)); }
    confidencePercent() { return this.recognizedId ? Math.round(this.confidence * 100) : 0; }

    async loadModel() {
      // No hacer ningún await antes de abrir el selector; el navegador exige gesto humano.
      try {
        const file = await askForJSONFile();
        if (!file) { this.statusText = 'Carga cancelada.'; return; }
        if (file.size > 18 * 1024 * 1024) throw new Error('El JSON supera 18 MB.');
        const raw = JSON.parse(await file.text());
        const model = validateJSON(raw);
        this.classes = model.classes;
        this.samples = model.samples;
        this.prepareReadyClasses();
        this.resetDetection();
        this.statusText = 'Modelo cargado: ' + this.classes.length + ' clases, ' + this.samples.length + ' muestras.';
      } catch (err) {
        this.statusText = 'Error al cargar JSON: ' + (err && err.message || err);
        console.error('[Gestos entrenados]', err);
      }
    }

    prepareReadyClasses() {
      const counts = new Map(this.classes.map(c => [c.id, 0]));
      for (const sample of this.samples) counts.set(sample.label, counts.get(sample.label) + 1);
      this.readyIds = new Set(this.classes.filter(c => counts.get(c.id) >= MIN_PER_CLASS).map(c => c.id));
    }

    resetDetection() {
      this.recognizedId = '';
      this.confidence = 0;
      this.seenId = '';
      this.seenCount = 0;
      this.missingFrames = 0;
    }

    classify(features) {
      if (!features || this.readyIds.size < 2) return null;
      // Mismo KNN de 7 vecinos y pesos que la versión de entrenamiento.
      const near = this.samples.filter(s => this.readyIds.has(s.label))
        .map(s => ({label:s.label, d:s.features.reduce((sum, v, i) => sum + (v - features[i]) ** 2, 0)}))
        .sort((a,b) => a.d-b.d).slice(0, K);
      const scores = Object.fromEntries([...this.readyIds].map(id => [id, 0]));
      let total = 0;
      for (const n of near) { const weight = 1 / (0.15 + n.d); scores[n.label] += weight; total += weight; }
      const ready = [...this.readyIds];
      const label = ready.reduce((best,id) => scores[id] > scores[best] ? id : best, ready[0]);
      const match = scores[label] / total;
      if (!Number.isFinite(match) || match < 0.70 || near[0].d > 9) return {label:'', match};
      return {label, match};
    }

    newPrediction(id, match) {
      if (id !== this.seenId) { this.seenId = id; this.seenCount = 1; return; }
      this.seenCount++;
      if (this.seenCount >= 5 && this.recognizedId !== id) {
        this.recognizedId = id;
        this.confidence = Number.isFinite(match) ? match : 0;
        if (id) {
          // Una detección estable lanza el bloque de evento solo una vez.
          Scratch.vm.runtime.startHats(EXT_ID + '_whenGesture', {GESTURE:id});
        }
      } else if (this.recognizedId === id) {
        this.confidence = Number.isFinite(match) ? match : 0;
      }
    }

    drawPoints(hand) {
      if (!this.ctx) return;
      const ctx = this.ctx;
      ctx.clearRect(0, 0, 640, 480);
      if (!hand || !hand.keypoints) return;
      const p = hand.keypoints;
      const chains = [[0,1,2,3,4],[0,5,6,7,8],[0,9,10,11,12],[0,13,14,15,16],[0,17,18,19,20],[5,9,13,17]];
      ctx.lineWidth = 3; ctx.strokeStyle = '#26e9d0';
      for (const chain of chains) {
        ctx.beginPath();
        chain.forEach((index,i) => {
          if (i===0) ctx.moveTo(p[index].x,p[index].y); else ctx.lineTo(p[index].x,p[index].y);
        });
        ctx.stroke();
      }
      for (const q of p) {ctx.beginPath();ctx.arc(q.x,q.y,4,0,Math.PI*2);ctx.fillStyle='#ffee9a';ctx.fill();}
    }

    onHands(results) {
      if (!this.running) return;
      const hand = Array.isArray(results) && results.length ? results[0] : null;
      this.drawPoints(hand);
      const features = getFeatures(hand);
      if (!features) {
        this.missingFrames++;
        if (this.missingFrames >= 8) this.newPrediction('',0);
        return;
      }
      this.missingFrames = 0;
      const prediction = this.classify(features);
      this.newPrediction(prediction?.label || '', prediction?.match || 0);
    }

    ensurePanel() {
      if (this.panel && this.panel.isConnected) return;
      const root = document.createElement('section');
      root.setAttribute('aria-label','Vista previa de gestos entrenados');
      root.style.cssText = 'position:fixed;z-index:99999;bottom:18px;right:18px;width:228px;'+
        'background:#122638;border:1px solid #547585;border-radius:12px;padding:8px;'+
        'box-shadow:0 4px 22px #0006;color:white;font:12px system-ui,sans-serif';
      const heading = document.createElement('div');
      heading.textContent = 'Gestos entrenados · webcam';
      heading.style.cssText='padding:3px 2px 9px;font-weight:600';
      const frame = document.createElement('div');
      frame.style.cssText='position:relative;width:100%;aspect-ratio:4/3;overflow:hidden;border-radius:8px;background:#243b50';
      const holder = document.createElement('div');
      holder.style.cssText='position:absolute;inset:0';
      const canvas = document.createElement('canvas');
      canvas.width=640;canvas.height=480;
      canvas.style.cssText='position:absolute;inset:0;width:100%;height:100%;transform:scaleX(-1);pointer-events:none';
      frame.append(holder,canvas);
      const close = document.createElement('button');
      close.type='button';close.textContent='Detener webcam';
      close.style.cssText='margin-top:7px;padding:5px 9px;width:100%;background:#285d71;color:white;border:0;border-radius:6px;cursor:pointer';
      close.addEventListener('click',()=>this.stopCamera());
      root.append(heading,frame,close);
      document.body.appendChild(root);
      this.panel=root;this.camHost=holder;this.canvas=canvas;this.ctx=canvas.getContext('2d');
    }

    async startCamera() {
      if (this.running || this.loading) return;
      if (this.readyIds.size < 2) {
        this.statusText = 'Primero carga un JSON con al menos dos clases entrenadas (8 muestras por clase).';
        return;
      }
      if (!navigator.mediaDevices?.getUserMedia || !window.isSecureContext) {
        this.statusText = 'La webcam necesita HTTPS o localhost y permiso del navegador.';
        return;
      }
      this.loading=true;
      try {
        this.statusText='Cargando ml5.js y preparando webcam…';
        await ensureLibrary(P5_CDN,'p5');
        await ensureLibrary(ML5_CDN,'ml5');
        this.ensurePanel();
        if (!this.p5instance) {
          this.p5instance = await new Promise(resolve => {
            new window.p5(p => {p.setup = () => {p.noCanvas();resolve(p);};});
          });
        }
        const camera = this.p5instance.createCapture(this.p5instance.VIDEO);
        camera.size(640,480);
        camera.parent(this.camHost);
        camera.elt.setAttribute('playsinline','');
        camera.elt.style.cssText = 'position:absolute;inset:0;width:100%;height:100%;object-fit:cover;transform:scaleX(-1)';
        this.video=camera;
        await new Promise((resolve,reject)=>{
          if(camera.elt.readyState>=2){resolve();return;}
          const timeout=setTimeout(()=>reject(new Error('La cámara no responde. Revisa los permisos.')),15000);
          camera.elt.addEventListener('loadeddata',()=>{clearTimeout(timeout);resolve();},{once:true});
        });
        if (!this.handPose) this.handPose = await window.ml5.handPose({maxHands:1});
        this.resetDetection();
        this.running=true;
        this.handPose.detectStart(camera, results => this.onHands(results));
        this.statusText='Reconociendo gestos. Vista previa de cámara activa.';
      } catch (err) {
        this.statusText='Error de webcam: '+(err && err.message || err);
        console.error('[Gestos entrenados]',err);
        this.stopCamera(false);
      } finally {this.loading=false;}
    }

    stopCamera(setStatus = true) {
      this.running=false;
      try {this.handPose?.detectStop?.();} catch(e) {console.warn(e);}
      try {this.video?.elt?.srcObject?.getTracks().forEach(track => track.stop());} catch(e) {console.warn(e);}
      try {this.video?.remove?.();} catch(e) {console.warn(e);}
      this.video=null;
      if (this.panel) this.panel.remove();
      this.panel=null;this.camHost=null;this.canvas=null;this.ctx=null;
      this.resetDetection();
      if (setStatus) this.statusText='Webcam detenida.';
    }
  }

  Scratch.extensions.register(new GestosEntrenados());
})(Scratch);
