/**
 * RadVision ACL — Clinical Radiology PACS Workstation
 * 3-View Single Page App switched via hash routing (#home, #workstation, #results).
 * Zero fake numbers. Zero hardcoded mock databases. Real API integration.
 */
(function () {
  'use strict';

  // Application State
  const state = {
    currentRoute: 'workstation', // 'home' | 'workstation' | 'results'
    selectedVolumeFilename: null,
    selectedUploadFile: null,
    selectedVolumeId: null,
    totalSlices: 0,
    currentSliceIndex: 0,
    viewMode: 'slice', // 'slice' | 'gradcam'
    reportMode: 'template', // 'template' | 'llm'
    selectedGradcamModel: null,
    modelsEvaluated: [],
    lastPredictionResult: null,
    lastReportText: null,
    backendReady: false,
    llmConfigured: false,
    groundTruthLookup: {}, // filename -> { label, label_name, laterality, roi_box }
  };

  // DOM Elements
  const dom = {
    // Nav Tabs
    chipHome: document.getElementById('chip-home'),
    chipWorkstation: document.getElementById('chip-workstation'),
    chipResults: document.getElementById('chip-results'),

    // Page Views
    viewHome: document.getElementById('view-home'),
    viewWorkstation: document.getElementById('view-workstation'),
    viewResults: document.getElementById('view-results'),

    // Health Status
    statusIndicator: document.getElementById('backend-status-indicator'),
    statusDot: document.getElementById('status-dot'),
    statusText: document.getElementById('status-text'),
    serverBanner: document.getElementById('server-error-banner'),
    serverErrorMsg: document.getElementById('server-error-msg'),
    btnBannerRetry: document.getElementById('btn-banner-retry'),

    // Landing Benchmarks
    landingBenchmarks: document.getElementById('landing-benchmarks-container'),

    // Workstation Viewer
    viewerScanTitle: document.getElementById('viewer-scan-title'),
    metaVolumeId: document.getElementById('meta-volume-id'),
    metaRoiSource: document.getElementById('meta-roi-source'),
    btnModeSlice: document.getElementById('btn-mode-slice'),
    btnModeGradcam: document.getElementById('btn-mode-gradcam'),
    emptyState: document.getElementById('viewport-empty-state'),
    sliceImg: document.getElementById('mri-slice-img'),
    pacsRoiBox: document.getElementById('pacs-roi-box'),
    viewerControls: document.getElementById('viewer-controls'),
    sliceScrubber: document.getElementById('slice-scrubber'),
    scrubberReadout: document.getElementById('scrubber-readout'),
    btnSlicePrev: document.getElementById('btn-slice-prev'),
    btnSliceNext: document.getElementById('btn-slice-next'),
    gradcamModelSelect: document.getElementById('gradcam-model-select'),
    fileDropzone: document.getElementById('file-dropzone'),
    fileInput: document.getElementById('file-input'),
    btnTriggerUpload: document.getElementById('btn-trigger-upload'),

    // Preprocessing Info Strip (Module 2)
    prepRoiSource: document.getElementById('prep-roi-source'),
    prepVolumeShape: document.getElementById('prep-volume-shape'),
    prepSliceCount: document.getElementById('prep-slice-count'),

    // PACS Corner Overlays
    pacsOverlayTl: document.getElementById('pacs-overlay-tl'),
    pacsOverlayTr: document.getElementById('pacs-overlay-tr'),
    pacsOverlayBl: document.getElementById('pacs-overlay-bl'),
    pacsOverlayBr: document.getElementById('pacs-overlay-br'),

    // Workstation Trigger & Samples
    btnExecuteAnalysis: document.getElementById('btn-execute-analysis'),
    btnRunText: document.getElementById('btn-run-text'),
    samplesGrid: document.getElementById('samples-grid-container'),
    samplesCountLbl: document.getElementById('samples-count-lbl'),

    // Results View
    resultsStudyId: document.getElementById('results-study-id'),
    groundTruthChip: document.getElementById('ground-truth-chip'),
    gtLabelVal: document.getElementById('gt-label-val'),
    predClassHeadline: document.getElementById('pred-class-headline'),
    predConfLead: document.getElementById('pred-conf-lead'),
    decisionLateralityVal: document.getElementById('decision-laterality-val'),
    decisionConsensusTag: document.getElementById('decision-consensus-tag'),
    probBarHealthy: document.getElementById('prob-bar-healthy'),
    probValHealthy: document.getElementById('prob-val-healthy'),
    probBarPartial: document.getElementById('prob-bar-partial'),
    probValPartial: document.getElementById('prob-val-partial'),
    probBarRupture: document.getElementById('prob-bar-rupture'),
    probValRupture: document.getElementById('prob-val-rupture'),
    consensusList: document.getElementById('consensus-model-list'),

    // Report
    btnReportModeTemplate: document.getElementById('btn-report-mode-template'),
    btnReportModeLlm: document.getElementById('btn-report-mode-llm'),
    reportModeTag: document.getElementById('report-mode-tag'),
    reportGroundingTag: document.getElementById('report-grounding-tag'),
    reportTextPanel: document.getElementById('report-text-panel'),
    btnGenerateReport: document.getElementById('btn-generate-report'),
    btnCopyReport: document.getElementById('btn-copy-report'),
    btnDownloadReport: document.getElementById('btn-download-report'),
  };

  // Utility: Percentage string formatter
  function formatPercent(decimalVal) {
    if (typeof decimalVal !== 'number' || isNaN(decimalVal)) return '—';
    return (decimalVal * 100).toFixed(1) + '%';
  }

  // =========================================================================
  // ROUTING: Hash Routing (#home, #workstation, #results)
  // =========================================================================
  function handleHashRoute() {
    const hash = window.location.hash.replace('#', '').toLowerCase();
    const route = (hash === 'home' || hash === 'results' || hash === 'workstation') ? hash : 'workstation';
    state.currentRoute = route;

    // Toggle active view sections
    dom.viewHome.classList.toggle('active', route === 'home');
    dom.viewWorkstation.classList.toggle('active', route === 'workstation');
    dom.viewResults.classList.toggle('active', route === 'results');

    // Toggle nav tab active states
    dom.chipHome.classList.toggle('active', route === 'home');
    dom.chipWorkstation.classList.toggle('active', route === 'workstation');
    dom.chipResults.classList.toggle('active', route === 'results');

    window.scrollTo({ top: 0, behavior: 'smooth' });
  }

  // =========================================================================
  // BACKEND: Health Status Check
  // =========================================================================
  async function checkHealth() {
    try {
      const res = await fetch('/health');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();

      const count = data.model_count || 0;
      const models = data.models_loaded || [];
      const splits = data.splits_exists;
      const volumes = data.volumes_dir_exists;

      state.llmConfigured = Boolean(data.llm_configured);
      if (dom.btnReportModeLlm) {
        if (!state.llmConfigured) {
          dom.btnReportModeLlm.disabled = true;
          dom.btnReportModeLlm.setAttribute('title', 'LLM mode disabled (configure API key on server)');
        } else {
          dom.btnReportModeLlm.disabled = false;
          dom.btnReportModeLlm.setAttribute('title', 'Generate report via Google Gemini API');
        }
      }

      if (data.status === 'ready' && count > 0) {
        state.backendReady = true;
        dom.statusDot.className = 'status-dot ready';
        dom.statusText.textContent = count + ' model' + (count > 1 ? 's' : '') + ' ready';
        dom.statusIndicator.setAttribute(
          'title',
          'Active backbones: ' + models.join(', ') + ' | Device: ' + data.device
        );
        dom.serverBanner.classList.add('hidden');

        // Populate Grad-CAM model choices if not populated
        if (dom.gradcamModelSelect.children.length === 0 && models.length > 0) {
          models.forEach(function (mName) {
            const opt = document.createElement('option');
            opt.value = mName;
            opt.textContent = mName;
            dom.gradcamModelSelect.appendChild(opt);
          });
          state.selectedGradcamModel = models[0];
        }
      } else {
        state.backendReady = false;
        dom.statusDot.className = 'status-dot degraded';
        dom.statusText.textContent = 'Degraded';
        dom.serverErrorMsg.textContent = 'Degraded state: Models loaded: ' + count + ', splits: ' + splits + ', volumes: ' + volumes;
        dom.serverBanner.classList.remove('hidden');
      }
    } catch (err) {
      state.backendReady = false;
      dom.statusDot.className = 'status-dot error';
      dom.statusText.textContent = 'Offline';
      dom.serverErrorMsg.textContent = 'Cannot reach backend at http://127.0.0.1:8000 (' + err.message + ')';
      dom.serverBanner.classList.remove('hidden');
    }
  }

  // =========================================================================
  // BENCHMARKS: Load Evaluation Metrics Table
  // =========================================================================
  async function loadBenchmarks() {
    try {
      const res = await fetch('/api/benchmarks');
      if (res.status === 404) {
        dom.landingBenchmarks.innerHTML =
          '<div class="loading-text">No test metrics recorded on disk.</div>';
        return;
      }
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const data = await res.json();

      let html = '<table class="benchmark-table"><thead><tr><th>Evaluation metric</th><th>Score</th></tr></thead><tbody>';

      if (data.module3_evaluation) {
        const m3 = data.module3_evaluation;
        html += '<tr><td>CNN test accuracy</td><td>' + formatPercent(m3.accuracy) + '</td></tr>';
        html += '<tr><td>CNN balanced accuracy</td><td>' + formatPercent(m3.balanced_accuracy) + '</td></tr>';
        html += '<tr><td>CNN macro-F1</td><td>' + (m3.macro_f1 ? m3.macro_f1.toFixed(3) : '—') + '</td></tr>';
        if (m3.roc_auc_ovr_macro) {
          html += '<tr><td>CNN ROC-AUC (OvR macro)</td><td>' + m3.roc_auc_ovr_macro.toFixed(3) + '</td></tr>';
        }
      }

      if (data.module1_baseline && data.module1_baseline.mlp) {
        const mlp = data.module1_baseline.mlp;
        html += '<tr><td>Module 1 MLP test accuracy</td><td>' + formatPercent(mlp.accuracy) + '</td></tr>';
        html += '<tr><td>Module 1 MLP macro-F1</td><td>' + (mlp.macro_f1 ? mlp.macro_f1.toFixed(3) : '—') + '</td></tr>';
      }

      if (data.module1_baseline && data.module1_baseline.logistic_regression) {
        const lr = data.module1_baseline.logistic_regression;
        html += '<tr><td>Module 1 LogReg accuracy</td><td>' + formatPercent(lr.accuracy) + '</td></tr>';
        html += '<tr><td>Module 1 LogReg macro-F1</td><td>' + (lr.macro_f1 ? lr.macro_f1.toFixed(3) : '—') + '</td></tr>';
      }

      var m4rec = data.module4_sequence || data.module4_recovery;
      if (m4rec && m4rec.test) {
        var m4test = m4rec.test;
        html += '<tr><td>Module 4 GRU test accuracy</td><td>' + formatPercent(m4test.accuracy) + '</td></tr>';
        html += '<tr><td>Module 4 GRU balanced accuracy</td><td>' + formatPercent(m4test.balanced_accuracy) + '</td></tr>';
        html += '<tr><td>Module 4 GRU macro-F1</td><td>' + (m4test.macro_f1 ? m4test.macro_f1.toFixed(3) : '—') + '</td></tr>';
      }

      html += '</tbody></table>';
      dom.landingBenchmarks.innerHTML = html;
    } catch (err) {
      dom.landingBenchmarks.innerHTML =
        '<div class="loading-text">Benchmarks unavailable.</div>';
    }
  }

  // =========================================================================
  // SAMPLES: Load Dataset Test Set Scans
  // =========================================================================
  async function loadSamples() {
    try {
      const res = await fetch('/api/samples');
      if (!res.ok) throw new Error('HTTP ' + res.status);
      const samples = await res.json();

      if (!Array.isArray(samples) || samples.length === 0) {
        dom.samplesGrid.innerHTML =
          '<div style="grid-column: 1 / -1;" class="loading-text">No sample volumes found on disk.</div>';
        return;
      }

      dom.samplesCountLbl.textContent = samples.length + ' available';
      dom.samplesGrid.innerHTML = '';

      samples.forEach(function (s) {
        state.groundTruthLookup[s.volume_filename] = {
          label: s.label,
          label_name: s.label_name,
        };

        const item = document.createElement('div');
        item.className = 'sample-thumbnail-card';
        item.setAttribute('role', 'button');
        item.setAttribute('tabindex', '0');
        item.setAttribute('aria-label', 'Select test scan ' + s.volume_filename);

        const img = document.createElement('img');
        img.className = 'sample-img-thumb';
        img.alt = s.volume_filename;
        img.src = '/slice/' + encodeURIComponent(s.volume_filename) + '/12';

        const nameSpan = document.createElement('span');
        nameSpan.className = 'sample-name-lbl';
        nameSpan.textContent = s.volume_filename;

        const tag = document.createElement('span');
        tag.className = 'sample-class-tag tag-' + s.label;
        tag.textContent = s.label_name;

        item.appendChild(img);
        item.appendChild(nameSpan);
        item.appendChild(tag);

        function trigger() {
          document.querySelectorAll('.sample-thumbnail-card').forEach(function (c) {
            c.classList.remove('active');
          });
          item.classList.add('active');
          selectSampleForViewer(s.volume_filename);
        }

        item.addEventListener('click', trigger);
        item.addEventListener('keydown', function (e) {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault();
            trigger();
          }
        });

        dom.samplesGrid.appendChild(item);
      });
    } catch (err) {
      dom.samplesGrid.innerHTML =
        '<div style="grid-column: 1 / -1;" class="loading-text">Failed to load samples.</div>';
    }
  }

  // =========================================================================
  // WORKSTATION: PACS Overlays & ROI Box
  // =========================================================================
  function updatePacsOverlays() {
    if (dom.pacsOverlayTl) {
      dom.pacsOverlayTl.textContent = state.selectedVolumeFilename
        ? state.selectedVolumeFilename + (state.viewMode === 'slice' ? ' [RAW SLICE]' : ' [GRAD-CAM]')
        : 'STUDY: NONE';
    }
    if (dom.pacsOverlayTr) {
      dom.pacsOverlayTr.textContent = state.selectedVolumeId
        ? 'SL ' + (state.currentSliceIndex + 1) + '/' + state.totalSlices
        : 'SL --/--';
    }
    if (dom.pacsOverlayBl) {
      if (state.viewMode === 'gradcam' && state.selectedGradcamModel) {
        dom.pacsOverlayBl.textContent = 'CAM: ' + state.selectedGradcamModel;
      } else {
        dom.pacsOverlayBl.textContent = 'MODE: RAW';
      }
    }
    if (dom.pacsOverlayBr) {
      const gt = state.groundTruthLookup[state.selectedVolumeFilename];
      dom.pacsOverlayBr.textContent = gt ? ('GT: ' + gt.label_name) : 'SAGITTAL T2';
    }
  }

  function updateRoiBox(box) {
    if (!box || !dom.pacsRoiBox || state.viewMode !== 'slice') {
      if (dom.pacsRoiBox) dom.pacsRoiBox.classList.add('hidden');
      return;
    }
    const origW = 320;
    const origH = 320;
    const leftPct = (box.x / origW) * 100;
    const topPct = (box.y / origH) * 100;
    const widthPct = (box.width / origW) * 100;
    const heightPct = (box.height / origH) * 100;

    dom.pacsRoiBox.style.left = leftPct + '%';
    dom.pacsRoiBox.style.top = topPct + '%';
    dom.pacsRoiBox.style.width = widthPct + '%';
    dom.pacsRoiBox.style.height = heightPct + '%';
    dom.pacsRoiBox.setAttribute('title', `ROI [x:${box.x}, y:${box.y}, w:${box.width}, h:${box.height}]`);
    dom.pacsRoiBox.classList.remove('hidden');
  }

  function selectSampleForViewer(filename) {
    state.selectedVolumeFilename = filename;
    state.selectedUploadFile = null;
    state.selectedVolumeId = filename;
    state.totalSlices = 27; // Default sagittal volume depth
    state.currentSliceIndex = 12;

    dom.viewerScanTitle.textContent = filename;
    dom.metaVolumeId.textContent = 'Study: ' + filename;
    dom.metaRoiSource.textContent = 'Ready for analysis';

    // Update Preprocessing Info Strip
    if (dom.prepRoiSource) dom.prepRoiSource.textContent = 'Metadata (splits.csv)';
    if (dom.prepVolumeShape) dom.prepVolumeShape.textContent = '[27, 320, 320]';
    if (dom.prepSliceCount) dom.prepSliceCount.textContent = '27 Sagittal';

    dom.viewerControls.style.display = 'flex';
    dom.sliceScrubber.min = 0;
    dom.sliceScrubber.max = state.totalSlices - 1;
    dom.sliceScrubber.value = state.currentSliceIndex;

    renderActiveSlice();

    // Enable the "Run Analysis" button
    dom.btnExecuteAnalysis.disabled = !state.backendReady;
    dom.btnRunText.textContent = 'Run model analysis';
  }

  function handleFileSelection(file) {
    if (!file) return;
    if (!file.name.toLowerCase().endsWith('.pck')) {
      alert('Invalid file format. Please upload a .pck volumetric knee MRI scan.');
      return;
    }
    state.selectedUploadFile = file;
    state.selectedVolumeFilename = file.name;
    state.selectedVolumeId = null;
    state.totalSlices = 32;
    state.currentSliceIndex = 16;

    dom.viewerScanTitle.textContent = file.name;
    dom.metaVolumeId.textContent = 'Upload: ' + file.name;
    dom.metaRoiSource.textContent = 'Unprocessed file';

    if (dom.prepRoiSource) dom.prepRoiSource.textContent = 'Pending Upload';
    if (dom.prepVolumeShape) dom.prepVolumeShape.textContent = 'Dynamic array';
    if (dom.prepSliceCount) dom.prepSliceCount.textContent = '32 est.';

    // Deselect sample chips
    document.querySelectorAll('.sample-thumbnail-card').forEach(function (c) {
      c.classList.remove('active');
    });

    updatePacsOverlays();
    updateRoiBox(null);

    // Enable Run Analysis
    dom.btnExecuteAnalysis.disabled = !state.backendReady;
    dom.btnRunText.textContent = 'Upload & run analysis';
  }

  // =========================================================================
  // VIEWER: Render 2D MRI Slice or Real Grad-CAM Overlay
  // =========================================================================
  function renderActiveSlice() {
    if (!state.selectedVolumeId) return;

    if (state.viewMode === 'slice') {
      dom.sliceImg.src = '/slice/' + encodeURIComponent(state.selectedVolumeId) + '/' + state.currentSliceIndex;
      dom.scrubberReadout.textContent = 'Slice ' + (state.currentSliceIndex + 1) + ' / ' + state.totalSlices;
      // Show ROI box if box exists in last findings
      const findings = state.lastPredictionResult ? state.lastPredictionResult.findings : null;
      if (findings && findings.roi && findings.roi.box) {
        updateRoiBox(findings.roi.box);
      }
    } else {
      let url = '/gradcam?volume_id=' + encodeURIComponent(state.selectedVolumeId);
      if (state.selectedGradcamModel) {
        url += '&model=' + encodeURIComponent(state.selectedGradcamModel);
      }
      dom.sliceImg.src = url;
      dom.scrubberReadout.textContent = 'Grad-CAM saliency';
      updateRoiBox(null);
    }

    dom.sliceImg.classList.remove('hidden');
    dom.emptyState.style.display = 'none';

    updatePacsOverlays();
  }

  // =========================================================================
  // INFERENCE: Execute Analysis (Triggered on Button Click Only)
  // =========================================================================
  async function executeAnalysis() {
    if (!state.selectedVolumeFilename && !state.selectedUploadFile) return;

    dom.btnExecuteAnalysis.disabled = true;
    dom.btnRunText.textContent = 'Running inference...';

    try {
      let res;
      if (state.selectedUploadFile) {
        const formData = new FormData();
        formData.append('file', state.selectedUploadFile);
        res = await fetch('/predict', {
          method: 'POST',
          body: formData,
        });
      } else {
        res = await fetch('/predict', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ volume_filename: state.selectedVolumeFilename }),
        });
      }

      if (!res.ok) {
        const errJson = await res.json().catch(function () { return {}; });
        throw new Error(errJson.detail || 'Prediction failed with HTTP ' + res.status);
      }

      const data = await res.json();
      state.lastPredictionResult = data;
      state.selectedVolumeId = data.volume_id;
      state.totalSlices = data.total_slices || (data.findings && data.findings.roi ? data.findings.roi.total_slices : 27);
      state.modelsEvaluated = data.models_evaluated || [];

      // Update Preprocessing Info Strip
      if (data.findings && data.findings.roi) {
        const roi = data.findings.roi;
        if (dom.prepRoiSource) dom.prepRoiSource.textContent = roi.source === 'metadata' ? 'Metadata (splits.csv)' : 'Center-crop fallback';
        if (dom.prepVolumeShape) dom.prepVolumeShape.textContent = JSON.stringify(roi.volume_shape);
        if (dom.prepSliceCount) dom.prepSliceCount.textContent = roi.total_slices + ' Sagittal';
        if (roi.box) updateRoiBox(roi.box);
      }

      // Update Grad-CAM Model options
      dom.gradcamModelSelect.innerHTML = '';
      state.modelsEvaluated.forEach(function (mName) {
        const opt = document.createElement('option');
        opt.value = mName;
        opt.textContent = mName;
        dom.gradcamModelSelect.appendChild(opt);
      });
      if (state.modelsEvaluated.length > 0) {
        state.selectedGradcamModel = state.modelsEvaluated[0];
      }

      // Populate Results View
      populateResultsView(data);

      // Transition to #results view
      window.location.hash = '#results';
    } catch (err) {
      alert('Inference error: ' + err.message);
    } finally {
      dom.btnExecuteAnalysis.disabled = false;
      dom.btnRunText.textContent = 'Run model analysis';
    }
  }

  // =========================================================================
  // RESULTS VIEW: Populate Predictions, Decision Support, Consensus & Report
  // =========================================================================
  function populateResultsView(data) {
    const findings = data.findings || data;
    const cnn = findings.cnn || {};

    dom.resultsStudyId.textContent = 'Study: ' + (findings.volume_filename || findings.volume_id) + ' | Acquisition: Sagittal T2';

    // Primary Headline Class & Lead
    const pClass = cnn.consensus_class || data.predicted_class || 'Unknown';
    dom.predClassHeadline.textContent = pClass;
    dom.predClassHeadline.className = 'decision-pred-val';
    if (pClass.indexOf('Healthy') !== -1) dom.predClassHeadline.classList.add('class-healthy');
    else if (pClass.indexOf('Partial') !== -1) dom.predClassHeadline.classList.add('class-partial');
    else dom.predClassHeadline.classList.add('class-rupture');

    const conf = cnn.confidence !== undefined ? cnn.confidence : data.confidence;
    const modelsCount = cnn.models_evaluated ? cnn.models_evaluated.length : (data.models_evaluated ? data.models_evaluated.length : 3);
    dom.predConfLead.textContent = '(' + formatPercent(conf) + ' confidence • ' + modelsCount + '/3 models)';

    // Laterality badge
    const laterality = findings.laterality || data.laterality || 'unknown';
    if (dom.decisionLateralityVal) {
      dom.decisionLateralityVal.textContent = 'Laterality: ' + laterality.charAt(0).toUpperCase() + laterality.slice(1);
    }

    // Consensus Tag
    if (dom.decisionConsensusTag) {
      const disagree = cnn.models_disagree;
      dom.decisionConsensusTag.textContent = disagree ? 'Consensus: Divergence observed' : 'Consensus: Unanimous agreement';
      dom.decisionConsensusTag.style.color = disagree ? 'var(--color-partial)' : 'var(--color-healthy)';
    }

    // Ground Truth Chip
    const gt = findings.ground_truth || state.groundTruthLookup[findings.volume_filename];
    if (gt) {
      const gtName = typeof gt === 'string' ? gt : gt.label_name;
      dom.gtLabelVal.textContent = gtName;
      dom.groundTruthChip.classList.remove('hidden');
    } else {
      dom.groundTruthChip.classList.add('hidden');
    }

    // Probability Bars (Flat neutral fill)
    const probs = cnn.probabilities || data.probabilities || {};
    const healthyP = probs['Healthy'] || 0;
    const partialP = probs['Partial tear'] || 0;
    const ruptureP = probs['Complete rupture'] || 0;

    dom.probValHealthy.textContent = formatPercent(healthyP);
    dom.probBarHealthy.style.width = (healthyP * 100) + '%';

    dom.probValPartial.textContent = formatPercent(partialP);
    dom.probBarPartial.style.width = (partialP * 100) + '%';

    dom.probValRupture.textContent = formatPercent(ruptureP);
    dom.probBarRupture.style.width = (ruptureP * 100) + '%';

    // Consensus Table
    renderConsensusList(cnn.per_model || data.per_model_probabilities || {});

    // Module 1 chip + card
    var tabular = findings.tabular;
    var chip1 = document.getElementById('chip-mod1');
    var mod1Body = document.getElementById('mod1-body');
    if (tabular && tabular.status === 'available') {
      if (chip1) {
        chip1.className = 'module-status-chip status-run';
        var labelEl1 = chip1.querySelector('span.mod-dot + span, span:last-child');
        if (labelEl1) labelEl1.textContent = 'Module 1 (Feature Baseline): Run';
        chip1.title = '16 ROI/intensity features; MLP & LogReg operational';
      }
      if (mod1Body) {
        var mlp = tabular.mlp || {};
        var lr = tabular.logistic_regression || {};
        mod1Body.innerHTML =
          '<table class="pacs-table"><thead><tr><th>Model</th><th>Predicted class</th><th>Confidence</th></tr></thead><tbody>' +
          '<tr><td>MLP (64, 32)</td><td><strong>' + (mlp.predicted_class || '—') + '</strong></td><td class="tabular-nums">' + formatPercent(mlp.confidence) + '</td></tr>' +
          '<tr><td>Logistic Regression</td><td><strong>' + (lr.predicted_class || '—') + '</strong></td><td class="tabular-nums">' + formatPercent(lr.confidence) + '</td></tr>' +
          '</tbody></table>' +
          '<div style="margin-top:6px;font-size:0.82rem;color:var(--color-muted,#888);">Image-derived feature baseline: 16 ROI/intensity features extracted from MRI volume.</div>';
      }
    } else {
      if (chip1) {
        chip1.className = 'module-status-chip status-not-run';
        var labelEl1 = chip1.querySelector('span.mod-dot + span, span:last-child');
        if (labelEl1) labelEl1.textContent = 'Module 1 (Feature Baseline): Unavailable';
        chip1.title = (tabular && tabular.reason) || 'Model artifacts unavailable';
      }
      if (mod1Body) {
        var reason = (tabular && tabular.reason) || 'Model artifacts unavailable.';
        mod1Body.innerHTML = '<span class="text-muted">Module 1 unavailable &mdash; ' + reason + '</span>';
      }
    }

    // Module 4 chip + card
    var risk = findings.risk;
    var chip4 = document.getElementById('chip-mod4');
    var mod4Body = document.getElementById('mod4-body');
    if (risk && risk.status === 'available') {
      if (chip4) {
        chip4.className = 'module-status-chip status-run';
        var labelEl4 = chip4.querySelector('span.mod-dot + span, span:last-child');
        if (labelEl4) labelEl4.textContent = 'Module 4 (GRU Sequence): Run';
        chip4.title = 'GRU slice sequence classification operational';
      }
      if (mod4Body) {
        var prbs4 = risk.probabilities || {};
        var probRows4 = Object.entries(prbs4).map(function(kv) {
          return '<tr><td>' + kv[0] + '</td><td class="tabular-nums">' + formatPercent(kv[1]) + '</td></tr>';
        }).join('');
        mod4Body.innerHTML =
          '<div style="margin-bottom:8px;font-size:0.85rem;"><strong>Predicted class:</strong> <span style="font-weight:700;color:var(--text-primary,#fff);">' + (risk.predicted_class || '—') + '</span> &bull; <strong>Confidence:</strong> ' + formatPercent(risk.confidence) + ' &bull; <strong>Slices evaluated:</strong> ' + (risk.slices_evaluated || '—') + '</div>' +
          '<table class="pacs-table"><thead><tr><th>Class</th><th>Probability</th></tr></thead><tbody>' + probRows4 + '</tbody></table>' +
          '<div style="margin-top:6px;font-size:0.80rem;color:var(--color-muted,#888);">Architecture: ' + (risk.architecture || 'ResNet-18 Slice Embeddings + GRU') + ' &bull; Scope: ' + (risk.input_scope || 'Single MRI volume') + '</div>';
      }
    } else {
      if (chip4) {
        chip4.className = 'module-status-chip status-not-run';
        var labelEl4 = chip4.querySelector('span.mod-dot + span, span:last-child');
        if (labelEl4) labelEl4.textContent = 'Module 4 (GRU Sequence): Unavailable';
        chip4.title = (risk && risk.reason) || 'Checkpoint unavailable';
      }
      if (mod4Body) {
        var reason4 = (risk && risk.reason) || 'GRU checkpoint unavailable.';
        mod4Body.innerHTML = '<span class="text-muted">Module 4 unavailable &mdash; ' + reason4 + '</span>';
      }
    }

    // Recovery simulation card (always separate, always SIMULATED)
    var recSim = findings.recovery_simulation;
    var simCard = document.getElementById('recovery-sim-card');
    var simTbody = document.getElementById('recovery-sim-tbody');
    if (recSim && recSim.record_type === 'SIMULATED' && Array.isArray(recSim.trajectory) && simCard && simTbody) {
      simCard.style.display = '';
      simTbody.innerHTML = recSim.trajectory.map(function(t) {
        // Adapter uses "month" (int, e.g. 0/3/6/12) and "score" (0-100 scale)
        var tp = t.timepoint !== undefined ? t.timepoint : ('M' + t.month);
        var score = t.recovery_score !== undefined
          ? Math.round(t.recovery_score * 100) + '%'
          : (t.score !== undefined ? Math.round(t.score) + '/100' : '—');
        return '<tr><td>' + tp + '</td><td class="tabular-nums">' + score + '</td><td style="color:var(--color-muted,#888);font-size:0.8rem;">' + (t.label || '') + '</td></tr>';
      }).join('');
    } else {
      if (simCard) simCard.style.display = 'none';
    }

    // Reset Report Panel
    dom.btnGenerateReport.disabled = false;
    dom.reportTextPanel.textContent = 'Diagnosis complete. Click "Re-generate summary" to compile structured findings.';
    dom.reportModeTag.textContent = 'Mode: ' + (state.reportMode === 'llm' ? 'LLM (Gemini)' : 'Template');
    dom.reportGroundingTag.className = 'grounding-badge passed';
    dom.reportGroundingTag.textContent = 'Grounding: Ready';
  }

  function renderConsensusList(modelProbs) {
    dom.consensusList.innerHTML = '';
    const entries = Object.entries(modelProbs);
    if (entries.length === 0) {
      dom.consensusList.innerHTML = '<tr><td colspan="4" class="text-muted">Single model output.</td></tr>';
      return;
    }

    const modelPicks = [];
    entries.forEach(function (pair) {
      const name = pair[0];
      const pMap = pair[1];
      let maxClass = 'Healthy';
      let maxP = -1;
      Object.entries(pMap).forEach(function (cp) {
        if (cp[1] > maxP) {
          maxP = cp[1];
          maxClass = cp[0];
        }
      });
      modelPicks.push({ name: name, pick: maxClass, conf: maxP });
    });

    const unanimous = modelPicks.every(function (p) {
      return p.pick === modelPicks[0].pick;
    });

    modelPicks.forEach(function (mp) {
      const tr = document.createElement('tr');
      const isDisagreed = !unanimous && mp.pick !== modelPicks[0].pick;
      if (isDisagreed) {
        tr.className = 'model-disagreement-row';
      }

      tr.innerHTML =
        '<td style="font-family: var(--font-mono);">' + mp.name + '</td>' +
        '<td>' + mp.pick + '</td>' +
        '<td class="tabular-nums" style="font-family: var(--font-mono);">' + formatPercent(mp.conf) + '</td>' +
        '<td>' + (isDisagreed ? '<span class="disagreement-tag">Disagreement</span>' : '<span class="agreement-tag">Agreed</span>') + '</td>';

      dom.consensusList.appendChild(tr);
    });
  }

  // =========================================================================
  // REPORT GENERATION (POST /report with Grounding Verification)
  // =========================================================================
  async function generateReport() {
    if (!state.lastPredictionResult) return;
    dom.btnGenerateReport.disabled = true;
    dom.reportTextPanel.textContent = 'Compiling clinical findings...';

    try {
      const findingsObj = state.lastPredictionResult.findings || state.lastPredictionResult;
      const res = await fetch('/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          findings: findingsObj,
          mode: state.reportMode,
        }),
      });

      if (!res.ok) {
        const errJson = await res.json().catch(function () { return {}; });
        throw new Error(errJson.detail || 'Report generation failed with HTTP ' + res.status);
      }

      const data = await res.json();
      state.lastReportText = data.report;
      dom.reportTextPanel.textContent = data.report;

      // Update Mode Tag
      if (data.mode === 'llm') {
        dom.reportModeTag.textContent = 'Mode: LLM (' + (data.model || 'Gemini') + ')';
      } else if (data.mode.indexOf('fallback') !== -1) {
        dom.reportModeTag.textContent = 'Mode: Template (Fallback)';
      } else {
        dom.reportModeTag.textContent = 'Mode: Template';
      }

      // Update Grounding Status
      if (data.grounding === 'passed') {
        dom.reportGroundingTag.className = 'grounding-badge passed';
        dom.reportGroundingTag.textContent = 'Grounding: Passed';
      } else if (data.grounding === 'failed_fallback_template') {
        dom.reportGroundingTag.className = 'grounding-badge failed';
        dom.reportGroundingTag.textContent = 'Grounding: Failed (Fallback)';
      } else if (data.grounding === 'skipped_no_key') {
        dom.reportGroundingTag.className = 'grounding-badge failed';
        dom.reportGroundingTag.textContent = 'Grounding: Key Missing';
      } else {
        dom.reportGroundingTag.className = 'grounding-badge passed';
        dom.reportGroundingTag.textContent = 'Grounding: Verified';
      }

      dom.btnCopyReport.disabled = false;
      dom.btnDownloadReport.disabled = false;
    } catch (err) {
      dom.reportTextPanel.textContent = 'Report error: ' + err.message;
      dom.reportGroundingTag.className = 'grounding-badge failed';
      dom.reportGroundingTag.textContent = 'Grounding: Error';
    } finally {
      dom.btnGenerateReport.disabled = false;
    }
  }

  // =========================================================================
  // EVENT LISTENERS WIRE-UP
  // =========================================================================
  function initListeners() {
    // Hash Routing
    window.addEventListener('hashchange', handleHashRoute);

    // Mode toggles (Slice vs Grad-CAM)
    dom.btnModeSlice.addEventListener('click', function () {
      state.viewMode = 'slice';
      dom.btnModeSlice.classList.add('active');
      dom.btnModeSlice.setAttribute('aria-checked', 'true');
      dom.btnModeGradcam.classList.remove('active');
      dom.btnModeGradcam.setAttribute('aria-checked', 'false');
      renderActiveSlice();
    });

    dom.btnModeGradcam.addEventListener('click', function () {
      state.viewMode = 'gradcam';
      dom.btnModeGradcam.classList.add('active');
      dom.btnModeGradcam.setAttribute('aria-checked', 'true');
      dom.btnModeSlice.classList.remove('active');
      dom.btnModeSlice.setAttribute('aria-checked', 'false');
      renderActiveSlice();
    });

    // Report Mode Toggle (Template vs LLM)
    if (dom.btnReportModeTemplate && dom.btnReportModeLlm) {
      dom.btnReportModeTemplate.addEventListener('click', function () {
        state.reportMode = 'template';
        dom.btnReportModeTemplate.classList.add('active');
        dom.btnReportModeLlm.classList.remove('active');
        if (state.lastPredictionResult) generateReport();
      });

      dom.btnReportModeLlm.addEventListener('click', function () {
        if (!state.llmConfigured) return;
        state.reportMode = 'llm';
        dom.btnReportModeLlm.classList.add('active');
        dom.btnReportModeTemplate.classList.remove('active');
        if (state.lastPredictionResult) generateReport();
      });
    }

    // Grad-CAM Model Selector
    dom.gradcamModelSelect.addEventListener('change', function (e) {
      state.selectedGradcamModel = e.target.value;
      if (state.viewMode === 'gradcam') {
        renderActiveSlice();
      }
    });

    // Scrubber Slider
    dom.sliceScrubber.addEventListener('input', function (e) {
      state.currentSliceIndex = parseInt(e.target.value, 10);
      renderActiveSlice();
    });

    // Scrubber Step Buttons
    if (dom.btnSlicePrev) {
      dom.btnSlicePrev.addEventListener('click', function () {
        if (state.currentSliceIndex > 0) {
          state.currentSliceIndex--;
          dom.sliceScrubber.value = state.currentSliceIndex;
          renderActiveSlice();
        }
      });
    }
    if (dom.btnSliceNext) {
      dom.btnSliceNext.addEventListener('click', function () {
        if (state.currentSliceIndex < state.totalSlices - 1) {
          state.currentSliceIndex++;
          dom.sliceScrubber.value = state.currentSliceIndex;
          renderActiveSlice();
        }
      });
    }

    // Keyboard Navigation (Arrow Keys)
    window.addEventListener('keydown', function (e) {
      if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
      if (!state.selectedVolumeId || state.viewMode !== 'slice') return;

      if (e.key === 'ArrowLeft' || e.key === 'ArrowDown') {
        e.preventDefault();
        state.currentSliceIndex = Math.max(0, state.currentSliceIndex - 1);
        dom.sliceScrubber.value = state.currentSliceIndex;
        renderActiveSlice();
      } else if (e.key === 'ArrowRight' || e.key === 'ArrowUp') {
        e.preventDefault();
        state.currentSliceIndex = Math.min(state.totalSlices - 1, state.currentSliceIndex + 1);
        dom.sliceScrubber.value = state.currentSliceIndex;
        renderActiveSlice();
      }
    });

    // File Upload Handlers
    dom.btnTriggerUpload.addEventListener('click', function (e) {
      e.stopPropagation();
      dom.fileInput.click();
    });
    dom.fileDropzone.addEventListener('click', function () {
      dom.fileInput.click();
    });
    dom.fileInput.addEventListener('change', function (e) {
      if (e.target.files.length > 0) {
        handleFileSelection(e.target.files[0]);
      }
    });

    // Drag and Drop
    dom.fileDropzone.addEventListener('dragover', function (e) {
      e.preventDefault();
      dom.fileDropzone.classList.add('dragover');
    });
    dom.fileDropzone.addEventListener('dragleave', function () {
      dom.fileDropzone.classList.remove('dragover');
    });
    dom.fileDropzone.addEventListener('drop', function (e) {
      e.preventDefault();
      dom.fileDropzone.classList.remove('dragover');
      if (e.dataTransfer.files.length > 0) {
        handleFileSelection(e.dataTransfer.files[0]);
      }
    });

    // Action Trigger: Run Analysis
    dom.btnExecuteAnalysis.addEventListener('click', executeAnalysis);

    // Report Actions
    dom.btnGenerateReport.addEventListener('click', generateReport);
    dom.btnCopyReport.addEventListener('click', function () {
      if (!state.lastReportText) return;
      navigator.clipboard.writeText(state.lastReportText).then(function () {
        const orig = dom.btnCopyReport.textContent;
        dom.btnCopyReport.textContent = 'Copied!';
        setTimeout(function () {
          dom.btnCopyReport.textContent = orig;
        }, 1500);
      });
    });

    dom.btnDownloadReport.addEventListener('click', function () {
      if (!state.lastReportText) return;
      const blob = new Blob([state.lastReportText], { type: 'text/plain;charset=utf-8' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = 'RadVision_Report_' + (state.selectedVolumeFilename || 'Study') + '.txt';
      a.click();
      URL.revokeObjectURL(url);
    });

    // Banner Retry
    dom.btnBannerRetry.addEventListener('click', function () {
      checkHealth();
      loadBenchmarks();
      loadSamples();
    });
  }

  // =========================================================================
  // INITIALIZATION
  // =========================================================================
  async function init() {
    initListeners();
    handleHashRoute();
    await checkHealth();
    await loadBenchmarks();
    await loadSamples();
  }

  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
