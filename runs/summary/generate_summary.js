import fs from 'node:fs/promises';
import fssync from 'node:fs';
import path from 'node:path';
import crypto from 'node:crypto';
import { fileURLToPath } from 'node:url';

let sharpModule = null;
try { sharpModule = await import('sharp'); } catch (err) { sharpModule = null; }
const sharp = sharpModule ? (sharpModule.default || sharpModule) : null;

const THIS_FILE = fileURLToPath(import.meta.url);
const SUMMARY_DIR = path.dirname(THIS_FILE);
const ROOT = path.resolve(SUMMARY_DIR, '..', '..');
const RUNS_ROOT = path.join(ROOT, 'runs');
const PLOTS_DIR = path.join(SUMMARY_DIR, 'plots');
const DATASET_FOLDERS = ['cifar100', 'malimg', 'rawmaltf'];
const CONDITION_ORDER = [
  ['none', null],
  ['random', 4],
  ['random', 8],
  ['cam_low', 4],
  ['cam_high', 4],
  ['cam_low', 8],
  ['cam_high', 8]
];
const TRAIN_ACC_CANDIDATES = ['train_acc', 'train_accuracy', 'train_acc1', 'train_top1', 'acc_train'];
const EVAL_ACC_CANDIDATES = ['eval_acc', 'eval_accuracy', 'eval_acc1', 'val_acc', 'val_accuracy', 'val_acc1', 'test_acc', 'test_accuracy', 'test_acc1', 'valid_acc', 'valid_accuracy'];
const TRAIN_LOSS_CANDIDATES = ['train_loss', 'loss_train', 'training_loss'];
const EVAL_LOSS_CANDIDATES = ['eval_loss', 'val_loss', 'validation_loss', 'valid_loss', 'test_loss', 'loss_eval'];
const PALETTE = {
  'none': '#4b5563',
  'random M4': '#2563eb',
  'random M8': '#60a5fa',
  'cam_low M4': '#059669',
  'cam_high M4': '#dc2626',
  'cam_low M8': '#34d399',
  'cam_high M8': '#f87171',
  'CAM': '#7c3aed',
  'random': '#2563eb',
  'M4': '#0f766e',
  'M8': '#f97316',
  'cam_low': '#059669',
  'cam_high': '#dc2626'
};

function rel(p) { return path.relative(ROOT, p).replace(/\\/g, '/'); }
function exists(p) { return fssync.existsSync(p); }
async function ensureDirs() { await fs.mkdir(PLOTS_DIR, { recursive: true }); }
function normDataset(value) {
  const text = String(value || '').trim().toLowerCase();
  if (['drive_zip', 'rawmaltf', 'rawmal-tf', 'rawmal_tf'].includes(text)) return 'rawmaltf';
  return text;
}
function normMode(value) {
  const text = String(value || '').trim().toLowerCase();
  if (text === 'cam-low' || text === 'cam low') return 'cam_low';
  if (text === 'cam-high' || text === 'cam high') return 'cam_high';
  return text;
}
function safeFloat(value) {
  if (value === null || value === undefined) return null;
  if (typeof value === 'number') return Number.isFinite(value) ? value : null;
  const text = String(value).trim();
  if (!text || ['nan', 'none', 'null', 'na', 'n/a'].includes(text.toLowerCase())) return null;
  const num = Number(text);
  return Number.isFinite(num) ? num : null;
}
function safeInt(value) {
  const f = safeFloat(value);
  return f === null ? null : Math.trunc(f);
}
function compareArea(value) {
  const f = safeFloat(value);
  return f === null ? '' : Number(f).toPrecision(12).replace(/\.?0+$/, '');
}
function conditionLabel(mode, mValue) {
  const m = safeInt(mValue);
  const normalized = normMode(mode);
  if (normalized === 'none') return 'none';
  return m === null ? normalized : normalized + ' M' + m;
}
function conditionSlug(label) { return String(label).replace(/ /g, '_').replace(/-/g, '_'); }
function fmtFloat(value, digits) {
  const f = safeFloat(value);
  if (f === null) return '';
  return f.toFixed(digits === undefined ? 4 : digits);
}
function fmtPct(value, digits) {
  const f = safeFloat(value);
  if (f === null) return '';
  return (f * 100).toFixed(digits === undefined ? 2 : digits) + '%';
}
function mean(values) { return values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; }
function median(values) {
  if (!values.length) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}
function stats(values) {
  const vals = values.map(safeFloat).filter(v => v !== null);
  if (!vals.length) return { count: 0 };
  return {
    count: vals.length,
    mean: mean(vals),
    median: median(vals),
    min: Math.min(...vals),
    max: Math.max(...vals),
    positive_count: vals.filter(v => v > 0).length,
    zero_count: vals.filter(v => Math.abs(v) <= 1e-12).length,
    negative_count: vals.filter(v => v < 0).length
  };
}
function parseCSV(text) {
  const rows = [];
  let row = [];
  let cell = '';
  let inQuotes = false;
  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else inQuotes = false;
      } else cell += ch;
    } else {
      if (ch === '"') inQuotes = true;
      else if (ch === ',') { row.push(cell); cell = ''; }
      else if (ch === '\n') { row.push(cell); rows.push(row); row = []; cell = ''; }
      else if (ch !== '\r') cell += ch;
    }
  }
  if (cell.length || row.length) { row.push(cell); rows.push(row); }
  if (!rows.length) return { headers: [], records: [] };
  const headers = rows[0].map(h => h.replace(/^\uFEFF/, ''));
  const records = rows.slice(1).filter(r => r.some(c => String(c).trim() !== '')).map(r => {
    const obj = {};
    headers.forEach((h, i) => { obj[h] = r[i] === undefined ? '' : r[i]; });
    return obj;
  });
  return { headers, records };
}
function csvEscape(value) {
  if (value === null || value === undefined) return '';
  let text;
  if (typeof value === 'number') text = Number.isFinite(value) ? String(Number(value.toPrecision(12))) : '';
  else if (typeof value === 'boolean') text = value ? 'true' : 'false';
  else if (Array.isArray(value) || typeof value === 'object') text = JSON.stringify(value);
  else text = String(value);
  if (/[",\n\r]/.test(text)) return '"' + text.replace(/"/g, '""') + '"';
  return text;
}
async function writeCSV(filePath, rows, columns) {
  const lines = [columns.join(',')];
  for (const row of rows) lines.push(columns.map(c => csvEscape(row[c])).join(','));
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, lines.join('\n') + '\n', 'utf8');
}
async function readJSON(filePath) {
  try { return { value: JSON.parse(await fs.readFile(filePath, 'utf8')), error: '' }; }
  catch (err) { return { value: null, error: String(err.message || err) }; }
}
async function sha256File(filePath) {
  const h = crypto.createHash('sha256');
  const data = await fs.readFile(filePath);
  h.update(data);
  return h.digest('hex');
}
function parseRunName(name) {
  const out = { folder_model: '', folder_seed: null, folder_cutout_mode: '', folder_cutout_m: null, folder_cutout_area: null };
  const m = name.match(/^(.+?)_seed(\d+)_(none|random_M\d+_area[0-9.]+|cam_low_M\d+_area[0-9.]+|cam_high_M\d+_area[0-9.]+)$/);
  if (!m) return out;
  out.folder_model = m[1];
  out.folder_seed = Number(m[2]);
  const rest = m[3];
  if (rest === 'none') {
    out.folder_cutout_mode = 'none';
    out.folder_cutout_m = 0;
    return out;
  }
  const m2 = rest.match(/^(random|cam_low|cam_high)_M(\d+)_area([0-9.]+)$/);
  if (m2) {
    out.folder_cutout_mode = m2[1];
    out.folder_cutout_m = Number(m2[2]);
    out.folder_cutout_area = safeFloat(m2[3]);
  }
  return out;
}
function pickColumn(columns, candidates, kind) {
  const lower = new Map(columns.map(c => [c.toLowerCase(), c]));
  for (const cand of candidates) if (lower.has(cand.toLowerCase())) return lower.get(cand.toLowerCase());
  for (const col of columns) {
    const low = col.toLowerCase();
    if (kind === 'train_acc' && low.includes('train') && low.includes('acc')) return col;
    if (kind === 'eval_acc' && ['eval', 'val', 'valid', 'test'].some(x => low.includes(x)) && low.includes('acc')) return col;
    if (kind === 'train_loss' && low.includes('train') && low.includes('loss')) return col;
    if (kind === 'eval_loss' && ['eval', 'val', 'valid', 'test'].some(x => low.includes(x)) && low.includes('loss')) return col;
  }
  return '';
}
function numericSeries(rows, column) {
  if (!column) return { values: [], bad: 0 };
  let bad = 0;
  const values = rows.map(row => {
    const raw = row[column];
    if (raw === undefined || String(raw).trim() === '') { bad++; return null; }
    const f = safeFloat(raw);
    if (f === null) bad++;
    return f;
  });
  return { values, bad };
}
function normalizeAccuracy(values) {
  const finite = values.filter(v => v !== null);
  if (finite.length && Math.max(...finite) > 1.5) return { values: values.map(v => v === null ? null : v / 100), unit: 'percent_to_fraction' };
  return { values, unit: 'fraction' };
}
function maxWithIndex(values) {
  let best = null; let bestIdx = null;
  values.forEach((v, i) => { if (v !== null && (best === null || v > best)) { best = v; bestIdx = i; } });
  return { value: best, index: bestIdx };
}
function minWithIndex(values) {
  let best = null; let bestIdx = null;
  values.forEach((v, i) => { if (v !== null && (best === null || v < best)) { best = v; bestIdx = i; } });
  return { value: best, index: bestIdx };
}
function lastWithIndex(values) {
  for (let i = values.length - 1; i >= 0; i--) if (values[i] !== null) return { value: values[i], index: i };
  return { value: null, index: null };
}
function epochAt(epochs, index) { return index === null || index < 0 || index >= epochs.length ? null : epochs[index]; }
function numericMatrix(rows, columns) {
  const numericCols = columns.filter(col => rows.some(row => safeFloat(row[col]) !== null));
  return { columns: numericCols, matrix: rows.map(row => numericCols.map(col => safeFloat(row[col]))) };
}
function matricesEqual(a, b, tol) {
  const epsilon = tol || 1e-12;
  if (!a || !b) return false;
  if (JSON.stringify(a.columns || []) !== JSON.stringify(b.columns || [])) return false;
  const ma = a.matrix || []; const mb = b.matrix || [];
  if (ma.length !== mb.length) return false;
  for (let i = 0; i < ma.length; i++) {
    if (ma[i].length !== mb[i].length) return false;
    for (let j = 0; j < ma[i].length; j++) {
      const va = ma[i][j]; const vb = mb[i][j];
      if (va === null && vb === null) continue;
      if (va === null || vb === null) return false;
      if (Math.abs(va - vb) > epsilon) return false;
    }
  }
  return true;
}
async function inspectCheckpoint(filePath) {
  const info = { best_model_bytes: '', best_model_sha256: '', best_model_metadata_status: 'absent', best_model_metadata: '' };
  if (!exists(filePath)) return info;
  const st = await fs.stat(filePath);
  info.best_model_bytes = st.size;
  info.best_model_sha256 = await sha256File(filePath);
  info.best_model_metadata_status = 'file_hashed_only_pickle_not_loaded';
  return info;
}
async function summarizeRun(datasetFolder, modelFolder, runDir) {
  const runName = path.basename(runDir);
  const parsed = parseRunName(runName);
  const configPath = path.join(runDir, 'config.json');
  const metricsPath = path.join(runDir, 'metrics.csv');
  const plotPath = path.join(runDir, 'metrics_plot.png');
  const checkpointPath = path.join(runDir, 'best_model.pt');
  let config = {}; let configError = '';
  if (exists(configPath)) {
    const read = await readJSON(configPath);
    config = read.value || {};
    configError = read.error;
  }
  let metricRows = []; let metricColumns = []; let metricsError = '';
  if (exists(metricsPath)) {
    try {
      const parsedCsv = parseCSV(await fs.readFile(metricsPath, 'utf8'));
      metricRows = parsedCsv.records;
      metricColumns = parsedCsv.headers;
    } catch (err) { metricsError = String(err.message || err); }
  }
  const logFiles = exists(runDir) ? (await fs.readdir(runDir)).filter(n => n.toLowerCase().endsWith('.log')) : [];
  const cutoutMode = normMode(config.cutout_mode !== undefined ? config.cutout_mode : parsed.folder_cutout_mode);
  const cutoutM = safeInt(config.cutout_m !== undefined ? config.cutout_m : parsed.folder_cutout_m);
  const cutoutArea = config.cutout_area !== undefined ? config.cutout_area : parsed.folder_cutout_area;
  const seed = safeInt(config.seed !== undefined ? config.seed : parsed.folder_seed);
  const epochSeries = numericSeries(metricRows, metricColumns.includes('epoch') ? 'epoch' : '');
  const epochs = epochSeries.values.length ? epochSeries.values.map((v, i) => v === null ? i + 1 : Math.trunc(v)) : metricRows.map((_, i) => i + 1);
  const trainLossCol = pickColumn(metricColumns, TRAIN_LOSS_CANDIDATES, 'train_loss');
  const evalLossCol = pickColumn(metricColumns, EVAL_LOSS_CANDIDATES, 'eval_loss');
  const trainAccCol = pickColumn(metricColumns, TRAIN_ACC_CANDIDATES, 'train_acc');
  const evalAccCol = pickColumn(metricColumns, EVAL_ACC_CANDIDATES, 'eval_acc');
  const trainLoss = numericSeries(metricRows, trainLossCol);
  const evalLoss = numericSeries(metricRows, evalLossCol);
  const trainAccRaw = numericSeries(metricRows, trainAccCol);
  const evalAccRaw = numericSeries(metricRows, evalAccCol);
  const trainAccNorm = normalizeAccuracy(trainAccRaw.values);
  const evalAccNorm = normalizeAccuracy(evalAccRaw.values);
  const trainAcc = trainAccNorm.values;
  const evalAcc = evalAccNorm.values;
  const finalTrainLoss = lastWithIndex(trainLoss.values);
  const finalEvalLoss = lastWithIndex(evalLoss.values);
  const finalTrainAcc = lastWithIndex(trainAcc);
  const finalEvalAcc = lastWithIndex(evalAcc);
  const bestTrainLoss = minWithIndex(trainLoss.values);
  const bestEvalLoss = minWithIndex(evalLoss.values);
  const bestTrainAcc = maxWithIndex(trainAcc);
  const bestEvalAcc = maxWithIndex(evalAcc);
  const badDetails = { epoch: epochSeries.bad, train_loss: trainLoss.bad, eval_loss: evalLoss.bad, train_acc: trainAccRaw.bad, eval_acc: evalAccRaw.bad };
  const badTotal = Object.values(badDetails).reduce((a, b) => a + b, 0);
  const checkpointInfo = await inspectCheckpoint(checkpointPath);
  const notes = [];
  if (configError) notes.push('config read error: ' + configError);
  if (metricsError) notes.push('metrics read error: ' + metricsError);
  if (trainAccNorm.unit === 'percent_to_fraction' || evalAccNorm.unit === 'percent_to_fraction') notes.push('accuracy normalized from percent to fraction');
  if (!metricRows.length) notes.push('no metric rows');
  const run = {
    id: rel(runDir), dataset_folder: datasetFolder, model_folder: modelFolder, run_name: runName, relative_path: rel(runDir),
    config_dataset: String(config.dataset || ''), normalized_dataset: normDataset(config.dataset || datasetFolder), config_model: String(config.model || ''),
    seed: seed, cutout_mode: cutoutMode, cutout_m: cutoutM, cutout_area: cutoutArea, condition: conditionLabel(cutoutMode, cutoutM), configured_epochs: safeInt(config.epochs),
    observed_metric_epochs: metricRows.length, first_epoch: epochs.length ? epochs[0] : '', last_epoch: epochs.length ? epochs[epochs.length - 1] : '',
    grayscale: config.grayscale === undefined ? '' : config.grayscale, include_regex: config.include_regex || '', teacher_checkpoint: config.teacher_checkpoint || '', teacher_model: config.teacher_model || '',
    config_exists: exists(configPath), metrics_exists: exists(metricsPath), metrics_plot_exists: exists(plotPath), best_model_exists: exists(checkpointPath), log_exists: logFiles.length > 0, log_count: logFiles.length,
    metrics_columns: metricColumns.join('|'), metrics_hash: exists(metricsPath) ? await sha256File(metricsPath) : '', config_hash: exists(configPath) ? await sha256File(configPath) : '',
    folder_model: parsed.folder_model, folder_seed: parsed.folder_seed, folder_cutout_mode: parsed.folder_cutout_mode, folder_cutout_m: parsed.folder_cutout_m, folder_cutout_area: parsed.folder_cutout_area,
    train_loss_column: trainLossCol, train_accuracy_column: trainAccCol, eval_loss_column: evalLossCol, eval_accuracy_column: evalAccCol,
    eval_split: metricRows.length && metricColumns.includes('eval_split') ? metricRows[metricRows.length - 1].eval_split : '', bad_metric_values: badTotal, bad_metric_value_detail: JSON.stringify(badDetails),
    status: 'ok', notes: notes.join('; '), metric_rows: metricRows, metric_columns: metricColumns, epochs: epochs, numeric_matrix: numericMatrix(metricRows, metricColumns),
    series: { train_loss: trainLoss.values, eval_loss: evalLoss.values, train_acc: trainAcc, eval_acc: evalAcc },
    final_train_loss: finalTrainLoss.value, final_train_accuracy: finalTrainAcc.value, final_eval_loss: finalEvalLoss.value, final_eval_accuracy: finalEvalAcc.value,
    best_train_loss: bestTrainLoss.value, best_train_loss_epoch: epochAt(epochs, bestTrainLoss.index), best_train_accuracy: bestTrainAcc.value, best_train_accuracy_epoch: epochAt(epochs, bestTrainAcc.index),
    best_eval_loss: bestEvalLoss.value, best_eval_loss_epoch: epochAt(epochs, bestEvalLoss.index), best_eval_accuracy: bestEvalAcc.value, best_epoch: epochAt(epochs, bestEvalAcc.index),
    final_epoch: epochAt(epochs, finalEvalAcc.index), final_train_loss_epoch: epochAt(epochs, finalTrainLoss.index), final_eval_loss_epoch: epochAt(epochs, finalEvalLoss.index),
    final_train_accuracy_epoch: epochAt(epochs, finalTrainAcc.index), final_eval_accuracy_epoch: epochAt(epochs, finalEvalAcc.index),
    generalization_gap: finalTrainAcc.value !== null && finalEvalAcc.value !== null ? finalTrainAcc.value - finalEvalAcc.value : null,
    baseline_run_name: '', improvement_over_none: null, relative_improvement_over_none_pct: null, random_baseline_run_name: '', improvement_over_random: null, relative_improvement_over_random_pct: null,
    ...checkpointInfo
  };
  return run;
}
async function discoverRuns() {
  const runs = [];
  for (const dataset of DATASET_FOLDERS) {
    const datasetDir = path.join(RUNS_ROOT, dataset);
    if (!exists(datasetDir)) continue;
    const modelNames = (await fs.readdir(datasetDir, { withFileTypes: true })).filter(d => d.isDirectory()).map(d => d.name).sort();
    for (const model of modelNames) {
      const modelDir = path.join(datasetDir, model);
      const runNames = (await fs.readdir(modelDir, { withFileTypes: true })).filter(d => d.isDirectory()).map(d => d.name).sort();
      for (const runName of runNames) {
        const runDir = path.join(modelDir, runName);
        const names = await fs.readdir(runDir).catch(() => []);
        const hasArtifact = ['config.json', 'metrics.csv', 'best_model.pt'].some(n => names.includes(n)) || names.some(n => n.toLowerCase().endsWith('.log'));
        if (hasArtifact) runs.push(await summarizeRun(dataset, model, runDir));
      }
    }
  }
  return runs;
}
function addCheck(checks, checkType, severity, status, run, details, relatedRun, overrides) {
  overrides = overrides || {};
  checks.push({
    check_type: checkType, severity: severity, status: status,
    dataset_folder: overrides.dataset_folder || (run ? run.dataset_folder : ''), model_folder: overrides.model_folder || (run ? run.model_folder : ''), seed: overrides.seed !== undefined ? overrides.seed : (run ? run.seed : ''),
    run_name: run ? run.run_name : '', condition: run ? run.condition : '', related_run: relatedRun || '', details: details || '', run_id: run ? run.id : ''
  });
}
function buildIntegrityChecks(runs) {
  const checks = [];
  for (const run of runs) {
    if (!run.config_exists) addCheck(checks, 'missing_config', 'error', 'fail', run, 'config.json missing');
    if (!run.metrics_exists) addCheck(checks, 'missing_metrics', 'error', 'fail', run, 'metrics.csv missing');
    const configured = safeInt(run.configured_epochs); const observed = safeInt(run.observed_metric_epochs);
    if (configured !== null && observed !== null && configured !== observed) {
      if (run.dataset_folder === 'malimg' && observed < configured) addCheck(checks, 'configured_vs_observed_epochs', 'expected_possible', 'expected_possible', run, 'configured_epochs=' + configured + ', observed_metric_rows=' + observed + '; MalImg short runs may be expected');
      else addCheck(checks, 'configured_vs_observed_epochs', 'warning', 'warning', run, 'configured_epochs=' + configured + ', observed_metric_rows=' + observed);
    }
    if ((run.dataset_folder === 'rawmaltf' || run.dataset_folder === 'cifar100') && (configured !== 100 || observed !== 100)) addCheck(checks, 'rawmaltf_cifar_not_100_epochs', 'critical', 'fail', run, run.dataset_folder + ' run does not appear to be 100 epochs: configured=' + configured + ', observed=' + observed);
    if (run.dataset_folder === 'malimg' && observed !== null && observed < 100) addCheck(checks, 'malimg_short_run', 'expected_possible', 'expected_possible', run, 'MalImg observed_metric_rows=' + observed);
    if (safeInt(run.bad_metric_values)) addCheck(checks, 'nan_or_non_numeric_metrics', 'warning', 'warning', run, 'bad_metric_values=' + run.bad_metric_value_detail);
    const evalAcc = run.series.eval_acc.filter(v => v !== null);
    if (evalAcc.length > 3 && new Set(evalAcc.map(v => v.toFixed(12))).size <= 1) addCheck(checks, 'suspiciously_constant_accuracy', 'warning', 'warning', run, 'eval accuracy is constant');
    const trainAcc = run.series.train_acc.filter(v => v !== null);
    if (trainAcc.length > 3 && new Set(trainAcc.map(v => v.toFixed(12))).size <= 1) addCheck(checks, 'suspiciously_constant_accuracy', 'warning', 'warning', run, 'train accuracy is constant');
    if (run.config_dataset && normDataset(run.config_dataset) !== normDataset(run.dataset_folder)) addCheck(checks, 'folder_config_dataset_mismatch', 'warning', 'warning', run, 'folder=' + run.dataset_folder + ', config_dataset=' + run.config_dataset);
    if (run.config_model && String(run.config_model) !== String(run.model_folder)) addCheck(checks, 'folder_config_model_mismatch', 'warning', 'warning', run, 'folder=' + run.model_folder + ', config_model=' + run.config_model);
    if (run.folder_cutout_mode && normMode(run.folder_cutout_mode) !== normMode(run.cutout_mode)) addCheck(checks, 'folder_config_cutout_mode_mismatch', 'warning', 'warning', run, 'folder=' + run.folder_cutout_mode + ', config=' + run.cutout_mode);
    if (normMode(run.cutout_mode).startsWith('cam') && !String(run.teacher_checkpoint || '').trim()) addCheck(checks, 'cam_missing_teacher_checkpoint', 'critical', 'fail', run, 'CAM run has no teacher_checkpoint in config');
  }
  for (const hashField of ['metrics_hash', 'config_hash']) {
    const groups = new Map();
    for (const run of runs) {
      const h = run[hashField];
      if (!h) continue;
      if (!groups.has(h)) groups.set(h, []);
      groups.get(h).push(run);
    }
    for (const [h, members] of groups) {
      if (members.length > 1) {
        const names = members.map(m => m.relative_path).join(', ');
        const severity = hashField === 'metrics_hash' ? 'critical' : 'warning';
        for (const member of members) addCheck(checks, 'duplicate_' + hashField, severity, severity === 'critical' ? 'fail' : 'warning', member, hashField + '=' + h + '; duplicate among: ' + names);
      }
    }
  }
  const pairGroups = new Map();
  for (const run of runs) {
    const mode = normMode(run.cutout_mode);
    if (mode !== 'cam_low' && mode !== 'cam_high') continue;
    const key = [run.dataset_folder, run.model_folder, run.seed, run.cutout_m, compareArea(run.cutout_area)].join('|');
    if (!pairGroups.has(key)) pairGroups.set(key, { cam_low: [], cam_high: [], keyParts: [run.dataset_folder, run.model_folder, run.seed, run.cutout_m, compareArea(run.cutout_area)] });
    pairGroups.get(key)[mode].push(run);
  }
  for (const group of pairGroups.values()) {
    for (const low of group.cam_low) for (const high of group.cam_high) {
      const sameHash = !!low.metrics_hash && low.metrics_hash === high.metrics_hash;
      const sameNumeric = matricesEqual(low.numeric_matrix, high.numeric_matrix);
      if (sameHash || sameNumeric) addCheck(checks, 'identical_cam_low_high_metrics', 'critical', 'fail', low, 'cam_low=' + low.run_name + ' and cam_high=' + high.run_name + ' same_raw_csv_hash=' + sameHash + ', same_numeric_arrays=' + sameNumeric, high.run_name, { dataset_folder: group.keyParts[0], model_folder: group.keyParts[1], seed: group.keyParts[2] });
    }
  }
  return checks;
}
function applyStatuses(runs, checks) {
  const byId = new Map();
  for (const c of checks) {
    if (!c.run_id) continue;
    if (!byId.has(c.run_id)) byId.set(c.run_id, []);
    byId.get(c.run_id).push(c);
  }
  for (const run of runs) {
    const list = byId.get(run.id) || [];
    const severities = new Set(list.map(c => c.severity));
    if (severities.has('critical') || severities.has('error')) run.status = 'suspicious';
    else if (severities.has('warning')) run.status = 'warning';
    else if (severities.has('expected_possible')) run.status = 'expected_possible';
    else run.status = 'ok';
    const important = [...new Set(list.filter(c => ['critical', 'error', 'warning'].includes(c.severity)).map(c => c.check_type))].sort();
    if (important.length) run.notes = [run.notes, 'integrity: ' + important.join('|')].filter(Boolean).join('; ');
  }
}
function addImprovements(runs) {
  const baselines = new Map();
  const randoms = new Map();
  for (const run of runs) {
    const baseKey = [run.dataset_folder, run.model_folder, run.seed].join('|');
    if (normMode(run.cutout_mode) === 'none') {
      const current = baselines.get(baseKey);
      if (!current || (run.best_eval_accuracy || -1) > (current.best_eval_accuracy || -1)) baselines.set(baseKey, run);
    }
    if (normMode(run.cutout_mode) === 'random') {
      const rKey = [run.dataset_folder, run.model_folder, run.seed, run.cutout_m, compareArea(run.cutout_area)].join('|');
      const current = randoms.get(rKey);
      if (!current || (run.best_eval_accuracy || -1) > (current.best_eval_accuracy || -1)) randoms.set(rKey, run);
    }
  }
  for (const run of runs) {
    const baseKey = [run.dataset_folder, run.model_folder, run.seed].join('|');
    const base = baselines.get(baseKey);
    if (base && run.best_eval_accuracy !== null && base.best_eval_accuracy !== null) {
      const diff = run.best_eval_accuracy - base.best_eval_accuracy;
      run.baseline_run_name = base.run_name;
      run.improvement_over_none = diff;
      run.relative_improvement_over_none_pct = base.best_eval_accuracy ? diff / Math.abs(base.best_eval_accuracy) * 100 : null;
    }
    if (normMode(run.cutout_mode).startsWith('cam')) {
      const rKey = [run.dataset_folder, run.model_folder, run.seed, run.cutout_m, compareArea(run.cutout_area)].join('|');
      const random = randoms.get(rKey);
      if (random && run.best_eval_accuracy !== null && random.best_eval_accuracy !== null) {
        const diff = run.best_eval_accuracy - random.best_eval_accuracy;
        run.random_baseline_run_name = random.run_name;
        run.improvement_over_random = diff;
        run.relative_improvement_over_random_pct = random.best_eval_accuracy ? diff / Math.abs(random.best_eval_accuracy) * 100 : null;
      }
    }
  }
}
function makeComparisonTable(runs) {
  const groups = new Map();
  for (const run of runs) {
    const key = [run.dataset_folder, run.model_folder, run.seed].join('|');
    if (!groups.has(key)) groups.set(key, []);
    groups.get(key).push(run);
  }
  const rows = [];
  for (const [key, members] of [...groups.entries()].sort()) {
    const parts = key.split('|');
    const row = { dataset_folder: parts[0], model_folder: parts[1], seed: parts[2] };
    const baseline = members.find(m => normMode(m.cutout_mode) === 'none');
    row.baseline_run_name = baseline ? baseline.run_name : '';
    row.baseline_best_accuracy = baseline ? baseline.best_eval_accuracy : null;
    for (const [mode, mValue] of CONDITION_ORDER) {
      const label = conditionLabel(mode, mValue);
      const slug = conditionSlug(label);
      const match = members.find(member => normMode(member.cutout_mode) === mode && (mode === 'none' || safeInt(member.cutout_m) === mValue));
      row[slug + '_run_name'] = match ? match.run_name : '';
      row[slug + '_best_accuracy'] = match ? match.best_eval_accuracy : null;
      row[slug + '_final_accuracy'] = match ? match.final_eval_accuracy : null;
      row[slug + '_best_epoch'] = match ? match.best_epoch : null;
      row[slug + '_improvement_over_none'] = match ? match.improvement_over_none : null;
    }
    rows.push(row);
  }
  return rows;
}
function bestRunDict(run) {
  if (!run) return {};
  return { dataset_folder: run.dataset_folder, model_folder: run.model_folder, run_name: run.run_name, condition: run.condition, seed: run.seed, best_eval_accuracy: run.best_eval_accuracy, final_eval_accuracy: run.final_eval_accuracy, best_epoch: run.best_epoch, status: run.status };
}
function makeSummaryStats(runs, checks) {
  const sortable = runs.filter(r => r.best_eval_accuracy !== null).sort((a, b) => b.best_eval_accuracy - a.best_eval_accuracy);
  const perDataset = {};
  for (const dataset of [...new Set(runs.map(r => r.dataset_folder))].sort()) {
    const members = runs.filter(r => r.dataset_folder === dataset);
    const best = members.reduce((a, b) => (b.best_eval_accuracy || -1) > (a.best_eval_accuracy || -1) ? b : a, members[0]);
    const statusCounts = {};
    for (const m of members) statusCounts[m.status] = (statusCounts[m.status] || 0) + 1;
    perDataset[dataset] = { run_count: members.length, status_counts: statusCounts, best_run: bestRunDict(best) };
  }
  const perModel = {};
  for (const model of [...new Set(runs.map(r => r.model_folder))].sort()) {
    const members = runs.filter(r => r.model_folder === model);
    const best = members.reduce((a, b) => (b.best_eval_accuracy || -1) > (a.best_eval_accuracy || -1) ? b : a, members[0]);
    perModel[model] = { run_count: members.length, best_run: bestRunDict(best) };
  }
  const nested = {};
  for (const dataset of Object.keys(perDataset)) {
    nested[dataset] = {};
    for (const model of [...new Set(runs.filter(r => r.dataset_folder === dataset).map(r => r.model_folder))].sort()) {
      const members = runs.filter(r => r.dataset_folder === dataset && r.model_folder === model);
      const best = members.reduce((a, b) => (b.best_eval_accuracy || -1) > (a.best_eval_accuracy || -1) ? b : a, members[0]);
      nested[dataset][model] = { run_count: members.length, best_run: bestRunDict(best) };
    }
  }
  const pairGroups = new Map();
  for (const run of runs) {
    const mode = normMode(run.cutout_mode);
    if (mode !== 'cam_low' && mode !== 'cam_high') continue;
    const key = [run.dataset_folder, run.model_folder, run.seed, run.cutout_m, compareArea(run.cutout_area)].join('|');
    if (!pairGroups.has(key)) pairGroups.set(key, {});
    pairGroups.get(key)[mode] = run;
  }
  const lowHighDiffs = [];
  let lowBetter = 0; let highBetter = 0; let ties = 0; let identical = 0;
  for (const pair of pairGroups.values()) {
    if (!pair.cam_low || !pair.cam_high) continue;
    if (pair.cam_low.best_eval_accuracy === null || pair.cam_high.best_eval_accuracy === null) continue;
    const diff = pair.cam_low.best_eval_accuracy - pair.cam_high.best_eval_accuracy;
    lowHighDiffs.push(diff);
    if (Math.abs(diff) <= 1e-12) ties++; else if (diff > 0) lowBetter++; else highBetter++;
    if (pair.cam_low.metrics_hash === pair.cam_high.metrics_hash || matricesEqual(pair.cam_low.numeric_matrix, pair.cam_high.numeric_matrix)) identical++;
  }
  const statusCounts = {};
  for (const run of runs) statusCounts[run.status] = (statusCounts[run.status] || 0) + 1;
  const majorWarnings = checks.filter(c => ['critical', 'error'].includes(c.severity) || c.check_type === 'identical_cam_low_high_metrics').map(c => c.details);
  return {
    total_runs: runs.length,
    successful_count: runs.filter(r => r.status === 'ok').length,
    suspicious_count: runs.filter(r => r.status === 'suspicious').length,
    status_counts: statusCounts,
    integrity_check_count: checks.length,
    major_warning_count: majorWarnings.length,
    per_dataset: perDataset,
    per_model: perModel,
    per_dataset_per_model: nested,
    best_runs: sortable.slice(0, 10).map(bestRunDict),
    best_rawmaltf_run: bestRunDict(sortable.find(r => r.dataset_folder === 'rawmaltf')),
    improvement_statistics_over_no_cutout: stats(runs.filter(r => normMode(r.cutout_mode) !== 'none').map(r => r.improvement_over_none)),
    random_vs_cam_statistics: stats(runs.filter(r => normMode(r.cutout_mode).startsWith('cam')).map(r => r.improvement_over_random)),
    cam_low_vs_cam_high_statistics: { ...stats(lowHighDiffs), low_better_count: lowBetter, high_better_count: highBetter, tie_count: ties, identical_metric_pair_count: identical },
    major_warnings: majorWarnings
  };
}
function publicRun(run) {
  const copy = { ...run };
  delete copy.metric_rows; delete copy.metric_columns; delete copy.epochs; delete copy.numeric_matrix; delete copy.series; delete copy.id;
  return copy;
}
function markdownTable(rows, cols, limit) {
  if (!rows.length) return '_No rows available._';
  const clipped = rows.slice(0, limit || 20);
  const lines = [];
  lines.push('| ' + cols.map(c => c[0]).join(' | ') + ' |');
  lines.push('| ' + cols.map(() => '---').join(' | ') + ' |');
  for (const row of clipped) {
    lines.push('| ' + cols.map(c => {
      const key = c[1];
      const value = row[key];
      if (key.includes('accuracy') || key.includes('improvement') || key === 'generalization_gap') return fmtPct(value);
      return value === null || value === undefined ? '' : String(value).replace(/\|/g, '/');
    }).join(' | ') + ' |');
  }
  if (rows.length > clipped.length) lines.push('', '_Showing ' + clipped.length + ' of ' + rows.length + ' rows._');
  return lines.join('\n');
}
function makePaperSummary(runs, checks, summaryStats) {
  const bestRows = runs.filter(r => r.best_eval_accuracy !== null).sort((a, b) => b.best_eval_accuracy - a.best_eval_accuracy);
  const rawRows = bestRows.filter(r => r.dataset_folder === 'rawmaltf');
  const malRows = bestRows.filter(r => r.dataset_folder === 'malimg');
  const cifarRows = bestRows.filter(r => r.dataset_folder === 'cifar100');
  const critical = checks.filter(c => ['critical', 'error'].includes(c.severity));
  const identical = checks.filter(c => c.check_type === 'identical_cam_low_high_metrics');
  const lines = [];
  lines.push('# CAM-Regularization Run Summary', '');
  lines.push('## Research Context', '');
  lines.push('This package summarizes existing artifacts for CAM-guided cutout augmentation in image-based malware classification. It compares no cutout (none), standard random cutout (random), low-saliency CAM-guided cutout (cam_low), and high-saliency CAM-guided cutout (cam_high). RawMal-TF appears in configs as drive_zip and is treated as the primary publication dataset; CIFAR100 is a sanity check, and MalImg is secondary malware evidence.', '');
  lines.push('All conclusions below are computed only from files already present under runs/cifar100/, runs/malimg/, and runs/rawmaltf/.', '');
  lines.push('## Inventory', '');
  lines.push('- Total runs processed: ' + runs.length);
  lines.push('- Status counts: ' + JSON.stringify(summaryStats.status_counts));
  lines.push('- Integrity checks emitted: ' + checks.length, '');
  lines.push('Datasets/models found:');
  for (const dataset of Object.keys(summaryStats.per_dataset_per_model).sort()) {
    const models = summaryStats.per_dataset_per_model[dataset];
    lines.push('- ' + dataset + ': ' + Object.keys(models).sort().map(m => m + ' (' + models[m].run_count + ' runs)').join(', '));
  }
  if (identical.length) {
    lines.push('', '## Publication-Critical Warning', '');
    lines.push('**At least one cam_low/cam_high pair has identical metric arrays or identical raw metric CSV hashes. Do not claim low- and high-saliency CAM behavior differs for those pairs.**', '');
    lines.push(markdownTable(identical, [['Dataset', 'dataset_folder'], ['Model', 'model_folder'], ['Seed', 'seed'], ['Run', 'run_name'], ['Related run', 'related_run'], ['Details', 'details']], 12));
  }
  lines.push('', '## Best Runs Overall', '');
  lines.push(markdownTable(bestRows, [['Dataset', 'dataset_folder'], ['Model', 'model_folder'], ['Run', 'run_name'], ['Condition', 'condition'], ['Best acc', 'best_eval_accuracy'], ['Final acc', 'final_eval_accuracy'], ['Best epoch', 'best_epoch'], ['Status', 'status']], 15));
  lines.push('', '## RawMal-TF Focus', '');
  lines.push(markdownTable(rawRows, [['Model', 'model_folder'], ['Run', 'run_name'], ['Condition', 'condition'], ['Best acc', 'best_eval_accuracy'], ['Final acc', 'final_eval_accuracy'], ['Best epoch', 'best_epoch'], ['Vs none', 'improvement_over_none'], ['Vs random', 'improvement_over_random'], ['Status', 'status']], 20));
  lines.push('', 'Interpretation for RawMal-TF should prioritize the computed improvement columns. A CAM method should be described as better than random only where Vs random is positive for the matching model, seed, M, and area.', '');
  lines.push('## MalImg Summary', '');
  lines.push(markdownTable(malRows, [['Model', 'model_folder'], ['Run', 'run_name'], ['Condition', 'condition'], ['Best acc', 'best_eval_accuracy'], ['Best epoch', 'best_epoch'], ['Vs none', 'improvement_over_none'], ['Status', 'status']], 20));
  lines.push('', 'MalImg short runs are marked expected_possible in integrity checks because some MalImg runs may have 20 observed epochs.', '');
  lines.push('## CIFAR100 Sanity Check', '');
  lines.push(markdownTable(cifarRows, [['Model', 'model_folder'], ['Run', 'run_name'], ['Condition', 'condition'], ['Best acc', 'best_eval_accuracy'], ['Best epoch', 'best_epoch'], ['Vs none', 'improvement_over_none'], ['Status', 'status']], 20));
  lines.push('', '## Aggregate Statistics', '');
  lines.push('- Improvement over no-cutout: ' + JSON.stringify(summaryStats.improvement_statistics_over_no_cutout));
  lines.push('- CAM vs matching random: ' + JSON.stringify(summaryStats.random_vs_cam_statistics));
  lines.push('- CAM low vs high: ' + JSON.stringify(summaryStats.cam_low_vs_cam_high_statistics));
  lines.push('', '## Warnings', '');
  lines.push(critical.length ? markdownTable(critical, [['Check', 'check_type'], ['Severity', 'severity'], ['Dataset', 'dataset_folder'], ['Model', 'model_folder'], ['Run', 'run_name'], ['Details', 'details']], 30) : 'No critical integrity warnings were detected.');
  lines.push('', '## Next-Step Recommendations', '');
  lines.push('- Treat RawMal-TF / drive_zip grayscale-only results as the main publication evidence.');
  lines.push('- Before making a low-vs-high saliency claim, resolve any identical cam_low and cam_high metric warnings.');
  lines.push('- For any run with fewer than the expected 100 epochs outside MalImg, rerun or exclude it from headline comparisons.');
  lines.push('- Use comparison_table.csv for paper tables and integrity_checks.csv for audit notes.', '');
  return lines.join('\n');
}
function svgEscape(text) { return String(text === null || text === undefined ? '' : text).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;'); }
function niceRange(values, padFrac) {
  const vals = values.filter(v => v !== null && Number.isFinite(v));
  if (!vals.length) return { min: 0, max: 1 };
  let min = Math.min(...vals); let max = Math.max(...vals);
  if (min === max) { min -= 0.05; max += 0.05; }
  const pad = (max - min) * (padFrac || 0.08);
  return { min: Math.max(0, min - pad), max: max + pad };
}
async function renderSvg(name, svg) {
  if (!sharp) return false;
  await sharp(Buffer.from(svg)).png().toFile(path.join(PLOTS_DIR, name + '.png'));
  return true;
}
function lineChartSvg(title, rows, xKey, yKey, groupKey, yLabel) {
  const width = 980; const height = 560; const m = { l: 68, r: 220, t: 54, b: 68 };
  const xVals = rows.map(r => safeFloat(r[xKey])).filter(v => v !== null);
  const yVals = rows.map(r => safeFloat(r[yKey])).filter(v => v !== null);
  const xMin = xVals.length ? Math.min(...xVals) : 0; const xMax = xVals.length ? Math.max(...xVals) : 1;
  const yr = niceRange(yVals, 0.08);
  const sx = x => m.l + (xMax === xMin ? 0 : (x - xMin) / (xMax - xMin)) * (width - m.l - m.r);
  const sy = y => height - m.b - (yr.max === yr.min ? 0 : (y - yr.min) / (yr.max - yr.min)) * (height - m.t - m.b);
  const groups = [...new Set(rows.map(r => r[groupKey]))];
  let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"><rect width="100%" height="100%" fill="#ffffff"/>';
  svg += '<text x="' + m.l + '" y="30" font-family="Arial" font-size="22" font-weight="700" fill="#111827">' + svgEscape(title) + '</text>';
  for (let i = 0; i <= 5; i++) {
    const y = m.t + i * (height - m.t - m.b) / 5;
    const val = yr.max - i * (yr.max - yr.min) / 5;
    svg += '<line x1="' + m.l + '" x2="' + (width - m.r) + '" y1="' + y + '" y2="' + y + '" stroke="#e5e7eb"/>';
    svg += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">' + svgEscape(val.toFixed(3)) + '</text>';
  }
  svg += '<line x1="' + m.l + '" x2="' + (width - m.r) + '" y1="' + (height - m.b) + '" y2="' + (height - m.b) + '" stroke="#111827"/>';
  svg += '<line x1="' + m.l + '" x2="' + m.l + '" y1="' + m.t + '" y2="' + (height - m.b) + '" stroke="#111827"/>';
  svg += '<text x="' + ((m.l + width - m.r) / 2) + '" y="' + (height - 20) + '" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">Epoch</text>';
  svg += '<text x="20" y="' + ((m.t + height - m.b) / 2) + '" transform="rotate(-90 20 ' + ((m.t + height - m.b) / 2) + ')" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">' + svgEscape(yLabel) + '</text>';
  groups.forEach((g, idx) => {
    const pts = rows.filter(r => r[groupKey] === g && safeFloat(r[yKey]) !== null && safeFloat(r[xKey]) !== null).sort((a, b) => safeFloat(a[xKey]) - safeFloat(b[xKey]));
    const color = PALETTE[g] || ['#7c3aed', '#0891b2', '#ea580c', '#16a34a', '#be123c', '#4f46e5'][idx % 6];
    const points = pts.map(r => sx(safeFloat(r[xKey])).toFixed(1) + ',' + sy(safeFloat(r[yKey])).toFixed(1)).join(' ');
    if (points) svg += '<polyline points="' + points + '" fill="none" stroke="' + color + '" stroke-width="2"/>';
    const ly = m.t + 20 + idx * 20;
    svg += '<line x1="' + (width - m.r + 25) + '" x2="' + (width - m.r + 45) + '" y1="' + ly + '" y2="' + ly + '" stroke="' + color + '" stroke-width="3"/>';
    svg += '<text x="' + (width - m.r + 52) + '" y="' + (ly + 4) + '" font-family="Arial" font-size="12" fill="#111827">' + svgEscape(g) + '</text>';
  });
  svg += '</svg>';
  return svg;
}
function barChartSvg(title, rows, labelFn, valueKey, yLabel, horizontalZero) {
  const width = Math.max(980, rows.length * 54 + 220); const height = 610; const m = { l: 70, r: 35, t: 58, b: 170 };
  const vals = rows.map(r => safeFloat(r[valueKey])).filter(v => v !== null);
  let yr = niceRange(vals.concat(horizontalZero ? [0] : []), 0.12);
  if (horizontalZero) { yr.min = Math.min(yr.min, 0); yr.max = Math.max(yr.max, 0); }
  const sy = y => height - m.b - (yr.max === yr.min ? 0 : (y - yr.min) / (yr.max - yr.min)) * (height - m.t - m.b);
  const barW = (width - m.l - m.r) / Math.max(rows.length, 1) * 0.72;
  let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"><rect width="100%" height="100%" fill="#ffffff"/>';
  svg += '<text x="' + m.l + '" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#111827">' + svgEscape(title) + '</text>';
  for (let i = 0; i <= 5; i++) {
    const y = m.t + i * (height - m.t - m.b) / 5;
    const val = yr.max - i * (yr.max - yr.min) / 5;
    svg += '<line x1="' + m.l + '" x2="' + (width - m.r) + '" y1="' + y + '" y2="' + y + '" stroke="#e5e7eb"/>';
    svg += '<text x="' + (m.l - 8) + '" y="' + (y + 4) + '" text-anchor="end" font-family="Arial" font-size="12" fill="#4b5563">' + svgEscape(val.toFixed(3)) + '</text>';
  }
  if (horizontalZero) svg += '<line x1="' + m.l + '" x2="' + (width - m.r) + '" y1="' + sy(0) + '" y2="' + sy(0) + '" stroke="#111827" stroke-width="1.2"/>';
  svg += '<text x="20" y="' + ((m.t + height - m.b) / 2) + '" transform="rotate(-90 20 ' + ((m.t + height - m.b) / 2) + ')" text-anchor="middle" font-family="Arial" font-size="14" fill="#111827">' + svgEscape(yLabel) + '</text>';
  rows.forEach((r, i) => {
    const v = safeFloat(r[valueKey]);
    if (v === null) return;
    const x = m.l + i * (width - m.l - m.r) / Math.max(rows.length, 1) + ((width - m.l - m.r) / Math.max(rows.length, 1) - barW) / 2;
    const y0 = horizontalZero ? sy(0) : sy(yr.min);
    const y1 = sy(v);
    const y = Math.min(y0, y1); const h = Math.max(1, Math.abs(y1 - y0));
    const color = PALETTE[r.condition] || PALETTE[r.group] || '#64748b';
    svg += '<rect x="' + x.toFixed(1) + '" y="' + y.toFixed(1) + '" width="' + barW.toFixed(1) + '" height="' + h.toFixed(1) + '" fill="' + color + '" rx="2"/>';
    svg += '<text x="' + (x + barW / 2).toFixed(1) + '" y="' + (height - m.b + 18) + '" transform="rotate(55 ' + (x + barW / 2).toFixed(1) + ' ' + (height - m.b + 18) + ')" text-anchor="start" font-family="Arial" font-size="11" fill="#111827">' + svgEscape(labelFn(r)) + '</text>';
  });
  svg += '</svg>';
  return svg;
}
function heatmapSvg(title, rows) {
  const rowLabels = [...new Set(rows.map(r => r.dataset_folder + ' / ' + r.model_folder))].sort();
  const colLabels = CONDITION_ORDER.map(c => conditionLabel(c[0], c[1]));
  const cellW = 116; const cellH = 48; const m = { l: 170, r: 30, t: 92, b: 30 };
  const width = m.l + m.r + cellW * colLabels.length; const height = m.t + m.b + cellH * rowLabels.length;
  const vals = rows.map(r => safeFloat(r.best_eval_accuracy)).filter(v => v !== null);
  const min = vals.length ? Math.min(...vals) : 0; const max = vals.length ? Math.max(...vals) : 1;
  function color(v) {
    if (v === null) return '#f3f4f6';
    const t = max === min ? 0.5 : (v - min) / (max - min);
    const r = Math.round(45 + t * 15); const g = Math.round(90 + t * 120); const b = Math.round(120 + t * 50);
    return 'rgb(' + r + ',' + g + ',' + b + ')';
  }
  let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"><rect width="100%" height="100%" fill="#ffffff"/>';
  svg += '<text x="24" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#111827">' + svgEscape(title) + '</text>';
  colLabels.forEach((c, j) => svg += '<text x="' + (m.l + j * cellW + cellW / 2) + '" y="70" transform="rotate(-30 ' + (m.l + j * cellW + cellW / 2) + ' 70)" text-anchor="middle" font-family="Arial" font-size="12" fill="#111827">' + svgEscape(c) + '</text>');
  rowLabels.forEach((rlabel, i) => {
    svg += '<text x="' + (m.l - 10) + '" y="' + (m.t + i * cellH + cellH / 2 + 5) + '" text-anchor="end" font-family="Arial" font-size="13" fill="#111827">' + svgEscape(rlabel) + '</text>';
    colLabels.forEach((clabel, j) => {
      const match = rows.find(r => (r.dataset_folder + ' / ' + r.model_folder) === rlabel && r.condition === clabel);
      const v = match ? safeFloat(match.best_eval_accuracy) : null;
      const x = m.l + j * cellW; const y = m.t + i * cellH;
      svg += '<rect x="' + x + '" y="' + y + '" width="' + cellW + '" height="' + cellH + '" fill="' + color(v) + '" stroke="#ffffff"/>';
      svg += '<text x="' + (x + cellW / 2) + '" y="' + (y + cellH / 2 + 5) + '" text-anchor="middle" font-family="Arial" font-size="13" font-weight="700" fill="white">' + (v === null ? '' : v.toFixed(3)) + '</text>';
    });
  });
  svg += '</svg>';
  return svg;
}
function scatterSvg(title, rows) {
  const width = 720; const height = 620; const m = { l: 72, r: 42, t: 58, b: 70 };
  const vals = rows.flatMap(r => [safeFloat(r.final_eval_accuracy), safeFloat(r.best_eval_accuracy)]).filter(v => v !== null);
  let min = vals.length ? Math.min(...vals) : 0; let max = vals.length ? Math.max(...vals) : 1;
  const pad = (max - min || 1) * 0.08; min = Math.max(0, min - pad); max += pad;
  const sx = x => m.l + (x - min) / (max - min) * (width - m.l - m.r);
  const sy = y => height - m.b - (y - min) / (max - min) * (height - m.t - m.b);
  let svg = '<svg xmlns="http://www.w3.org/2000/svg" width="' + width + '" height="' + height + '" viewBox="0 0 ' + width + ' ' + height + '"><rect width="100%" height="100%" fill="#ffffff"/>';
  svg += '<text x="' + m.l + '" y="32" font-family="Arial" font-size="22" font-weight="700" fill="#111827">' + svgEscape(title) + '</text>';
  for (let i = 0; i <= 5; i++) {
    const x = m.l + i * (width - m.l - m.r) / 5; const y = m.t + i * (height - m.t - m.b) / 5;
    svg += '<line x1="' + x + '" x2="' + x + '" y1="' + m.t + '" y2="' + (height - m.b) + '" stroke="#e5e7eb"/>';
    svg += '<line x1="' + m.l + '" x2="' + (width - m.r) + '" y1="' + y + '" y2="' + y + '" stroke="#e5e7eb"/>';
  }
  svg += '<line x1="' + sx(min) + '" y1="' + sy(min) + '" x2="' + sx(max) + '" y2="' + sy(max) + '" stroke="#111827"/>';
  for (const r of rows) {
    const x = safeFloat(r.final_eval_accuracy); const y = safeFloat(r.best_eval_accuracy);
    if (x === null || y === null) continue;
    svg += '<circle cx="' + sx(x).toFixed(1) + '" cy="' + sy(y).toFixed(1) + '" r="4.5" fill="' + (PALETTE[r.condition] || '#2563eb') + '" opacity="0.85"><title>' + svgEscape(r.run_name) + '</title></circle>';
  }
  svg += '<text x="' + ((m.l + width - m.r) / 2) + '" y="' + (height - 24) + '" text-anchor="middle" font-family="Arial" font-size="14">Final accuracy</text>';
  svg += '<text x="22" y="' + ((m.t + height - m.b) / 2) + '" transform="rotate(-90 22 ' + ((m.t + height - m.b) / 2) + ')" text-anchor="middle" font-family="Arial" font-size="14">Best accuracy</text>';
  svg += '</svg>';
  return svg;
}
async function savePlotTableAndPng(name, rows, columns, svg) {
  await writeCSV(path.join(PLOTS_DIR, name + '.csv'), rows, columns);
  const ok = await renderSvg(name, svg);
  return ok ? 1 : 0;
}
async function makePlots(runs) {
  let plotCount = 0;
  const warnings = [];
  if (!sharp) warnings.push('sharp unavailable; PNG plots were not generated');
  for (const dataset of [...new Set(runs.map(r => r.dataset_folder))].sort()) {
    for (const model of [...new Set(runs.filter(r => r.dataset_folder === dataset).map(r => r.model_folder))].sort()) {
      const members = runs.filter(r => r.dataset_folder === dataset && r.model_folder === model);
      for (const spec of [{ metric: 'eval_acc', title: 'Accuracy', ylabel: 'Accuracy', suffix: 'accuracy_curves' }, { metric: 'eval_loss', title: 'Loss', ylabel: 'Loss', suffix: 'loss_curves' }]) {
        const rows = [];
        for (const run of members) {
          const values = run.series[spec.metric] || [];
          values.forEach((value, i) => { if (value !== null) rows.push({ dataset_folder: dataset, model_folder: model, run_name: run.run_name, condition: run.condition, epoch: run.epochs[i] || i + 1, value }); });
        }
        const name = dataset + '_' + model + '_' + spec.suffix;
        plotCount += await savePlotTableAndPng(name, rows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'epoch', 'value'], lineChartSvg(dataset + ' / ' + model + ' ' + spec.title + ' Curves', rows, 'epoch', 'value', 'condition', spec.ylabel));
      }
    }
  }
  const bestRows = runs.filter(r => r.best_eval_accuracy !== null).map(r => ({ dataset_folder: r.dataset_folder, model_folder: r.model_folder, run_name: r.run_name, condition: r.condition, best_eval_accuracy: r.best_eval_accuracy, final_eval_accuracy: r.final_eval_accuracy, best_epoch: r.best_epoch, improvement_over_none: r.improvement_over_none, improvement_over_random: r.improvement_over_random }));
  for (const dataset of [...new Set(bestRows.map(r => r.dataset_folder))].sort()) {
    const rows = bestRows.filter(r => r.dataset_folder === dataset);
    plotCount += await savePlotTableAndPng(dataset + '_best_accuracy_bars', rows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'best_eval_accuracy', 'final_eval_accuracy', 'best_epoch', 'improvement_over_none'], barChartSvg(dataset + ' Best Accuracy', rows, r => r.model_folder + ' ' + r.condition, 'best_eval_accuracy', 'Best accuracy', false));
  }
  plotCount += await savePlotTableAndPng('best_accuracy_bar_chart', bestRows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'best_eval_accuracy', 'final_eval_accuracy', 'best_epoch', 'improvement_over_none'], barChartSvg('Best Accuracy by Run', bestRows, r => r.dataset_folder + ' ' + r.model_folder + ' ' + r.condition, 'best_eval_accuracy', 'Best accuracy', false));
  const improvementRows = bestRows.filter(r => safeFloat(r.improvement_over_none) !== null);
  plotCount += await savePlotTableAndPng('improvement_over_baseline_bars', improvementRows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'improvement_over_none'], barChartSvg('Improvement Over Matching No-Cutout Baseline', improvementRows, r => r.dataset_folder + ' ' + r.model_folder + ' ' + r.condition, 'improvement_over_none', 'Accuracy difference', true));
  const rawImprovement = improvementRows.filter(r => r.dataset_folder === 'rawmaltf');
  plotCount += await savePlotTableAndPng('rawmaltf_improvement_over_baseline_bars', rawImprovement, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'improvement_over_none'], barChartSvg('RawMal-TF Improvement Over No-Cutout', rawImprovement, r => r.model_folder + ' ' + r.condition, 'improvement_over_none', 'Accuracy difference', true));
  const raw = bestRows.filter(r => r.dataset_folder === 'rawmaltf');
  plotCount += await savePlotTableAndPng('rawmaltf_model_condition_comparison', raw, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'best_eval_accuracy', 'final_eval_accuracy', 'best_epoch', 'improvement_over_none'], barChartSvg('RawMal-TF Model/Condition Best Accuracy', raw, r => r.model_folder + ' ' + r.condition, 'best_eval_accuracy', 'Best accuracy', false));
  for (const spec of [
    { name: 'rawmaltf_random_vs_cam', title: 'RawMal-TF Random vs CAM', group: r => r.condition.startsWith('random') ? 'random' : (r.condition.startsWith('cam') ? 'CAM' : 'none') },
    { name: 'rawmaltf_m4_vs_m8', title: 'RawMal-TF M4 vs M8', group: r => r.condition.includes('M4') ? 'M4' : (r.condition.includes('M8') ? 'M8' : 'none') },
    { name: 'rawmaltf_cam_low_vs_high', title: 'RawMal-TF CAM Low vs High', group: r => r.condition.startsWith('cam_low') ? 'cam_low' : (r.condition.startsWith('cam_high') ? 'cam_high' : r.condition) }
  ]) {
    const grouped = new Map();
    for (const r of raw) {
      const g = spec.group(r);
      if (!grouped.has(g)) grouped.set(g, []);
      grouped.get(g).push(r.best_eval_accuracy);
    }
    const rows = [...grouped.entries()].sort().map(([group, values]) => ({ group, mean_best_accuracy: mean(values), count: values.length, condition: group }));
    plotCount += await savePlotTableAndPng(spec.name, rows, ['group', 'mean_best_accuracy', 'count'], barChartSvg(spec.title, rows, r => r.group, 'mean_best_accuracy', 'Mean best accuracy', false));
  }
  plotCount += await savePlotTableAndPng('overall_best_accuracy_heatmap', bestRows, ['dataset_folder', 'model_folder', 'condition', 'best_eval_accuracy'], heatmapSvg('Overall Best Accuracy Heatmap', bestRows));
  plotCount += await savePlotTableAndPng('final_vs_best_accuracy', bestRows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'best_eval_accuracy', 'final_eval_accuracy'], scatterSvg('Final vs Best Accuracy', bestRows));
  plotCount += await savePlotTableAndPng('best_epoch', bestRows, ['dataset_folder', 'model_folder', 'run_name', 'condition', 'best_epoch', 'best_eval_accuracy'], barChartSvg('Best Epoch by Run', bestRows, r => r.dataset_folder + ' ' + r.model_folder + ' ' + r.condition, 'best_epoch', 'Best epoch', false));
  return { plotCount, warnings };
}
function inventoryColumns() { return ['dataset_folder', 'model_folder', 'run_name', 'relative_path', 'config_dataset', 'normalized_dataset', 'config_model', 'seed', 'cutout_mode', 'cutout_m', 'cutout_area', 'configured_epochs', 'observed_metric_epochs', 'first_epoch', 'last_epoch', 'grayscale', 'include_regex', 'teacher_checkpoint', 'teacher_model', 'config_exists', 'metrics_exists', 'metrics_plot_exists', 'best_model_exists', 'log_exists', 'log_count', 'best_model_bytes', 'best_model_sha256', 'best_model_metadata_status', 'best_model_metadata', 'metrics_hash', 'config_hash', 'metrics_columns', 'status', 'notes']; }
function summaryColumns() { return ['dataset_folder', 'model_folder', 'run_name', 'condition', 'seed', 'cutout_mode', 'cutout_m', 'cutout_area', 'eval_split', 'final_train_loss', 'final_train_accuracy', 'final_eval_loss', 'final_eval_accuracy', 'best_train_loss', 'best_train_loss_epoch', 'best_train_accuracy', 'best_train_accuracy_epoch', 'best_eval_loss', 'best_eval_loss_epoch', 'best_eval_accuracy', 'best_epoch', 'generalization_gap', 'baseline_run_name', 'improvement_over_none', 'relative_improvement_over_none_pct', 'random_baseline_run_name', 'improvement_over_random', 'relative_improvement_over_random_pct', 'status', 'notes']; }
function comparisonColumns() {
  const cols = ['dataset_folder', 'model_folder', 'seed', 'baseline_run_name', 'baseline_best_accuracy'];
  for (const [mode, m] of CONDITION_ORDER) {
    const slug = conditionSlug(conditionLabel(mode, m));
    cols.push(slug + '_run_name', slug + '_best_accuracy', slug + '_final_accuracy', slug + '_best_epoch', slug + '_improvement_over_none');
  }
  return cols;
}
function readmeText() {
  return ['# Summary Package', '', 'This folder contains publication-oriented summaries generated from existing run artifacts under runs/cifar100/, runs/malimg/, and runs/rawmaltf/.', '', '## Files', '', '- run_inventory.csv: one row per run with config fields, artifact flags, checkpoint file metadata, status, and notes.', '- run_summary.csv: final and best train/evaluation metrics, generalization gap, and improvements over matching no-cutout and random baselines.', '- comparison_table.csv: paper-friendly wide table by dataset/model/seed for none, random M4/M8, cam_low M4/M8, and cam_high M4/M8.', '- integrity_checks.csv: audit checks for missing files, epoch mismatches, NaNs, constant metrics, duplicate hashes, folder/config mismatches, missing CAM teachers, and identical CAM-low/CAM-high metrics.', '- summary_stats.json: aggregate counts, best runs, improvement statistics, CAM comparisons, and major warnings.', '- paper_summary.md: human-readable report for manuscript triage.', '- plots/: PNG plots plus the CSV data used to build each plot.', '', '## Rerun', '', 'From the repository root, run:', '', '    python runs/summary/generate_summary.py', '', 'or:', '', '    node runs/summary/generate_summary.js', '', 'The generator reads only the source run folders and writes only inside runs/summary/. Rerunning it refreshes the generated summary files in this folder.', ''].join('\n');
}
export async function main() {
  await ensureDirs();
  const runs = await discoverRuns();
  const checks = buildIntegrityChecks(runs);
  applyStatuses(runs, checks);
  addImprovements(runs);
  const comparison = makeComparisonTable(runs);
  const plotResult = await makePlots(runs);
  const publicRuns = runs.map(publicRun);
  let summaryStats = makeSummaryStats(runs, checks);
  if (plotResult.warnings.length) summaryStats.major_warnings = summaryStats.major_warnings.concat(plotResult.warnings);
  await writeCSV(path.join(SUMMARY_DIR, 'run_inventory.csv'), publicRuns, inventoryColumns());
  await writeCSV(path.join(SUMMARY_DIR, 'run_summary.csv'), publicRuns, summaryColumns());
  await writeCSV(path.join(SUMMARY_DIR, 'comparison_table.csv'), comparison, comparisonColumns());
  await writeCSV(path.join(SUMMARY_DIR, 'integrity_checks.csv'), checks, ['check_type', 'severity', 'status', 'dataset_folder', 'model_folder', 'seed', 'run_name', 'condition', 'related_run', 'details']);
  await fs.writeFile(path.join(SUMMARY_DIR, 'summary_stats.json'), JSON.stringify(summaryStats, null, 2) + '\n', 'utf8');
  await fs.writeFile(path.join(SUMMARY_DIR, 'paper_summary.md'), makePaperSummary(publicRuns, checks, summaryStats), 'utf8');
  await fs.writeFile(path.join(SUMMARY_DIR, 'README.md'), readmeText(), 'utf8');
  const allFiles = [];
  async function walk(dir) {
    for (const ent of await fs.readdir(dir, { withFileTypes: true })) {
      const p = path.join(dir, ent.name);
      if (ent.isDirectory()) await walk(p); else allFiles.push(p);
    }
  }
  await walk(SUMMARY_DIR);
  const result = { runs_processed: runs.length, files_created: allFiles.length, plots_created: plotResult.plotCount, major_warnings: summaryStats.major_warnings, readme: rel(path.join(SUMMARY_DIR, 'README.md')) };
  console.log('runs processed: ' + result.runs_processed);
  console.log('files created: ' + result.files_created);
  console.log('plots created: ' + result.plots_created);
  console.log('major warnings:');
  if (result.major_warnings.length) result.major_warnings.slice(0, 20).forEach(w => console.log('- ' + w)); else console.log('- none');
  if (result.major_warnings.length > 20) console.log('- ... ' + (result.major_warnings.length - 20) + ' more');
  console.log('README: ' + result.readme);
  return result;
}
await main();
