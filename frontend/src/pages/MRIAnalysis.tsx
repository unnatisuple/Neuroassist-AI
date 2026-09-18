import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { mriAPI, reportAPI } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';

type BrainRegionItem = {
  region_name: string;
  attribution_level: string;
  attribution_score: number;
  clinical_note?: string;
};

type BrainRegionAnalysis = {
  regions: BrainRegionItem[];
  is_estimated: boolean;
  methodology: string;
  disclaimer: string;
};

type XAIVisualization = {
  heatmap_base64?: string;
  overlay_base64?: string;
  description?: string;
};

type PredictionResult = {
  prediction_id: string;
  predicted_class: string;
  predicted_class_index?: number;
  confidence: number;
  class_probabilities: Record<string, number>;
  model_version: string;
  model_checksum: string;
  inference_latency_ms: number;
  trace_id: string;
  disclaimer: string;
  severe_stage_note: string;
  patient_id?: string;
  xai_overlays?: Record<string, string>;
  explainability?: {
    grad_cam?: XAIVisualization;
    grad_cam_plus_plus?: XAIVisualization;
    integrated_gradients?: XAIVisualization;
    hirescam?: XAIVisualization;
    brain_regions?: BrainRegionAnalysis;
  };
  brain_regions?: BrainRegionAnalysis;
};

const ORDERED_CLASSES = [
  'Non-Demented',
  'Very Mild Demented',
  'Mild Demented',
  'Moderate Demented',
];

export default function MRIAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

  // XAI View state
  const [activeXaiTab, setActiveXaiTab] = useState<
    'gradcam' | 'gradcam_plus_plus' | 'integrated_gradients' | 'brain_regions'
  >('gradcam');
  const [displayMode, setDisplayMode] = useState<'overlay' | 'heatmap' | 'original'>('overlay');

  // Report state
  const [generatingReport, setGeneratingReport] = useState(false);
  const [downloadingReport, setDownloadingReport] = useState(false);
  const [reportId, setReportId] = useState<string | null>(null);
  const [reportPreview, setReportPreview] = useState<string>('');

  const onDrop = useCallback((acceptedFiles: File[]) => {
    const f = acceptedFiles[0];
    if (f) {
      setFile(f);
      setPreview(URL.createObjectURL(f));
      setPrediction(null);
      setReportId(null);
      setReportPreview('');
      setError('');
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: { 'image/jpeg': [], 'image/png': [], 'image/tiff': [] },
    maxFiles: 1,
    maxSize: 50 * 1024 * 1024,
  });

  const handleUploadAndPredict = async () => {
    if (!file) return;
    setError('');
    setPrediction(null);
    setReportId(null);
    setReportPreview('');

    // Upload
    setUploading(true);
    try {
      const uploadRes = await mriAPI.upload(file);
      const fid = uploadRes.data.file_id;
      setUploading(false);

      // Predict
      setPredicting(true);
      const predRes = await mriAPI.predict(fid);
      setPrediction(predRes.data);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string; trace_id?: string } } };
      const detail = axiosErr.response?.data?.detail || 'Analysis failed.';
      const traceId = axiosErr.response?.data?.trace_id || '';
      setError(`${detail}${traceId ? ` (Trace: ${traceId})` : ''}`);
    } finally {
      setUploading(false);
      setPredicting(false);
    }
  };

  const handleGenerateReport = async () => {
    if (!prediction) return;
    setGeneratingReport(true);
    setError('');
    try {
      const res = await reportAPI.generate({
        prediction_id: prediction.prediction_id,
        patient_id: prediction.patient_id || undefined,
      });
      setReportId(res.data.report_id);
      setReportPreview(res.data.summary);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string; trace_id?: string } } };
      const detail = axiosErr.response?.data?.detail || 'Report generation failed.';
      const traceId = axiosErr.response?.data?.trace_id || '';
      setError(`${detail}${traceId ? ` (Trace: ${traceId})` : ''}`);
    } finally {
      setGeneratingReport(false);
    }
  };

  const handleDownloadReport = async () => {
    if (!reportId) return;
    setDownloadingReport(true);
    setError('');
    try {
      const res = await reportAPI.download(reportId);
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `NeuroAssist_Report_${reportId.substring(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch (err: unknown) {
      setError('Report download failed. Please try again.');
    } finally {
      setDownloadingReport(false);
    }
  };

  const handleDownloadImage = (base64Data: string, filename: string) => {
    const link = document.createElement('a');
    link.href = `data:image/png;base64,${base64Data}`;
    link.download = filename;
    document.body.appendChild(link);
    link.click();
    link.remove();
  };

  const stageColors: Record<string, string> = {
    'Non-Demented': 'text-emerald-400',
    'Very Mild Demented': 'text-yellow-400',
    'Mild Demented': 'text-orange-400',
    'Moderate Demented': 'text-red-400',
  };

  const stageGradients: Record<string, string> = {
    'Non-Demented': 'from-emerald-500 to-teal-400',
    'Very Mild Demented': 'from-yellow-500 to-amber-400',
    'Mild Demented': 'from-orange-500 to-amber-500',
    'Moderate Demented': 'from-rose-500 to-red-600',
  };

  // Helper to extract image based on active method & display mode
  const getActiveVisualization = () => {
    if (!prediction) return null;
    const overlays = prediction.xai_overlays || {};
    const explainability = prediction.explainability || {};

    let overlayKey = '';
    let heatmapKey = '';
    let title = '';
    let desc = '';
    let alt = '';

    if (activeXaiTab === 'gradcam') {
      overlayKey = 'gradcam_overlay';
      heatmapKey = 'gradcam_heatmap';
      title = 'Grad-CAM (Gradient-Weighted Class Activation Mapping)';
      desc =
        explainability.grad_cam?.description ||
        'Highlights coarse convolutional feature activations that contributed most strongly to the predicted class.';
      alt = 'Grad-CAM explanation overlay for MRI classification';
    } else if (activeXaiTab === 'gradcam_plus_plus') {
      overlayKey = 'gradcam_plus_plus_overlay';
      heatmapKey = 'gradcam_plus_plus_heatmap';
      title = 'Grad-CAM++ (Second-Order Gradient Weighting)';
      desc =
        explainability.grad_cam_plus_plus?.description ||
        'Provides refined class-discriminative localization using higher-order positive gradient weighting for fine structures.';
      alt = 'Grad-CAM++ explanation overlay for MRI classification';
    } else if (activeXaiTab === 'integrated_gradients') {
      overlayKey = 'integrated_gradients_overlay';
      heatmapKey = 'integrated_gradients_heatmap';
      title = 'Integrated Gradients (Path-Integral Attribution via Captum)';
      desc =
        explainability.integrated_gradients?.description ||
        'Computes pixel-level path attribution from a zero baseline directly to the input image, showing features influencing decision confidence.';
      alt = 'Integrated Gradients attribution map for MRI classification';
    }

    const overlayBase64 = overlays[overlayKey] || '';
    const heatmapBase64 = overlays[heatmapKey] || '';

    return {
      title,
      desc,
      alt,
      overlayBase64,
      heatmapBase64,
    };
  };

  const activeVis = getActiveVisualization();
  const brainRegionsData =
    prediction?.brain_regions || prediction?.explainability?.brain_regions;

  return (
    <div className="space-y-8">
      <div>
        <h1 className="section-title">MRI Analysis</h1>
        <p className="section-subtitle">
          Upload a brain MRI scan for AI-assisted dementia stage classification and live explainability
        </p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Left Column: Upload & Scan Preview */}
        <div className="space-y-4">
          <div
            {...getRootProps()}
            className={`glass-card p-10 text-center cursor-pointer transition-all duration-300 ${
              isDragActive ? 'border-brand-500 bg-brand-500/10' : 'hover:border-white/20'
            }`}
          >
            <input {...getInputProps()} />
            {preview ? (
              <div className="mri-viewer">
                <img
                  src={preview}
                  alt="Uploaded MRI scan preview"
                  className="max-h-64 mx-auto rounded-lg shadow-lg border border-white/10"
                />
              </div>
            ) : (
              <div className="py-8">
                <div className="text-5xl mb-4">🧠</div>
                <p className="text-white font-semibold">Drop a brain MRI scan here</p>
                <p className="text-slate-400 text-sm mt-2">JPEG, PNG, or TIFF — Max 50MB</p>
              </div>
            )}
          </div>

          {file && !prediction && (
            <button
              onClick={handleUploadAndPredict}
              disabled={uploading || predicting}
              className="btn-primary w-full"
            >
              {uploading ? '📤 Uploading...' : predicting ? '🧠 Running Inference...' : '🔬 Analyze Scan'}
            </button>
          )}

          {prediction && (
            <button
              onClick={() => {
                setFile(null);
                setPreview('');
                setPrediction(null);
                setReportId(null);
                setReportPreview('');
              }}
              className="btn-secondary w-full text-xs text-slate-300 py-2"
            >
              🔄 Upload Different MRI Scan
            </button>
          )}
        </div>

        {/* Right Column: Prediction, Probabilities, & Report Actions */}
        <AnimatePresence>
          {prediction && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              {/* Classification Result Card */}
              <div className="glass-card-elevated p-6">
                <h3 className="font-bold text-white mb-2">Classification Result</h3>

                <div className="text-center py-4 bg-white/5 rounded-xl border border-white/5 mb-4">
                  <p
                    className={`text-3xl font-extrabold ${
                      stageColors[prediction.predicted_class] || 'text-white'
                    }`}
                  >
                    {prediction.predicted_class}
                  </p>
                  <p className="text-slate-400 mt-2 text-sm">
                    Model Confidence:{' '}
                    <span className="text-white font-bold">
                      {(prediction.confidence * 100).toFixed(1)}%
                    </span>
                  </p>
                </div>

                {/* Class Probability Chart */}
                <div className="space-y-3 pt-2">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-slate-300 uppercase tracking-wider">
                      Class Probability Distribution
                    </span>
                    <span className="text-[11px] text-slate-500 font-mono">Softmax Logits</span>
                  </div>

                  <div
                    className="space-y-2.5 bg-black/20 p-3.5 rounded-xl border border-white/5"
                    aria-label="Class probability chart showing four MRI classification classes"
                  >
                    {ORDERED_CLASSES.map((cls) => {
                      const prob = prediction.class_probabilities[cls] ?? 0;
                      const pct = (prob * 100).toFixed(1);
                      const isPredicted = prediction.predicted_class === cls;
                      const gradient =
                        stageGradients[cls] || 'from-brand-600 to-medical-teal';

                      return (
                        <div
                          key={cls}
                          className={`p-2 rounded-lg transition-all ${
                            isPredicted
                              ? 'bg-white/10 border border-brand-500/40 shadow-sm'
                              : 'hover:bg-white/5'
                          }`}
                        >
                          <div className="flex items-center justify-between text-xs mb-1.5">
                            <div className="flex items-center gap-2">
                              <span
                                className={`w-2 h-2 rounded-full ${
                                  isPredicted ? 'bg-medical-teal animate-pulse' : 'bg-slate-600'
                                }`}
                              />
                              <span
                                className={`font-medium ${
                                  isPredicted ? 'text-white font-bold' : 'text-slate-400'
                                }`}
                              >
                                {cls}
                              </span>
                              {isPredicted && (
                                <span className="text-[10px] bg-brand-500/20 text-brand-300 border border-brand-500/30 px-1.5 py-0.5 rounded font-mono">
                                  PREDICTED
                                </span>
                              )}
                            </div>
                            <span className="text-white font-mono font-semibold">{pct}%</span>
                          </div>

                          <div className="h-2 bg-white/5 rounded-full overflow-hidden">
                            <div
                              className={`h-full bg-gradient-to-r ${gradient} rounded-full transition-all duration-700`}
                              style={{ width: `${Math.max(Number(pct), 1.5)}%` }}
                            />
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>

                {/* Metadata */}
                <div className="mt-4 pt-4 border-t border-white/5 flex flex-wrap items-center justify-between text-xs text-slate-500">
                  <span>
                    Model: <strong className="text-slate-400">{prediction.model_version}</strong> (
                    {prediction.model_checksum})
                  </span>
                  <span>Latency: {prediction.inference_latency_ms}ms</span>
                </div>
              </div>

              {/* Severe stage note */}
              <div className="disclaimer-banner">
                <span>⚠️</span>
                <span className="text-xs">{prediction.severe_stage_note}</span>
              </div>

              {/* Clinical Report Section */}
              <div className="glass-card p-6">
                <h3 className="font-bold text-white mb-3">Clinical Decision Report</h3>
                <div className="space-y-4">
                  {reportPreview && (
                    <div className="bg-white/5 rounded-xl p-4 border border-white/10 space-y-2">
                      <p className="text-xs text-medical-teal-light font-bold uppercase tracking-wider">
                        Report Preview (Grounding Groq LLM)
                      </p>
                      <p className="text-sm text-slate-300 line-clamp-4">{reportPreview}</p>
                    </div>
                  )}

                  <div className="flex gap-4">
                    <button
                      onClick={handleGenerateReport}
                      disabled={generatingReport || predicting}
                      className="btn-primary flex-1"
                    >
                      {generatingReport ? (
                        <span className="flex items-center justify-center gap-2">
                          <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Generating Report...
                        </span>
                      ) : (
                        '📝 Generate Report'
                      )}
                    </button>

                    <button
                      onClick={handleDownloadReport}
                      disabled={!reportId || downloadingReport}
                      className={`flex-1 py-3 px-4 rounded-xl font-bold transition-all duration-300 text-center ${
                        reportId
                          ? 'bg-gradient-to-r from-medical-teal to-brand-600 text-white shadow-neon hover:opacity-90 cursor-pointer'
                          : 'bg-white/5 text-slate-500 cursor-not-allowed border border-white/5'
                      }`}
                    >
                      {downloadingReport ? (
                        <span className="flex items-center justify-center gap-2">
                          <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                          Downloading...
                        </span>
                      ) : (
                        '📥 Download PDF'
                      )}
                    </button>
                  </div>
                </div>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Explainability (XAI) & Brain Region Analysis Full-Width Section */}
      <AnimatePresence>
        {prediction && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="space-y-4 pt-2"
          >
            <div className="glass-card p-6">
              <div className="flex flex-wrap items-center justify-between gap-4 mb-6 border-b border-white/5 pb-4">
                <div>
                  <h3 className="text-xl font-bold text-white flex items-center gap-2">
                    <span>🔬</span> Explainable AI (XAI) & Brain Region Analysis
                  </h3>
                  <p className="text-xs text-slate-400 mt-1">
                    Live convolutional activations, gradient attributions, and 2D image-space anatomical localization
                  </p>
                </div>

                {/* XAI Navigation Tabs */}
                <div className="flex items-center gap-1.5 bg-black/40 p-1.5 rounded-xl border border-white/10">
                  <button
                    onClick={() => setActiveXaiTab('gradcam')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeXaiTab === 'gradcam'
                        ? 'bg-brand-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    Grad-CAM
                  </button>
                  <button
                    onClick={() => setActiveXaiTab('gradcam_plus_plus')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeXaiTab === 'gradcam_plus_plus'
                        ? 'bg-brand-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    Grad-CAM++
                  </button>
                  <button
                    onClick={() => setActiveXaiTab('integrated_gradients')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeXaiTab === 'integrated_gradients'
                        ? 'bg-brand-600 text-white shadow-md'
                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    Integrated Gradients
                  </button>
                  <button
                    onClick={() => setActiveXaiTab('brain_regions')}
                    className={`px-3.5 py-1.5 rounded-lg text-xs font-semibold transition-all ${
                      activeXaiTab === 'brain_regions'
                        ? 'bg-medical-teal text-white shadow-md'
                        : 'text-slate-400 hover:text-white hover:bg-white/5'
                    }`}
                  >
                    Brain Regions
                  </button>
                </div>
              </div>

              {/* Tab 1, 2, 3: Heatmap Visualizations */}
              {activeXaiTab !== 'brain_regions' && activeVis && (
                <div className="grid grid-cols-1 md:grid-cols-12 gap-6 items-start">
                  {/* Image View Container */}
                  <div className="md:col-span-6 space-y-3">
                    <div className="flex items-center justify-between">
                      {/* Display Mode Toggle */}
                      <div className="flex items-center gap-1 bg-black/30 p-1 rounded-lg border border-white/5 text-xs">
                        <button
                          onClick={() => setDisplayMode('overlay')}
                          className={`px-3 py-1 rounded-md transition-all ${
                            displayMode === 'overlay'
                              ? 'bg-white/20 text-white font-bold'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Overlay
                        </button>
                        <button
                          onClick={() => setDisplayMode('heatmap')}
                          className={`px-3 py-1 rounded-md transition-all ${
                            displayMode === 'heatmap'
                              ? 'bg-white/20 text-white font-bold'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Heatmap
                        </button>
                        <button
                          onClick={() => setDisplayMode('original')}
                          className={`px-3 py-1 rounded-md transition-all ${
                            displayMode === 'original'
                              ? 'bg-white/20 text-white font-bold'
                              : 'text-slate-400 hover:text-white'
                          }`}
                        >
                          Original Scan
                        </button>
                      </div>

                      {/* Download image button */}
                      {(displayMode === 'overlay'
                        ? activeVis.overlayBase64
                        : activeVis.heatmapBase64) && (
                        <button
                          onClick={() =>
                            handleDownloadImage(
                              displayMode === 'overlay'
                                ? activeVis.overlayBase64
                                : activeVis.heatmapBase64,
                              `${activeXaiTab}_${displayMode}.png`
                            )
                          }
                          className="text-xs text-medical-teal-light hover:text-white flex items-center gap-1.5 px-2.5 py-1 bg-white/5 rounded-lg border border-white/10 hover:bg-white/10 transition-all"
                          title="Download high-resolution image"
                        >
                          <span>💾</span> Download {displayMode.toUpperCase()}
                        </button>
                      )}
                    </div>

                    {/* Image Viewer */}
                    <div className="border border-white/10 rounded-2xl overflow-hidden bg-black/60 aspect-square max-w-md mx-auto flex items-center justify-center p-2 relative shadow-inner">
                      {displayMode === 'original' && preview ? (
                        <img
                          src={preview}
                          alt="Original MRI scan slice"
                          className="w-full h-full object-contain rounded-xl"
                        />
                      ) : displayMode === 'heatmap' && activeVis.heatmapBase64 ? (
                        <img
                          src={`data:image/png;base64,${activeVis.heatmapBase64}`}
                          alt={`${activeVis.alt} (Heatmap only)`}
                          className="w-full h-full object-contain rounded-xl"
                        />
                      ) : activeVis.overlayBase64 ? (
                        <img
                          src={`data:image/png;base64,${activeVis.overlayBase64}`}
                          alt={activeVis.alt}
                          className="w-full h-full object-contain rounded-xl"
                        />
                      ) : (
                        <div className="text-center py-12 text-slate-400 text-sm">
                          ⚠️ Visualization not available for this technique.
                        </div>
                      )}
                    </div>
                  </div>

                  {/* Technical & Clinical Explanation */}
                  <div className="md:col-span-6 space-y-4">
                    <div className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-2">
                      <h4 className="font-bold text-white text-base">{activeVis.title}</h4>
                      <p className="text-sm text-slate-300 leading-relaxed">{activeVis.desc}</p>
                    </div>

                    <div className="bg-white/5 border border-white/10 rounded-xl p-4 space-y-2">
                      <h5 className="text-xs font-semibold text-medical-teal-light uppercase tracking-wider">
                        Technical Computation Details
                      </h5>
                      <ul className="text-xs text-slate-300 space-y-1.5 list-disc list-inside">
                        {activeXaiTab === 'gradcam' && (
                          <>
                            <li>Target layer: Final residual convolutional stage (Conv2d 512 channels)</li>
                            <li>Target class: {prediction.predicted_class} (Index: {prediction.predicted_class_index ?? 'auto'})</li>
                            <li>Colormap: JET color scale overlay with alpha blending</li>
                          </>
                        )}
                        {activeXaiTab === 'gradcam_plus_plus' && (
                          <>
                            <li>Target layer: Final residual stage with positive second-order gradients</li>
                            <li>Resolution: Preserves localized peak activations with reduced false positives</li>
                            <li>Colormap: High-contrast JET overlay on patient MRI scan</li>
                          </>
                        )}
                        {activeXaiTab === 'integrated_gradients' && (
                          <>
                            <li>Method: Captum Integrated Gradients path-integral formulation</li>
                            <li>Baseline: Zero-tensor black baseline (absence of visual MRI signal)</li>
                            <li>Steps: 25 Gauss-Legendre integration intervals targeting predicted class</li>
                          </>
                        )}
                      </ul>
                    </div>

                    <div className="disclaimer-banner">
                      <span>⚠️</span>
                      <span className="text-[11px] leading-tight">
                        <strong>Clinical Interpretation Disclaimer:</strong> Saliency heatmaps indicate
                        regions weighted by the deep learning classifier and do NOT constitute proof of
                        anatomical lesions or autonomous clinical diagnosis.
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {/* Tab 4: Brain Region Analysis Table */}
              {activeXaiTab === 'brain_regions' && (
                <div className="space-y-4">
                  <div className="bg-white/5 border border-white/10 rounded-xl p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2 mb-2">
                      <h4 className="font-bold text-white text-base">
                        2D Image-Space Brain Region Localization
                      </h4>
                      <span className="text-[11px] bg-medical-teal/20 text-medical-teal-light border border-medical-teal/30 px-2 py-0.5 rounded-full font-mono">
                        ESTIMATED ANATOMICAL LOCALIZATION
                      </span>
                    </div>
                    <p className="text-xs text-slate-400">
                      Attribution density calculated by intersecting model XAI attention maps with canonical
                      2D axial brain template spatial priors.
                    </p>
                  </div>

                  {brainRegionsData?.regions && brainRegionsData.regions.length > 0 ? (
                    <div className="overflow-x-auto rounded-xl border border-white/10 bg-black/20">
                      <table className="w-full text-left text-xs text-slate-300">
                        <thead className="bg-white/5 text-slate-400 border-b border-white/10 uppercase tracking-wider text-[11px]">
                          <tr>
                            <th className="py-3 px-4">Anatomical Brain Region</th>
                            <th className="py-3 px-4">Model Attention / Status</th>
                            <th className="py-3 px-4">Attribution Share</th>
                            <th className="py-3 px-4">Clinical Relevance in AD</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-white/5">
                          {brainRegionsData.regions.map((region, idx) => {
                            let badgeStyle =
                              'bg-slate-700/50 text-slate-300 border border-slate-600';
                            if (region.attribution_level === 'High attribution') {
                              badgeStyle =
                                'bg-rose-500/20 text-rose-300 border border-rose-500/40 font-bold';
                            } else if (region.attribution_level === 'Moderate attribution') {
                              badgeStyle =
                                'bg-amber-500/20 text-amber-300 border border-amber-500/40 font-medium';
                            }

                            return (
                              <tr
                                key={region.region_name || idx}
                                className="hover:bg-white/5 transition-colors"
                              >
                                <td className="py-3 px-4 font-semibold text-white">
                                  {region.region_name}
                                </td>
                                <td className="py-3 px-4">
                                  <span
                                    className={`px-2 py-0.5 rounded-full text-[10px] inline-block ${badgeStyle}`}
                                  >
                                    {region.attribution_level}
                                  </span>
                                </td>
                                <td className="py-3 px-4 font-mono text-white">
                                  <div className="flex items-center gap-2">
                                    <div className="w-16 h-1.5 bg-white/10 rounded-full overflow-hidden">
                                      <div
                                        className="h-full bg-medical-teal rounded-full"
                                        style={{
                                          width: `${Math.min(
                                            region.attribution_score * 300,
                                            100
                                          )}%`,
                                        }}
                                      />
                                    </div>
                                    <span>{(region.attribution_score * 100).toFixed(1)}%</span>
                                  </div>
                                </td>
                                <td className="py-3 px-4 text-slate-400 leading-normal">
                                  {region.clinical_note || 'Anatomical region mapping'}
                                </td>
                              </tr>
                            );
                          })}
                        </tbody>
                      </table>
                    </div>
                  ) : (
                    <div className="text-center py-8 text-slate-400 text-sm">
                      ⚠️ Anatomical localization is not available for this image format.
                    </div>
                  )}

                  {/* Mandatory Clinical Limitation Notice */}
                  <div className="disclaimer-banner">
                    <span>⚠️</span>
                    <span className="text-[11px] leading-tight">
                      <strong>Clinical Limitation Note:</strong>{' '}
                      {brainRegionsData?.disclaimer ||
                        'Brain-region attribution represents model attention/attribution and is not equivalent to a confirmed anatomical lesion or clinical diagnosis.'}
                    </span>
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
