import { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { mriAPI, reportAPI } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';

type PredictionResult = {
  prediction_id: string;
  predicted_class: string;
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
};

export default function MRIAnalysis() {
  const [file, setFile] = useState<File | null>(null);
  const [preview, setPreview] = useState<string>('');
  const [uploading, setUploading] = useState(false);
  const [predicting, setPredicting] = useState(false);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState('');

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
      // Create blob and trigger native file download
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

  const stageColors: Record<string, string> = {
    'Non-Demented': 'text-emerald-400',
    'Very Mild Demented': 'text-yellow-400',
    'Mild Demented': 'text-orange-400',
    'Moderate Demented': 'text-red-400',
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="section-title">MRI Analysis</h1>
        <p className="section-subtitle">Upload a brain MRI scan for AI-assisted dementia stage classification</p>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-300 text-sm">
          {error}
        </div>
      )}

      <div className="grid grid-cols-2 gap-6">
        {/* Upload area */}
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
                <img src={preview} alt="MRI Preview" className="max-h-64 mx-auto rounded-lg shadow-lg" />
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
        </div>

        {/* Results */}
        <AnimatePresence>
          {prediction && (
            <motion.div
              initial={{ opacity: 0, x: 20 }}
              animate={{ opacity: 1, x: 0 }}
              className="space-y-4"
            >
              {/* Prediction result */}
              <div className="glass-card-elevated p-6">
                <h3 className="font-bold text-white mb-4">Classification Result</h3>

                <div className="text-center py-4">
                  <p className={`text-3xl font-extrabold ${stageColors[prediction.predicted_class] || 'text-white'}`}>
                    {prediction.predicted_class}
                  </p>
                  <p className="text-slate-400 mt-2">
                    Confidence: <span className="text-white font-bold">{(prediction.confidence * 100).toFixed(1)}%</span>
                  </p>
                </div>

                {/* Class probabilities */}
                <div className="space-y-2 mt-4">
                  {Object.entries(prediction.class_probabilities).map(([cls, prob]) => (
                    <div key={cls} className="flex items-center gap-3">
                      <span className="text-xs text-slate-400 w-32 truncate">{cls}</span>
                      <div className="flex-1 h-2 bg-white/5 rounded-full overflow-hidden">
                        <div
                          className="h-full bg-gradient-to-r from-brand-600 to-medical-teal rounded-full transition-all duration-500"
                          style={{ width: `${(prob as number) * 100}%` }}
                        />
                      </div>
                      <span className="text-xs text-white font-mono w-12 text-right">
                        {((prob as number) * 100).toFixed(1)}%
                      </span>
                    </div>
                  ))}
                </div>

                {/* Metadata */}
                <div className="mt-4 pt-4 border-t border-white/5 space-y-1">
                  <p className="text-xs text-slate-500">Model: {prediction.model_version} ({prediction.model_checksum})</p>
                  <p className="text-xs text-slate-500">Latency: {prediction.inference_latency_ms}ms</p>
                  <p className="text-xs text-slate-500">Trace: {prediction.trace_id}</p>
                </div>
              </div>

              {/* Severe stage note */}
              <div className="disclaimer-banner">
                <span>⚠️</span>
                <span className="text-xs">{prediction.severe_stage_note}</span>
              </div>

              {/* Explainability (XAI) */}
              <div className="glass-card p-6">
                <h3 className="font-bold text-white mb-4">Explainability (XAI) Overlays</h3>
                {prediction.xai_overlays && Object.keys(prediction.xai_overlays).length > 0 ? (
                  <div className="grid grid-cols-3 gap-4">
                    <div>
                      <p className="text-[10px] text-medical-teal-light font-bold mb-2 text-center uppercase tracking-wider">Grad-CAM</p>
                      <div className="border border-white/5 rounded-lg overflow-hidden bg-black/40">
                        <img
                          src={`data:image/png;base64,${prediction.xai_overlays.gradcam_overlay}`}
                          alt="Grad-CAM"
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                    <div>
                      <p className="text-[10px] text-medical-teal-light font-bold mb-2 text-center uppercase tracking-wider">Grad-CAM++</p>
                      <div className="border border-white/5 rounded-lg overflow-hidden bg-black/40">
                        <img
                          src={`data:image/png;base64,${prediction.xai_overlays.gradcam_plus_plus_overlay}`}
                          alt="Grad-CAM++"
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                    <div>
                      <p className="text-[10px] text-medical-teal-light font-bold mb-2 text-center uppercase tracking-wider">HiResCAM</p>
                      <div className="border border-white/5 rounded-lg overflow-hidden bg-black/40">
                        <img
                          src={`data:image/png;base64,${prediction.xai_overlays.hirescam_overlay}`}
                          alt="HiResCAM"
                          className="w-full h-auto"
                        />
                      </div>
                    </div>
                  </div>
                ) : (
                  <div className="text-center py-6 text-slate-400 text-sm">
                    ⚠️ Explainability overlays could not be computed.
                  </div>
                )}
              </div>

              {/* Clinical Report Section */}
              <div className="glass-card p-6">
                <h3 className="font-bold text-white mb-4">Clinical Report</h3>
                <div className="space-y-4">
                  {reportPreview && (
                    <div className="bg-white/5 rounded-xl p-4 border border-white/10 space-y-2">
                      <p className="text-xs text-medical-teal-light font-bold uppercase tracking-wider">Report Preview</p>
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
                          Generating...
                        </span>
                      ) : (
                        'Generate Report'
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
                        'Download Report'
                      )}
                    </button>
                  </div>
                </div>
              </div>

            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
}
