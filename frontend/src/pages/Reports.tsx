import { useState, useEffect } from 'react';
import { reportAPI, mriAPI } from '../services/api';
import { motion, AnimatePresence } from 'framer-motion';

type Report = {
  report_id: string;
  prediction_id: string;
  patient_id?: string;
  language: string;
  summary: string;
  created_at: string;
};

type Scan = {
  prediction_id: string;
  mri_file_id: string;
  predicted_class: string;
  confidence: number;
  class_probabilities: Record<string, number>;
  model_version: string;
  created_at: string;
  patient_id?: string;
  report_id?: string | null;
  xai_overlays?: Record<string, string>;
};

export default function Reports() {
  const [reports, setReports] = useState<Report[]>([]);
  const [scans, setScans] = useState<Scan[]>([]);
  const [loadingReports, setLoadingReports] = useState(true);
  const [loadingScans, setLoadingScans] = useState(true);
  const [activeTab, setActiveTab] = useState<'scans' | 'reports'>('scans');

  // Filtering / Sorting
  const [stageFilter, setStageFilter] = useState('all');
  const [sortBy, setSortBy] = useState('newest'); // newest, oldest, confidence

  // Action states
  const [generatingReportId, setGeneratingReportId] = useState<string | null>(null);
  const [downloadingReportId, setDownloadingReportId] = useState<string | null>(null);
  
  // Localized error maps to prevent global pollution
  const [pageError, setPageError] = useState('');
  const [scanErrors, setScanErrors] = useState<Record<string, string>>({});

  useEffect(() => {
    fetchReports();
    fetchScans();
  }, []);

  const fetchReports = async () => {
    try {
      const res = await reportAPI.list();
      setReports(res.data.reports || []);
    } catch {
      console.warn('Failed to load reports.');
    } finally {
      setLoadingReports(false);
    }
  };

  const fetchScans = async () => {
    try {
      const res = await mriAPI.list();
      setScans(res.data.predictions || []);
    } catch {
      console.warn('Failed to load scans.');
    } finally {
      setLoadingScans(false);
    }
  };

  const handleDownload = async (reportId: string) => {
    setDownloadingReportId(reportId);
    setPageError('');
    try {
      const res = await reportAPI.download(reportId);
      const blob = new Blob([res.data], { type: 'application/pdf' });
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `NeuroAssist_Report_${reportId.slice(0, 8)}.pdf`;
      document.body.appendChild(a);
      a.click();
      a.remove();
      window.URL.revokeObjectURL(url);
    } catch {
      setPageError('PDF download failed. Please try again.');
    } finally {
      setDownloadingReportId(null);
    }
  };

  const handleGenerateReportForScan = async (scan: Scan) => {
    setGeneratingReportId(scan.prediction_id);
    
    // Clear previous error for this scan card
    setScanErrors(prev => {
      const next = { ...prev };
      delete next[scan.prediction_id];
      return next;
    });

    try {
      const res = await reportAPI.generate({
        prediction_id: scan.prediction_id,
        patient_id: scan.patient_id || undefined,
      });
      // Refresh lists
      await Promise.all([fetchReports(), fetchScans()]);
      alert(`Report generated successfully for scan! ID: ${res.data.report_id.slice(0, 8)}`);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      const detail = axiosErr.response?.data?.detail || 'Report generation failed.';
      setScanErrors(prev => ({ ...prev, [scan.prediction_id]: detail }));
    } finally {
      setGeneratingReportId(null);
    }
  };

  // Filter and sort logic
  const filteredScans = scans
    .filter((scan) => {
      if (stageFilter === 'all') return true;
      return scan.predicted_class.toLowerCase() === stageFilter.toLowerCase();
    })
    .sort((a, b) => {
      if (sortBy === 'newest') {
        return new Date(b.created_at).getTime() - new Date(a.created_at).getTime();
      }
      if (sortBy === 'oldest') {
        return new Date(a.created_at).getTime() - new Date(b.created_at).getTime();
      }
      if (sortBy === 'confidence') {
        return b.confidence - a.confidence;
      }
      return 0;
    });

  const stageColors: Record<string, string> = {
    'Non-Demented': 'status-badge-success',
    'Very Mild Demented': 'status-badge-info',
    'Mild Demented': 'status-badge-warning',
    'Moderate Demented': 'status-badge-danger',
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="section-title">Reports & History</h1>
          <p className="section-subtitle">Manage patient scan files and generated clinical reports</p>
        </div>

        {/* Tabs */}
        <div className="flex bg-white/5 border border-white/5 p-1 rounded-xl">
          <button
            onClick={() => setActiveTab('scans')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
              activeTab === 'scans' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            🧠 Scan History
          </button>
          <button
            onClick={() => setActiveTab('reports')}
            className={`px-4 py-2 text-sm font-semibold rounded-lg transition-all ${
              activeTab === 'reports' ? 'bg-brand-600 text-white' : 'text-slate-400 hover:text-white'
            }`}
          >
            📄 Clinical Reports
          </button>
        </div>
      </div>

      {pageError && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-300 text-sm">
          {pageError}
        </div>
      )}

      {/* Filter panel for Scan History */}
      {activeTab === 'scans' && (
        <div className="glass-card p-4 flex gap-4 items-center">
          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 uppercase font-bold">Stage:</span>
            <select
              value={stageFilter}
              onChange={(e) => setStageFilter(e.target.value)}
              className="input-field py-1 px-3 text-xs w-48"
            >
              <option value="all">All Stages</option>
              <option value="Non-Demented">Non-Demented</option>
              <option value="Very Mild Demented">Very Mild Demented</option>
              <option value="Mild Demented">Mild Demented</option>
              <option value="Moderate Demented">Moderate Demented</option>
            </select>
          </div>

          <div className="flex items-center gap-2">
            <span className="text-xs text-slate-400 uppercase font-bold">Sort By:</span>
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value)}
              className="input-field py-1 px-3 text-xs w-40"
            >
              <option value="newest">Newest Scans</option>
              <option value="oldest">Oldest Scans</option>
              <option value="confidence">Highest Confidence</option>
            </select>
          </div>
        </div>
      )}

      {/* Tab Panels */}
      <AnimatePresence mode="wait">
        {activeTab === 'scans' ? (
          <motion.div
            key="scans-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            {loadingScans ? (
              <div className="flex items-center justify-center py-20">
                <div className="w-8 h-8 border-3 border-brand-500 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : filteredScans.length === 0 ? (
              <div className="glass-card p-12 text-center">
                <span className="text-5xl block mb-4">🧠</span>
                <p className="text-white font-semibold">No scans found</p>
                <p className="text-slate-400 text-sm mt-2">Upload a scan on the MRI Analysis page to begin tracking history.</p>
              </div>
            ) : (
              <div className="grid grid-cols-1 gap-4">
                {filteredScans.map((scan) => (
                  <div key={scan.prediction_id} className="glass-card p-4 flex flex-col gap-3 hover:border-white/10 transition-all">
                    <div className="flex gap-5 items-center">
                      {/* Thumbnail: Streamed directly with secure JWT query token */}
                      <div className="w-20 h-20 rounded-lg overflow-hidden bg-black/40 border border-white/5 flex-shrink-0 flex items-center justify-center">
                        <img
                          src={`/api/mri/file/${scan.prediction_id}?token=${localStorage.getItem('neuroassist_token')}`}
                          alt="Scan Thumbnail"
                          className="w-full h-full object-cover"
                          onError={(e) => {
                            (e.target as HTMLImageElement).src = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="100" height="100"><rect width="100" height="100" fill="%231E293B"/><text x="50" y="55" fill="%2394A3B8" font-size="12" text-anchor="middle">No Scan</text></svg>';
                          }}
                        />
                      </div>

                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-3 mb-1">
                          <span className={`status-badge ${stageColors[scan.predicted_class] || 'status-badge-info'}`}>
                            {scan.predicted_class}
                          </span>
                          <span className="text-xs text-white font-bold">{(scan.confidence * 100).toFixed(1)}%</span>
                        </div>
                        <p className="text-xs text-slate-400 truncate">
                          ID: <span className="font-mono">{scan.prediction_id.slice(0, 8)}</span> | Model: {scan.model_version}
                        </p>
                        <p className="text-[11px] text-slate-500 mt-1">
                          Analyzed: {new Date(scan.created_at).toLocaleString()}
                        </p>
                      </div>

                      <div className="flex gap-3">
                        {scan.report_id ? (
                          <button
                            onClick={() => handleDownload(scan.report_id!)}
                            disabled={downloadingReportId === scan.report_id}
                            className="btn-primary text-xs py-2 px-4 flex items-center gap-1"
                          >
                            {downloadingReportId === scan.report_id ? (
                              <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                            ) : (
                              '📥'
                            )}
                            Download PDF
                          </button>
                        ) : (
                          <button
                            onClick={() => handleGenerateReportForScan(scan)}
                            disabled={generatingReportId === scan.prediction_id}
                            className="btn-secondary text-xs py-2 px-4 flex items-center gap-1 hover:border-brand-500 hover:text-brand-300"
                          >
                            {generatingReportId === scan.prediction_id ? (
                              <span className="w-3.5 h-3.5 border-2 border-brand-500/30 border-t-brand-500 rounded-full animate-spin" />
                            ) : (
                              '📝'
                            )}
                            Generate Report
                          </button>
                        )}
                      </div>
                    </div>

                    {/* Local Scan Report Generation Error Banner */}
                    {scanErrors[scan.prediction_id] && (
                      <div className="bg-red-500/10 border border-red-500/30 rounded-lg px-3 py-2 text-red-300 text-xs">
                        ⚠️ {scanErrors[scan.prediction_id]}
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        ) : (
          <motion.div
            key="reports-panel"
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -10 }}
            className="space-y-4"
          >
            <div className="disclaimer-banner">
              <span>⚠️</span>
              <span className="text-xs">All reports include regulatory disclaimers. Reports are for decision-support purposes only.</span>
            </div>

            {loadingReports ? (
              <div className="flex items-center justify-center py-20">
                <div className="w-8 h-8 border-3 border-brand-500 border-t-transparent rounded-full animate-spin" />
              </div>
            ) : reports.length === 0 ? (
              <div className="glass-card p-12 text-center">
                <span className="text-5xl block mb-4">📄</span>
                <p className="text-white font-semibold">No reports generated yet</p>
                <p className="text-slate-400 text-sm mt-2">Run an MRI analysis first, then generate a clinical report from the results.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {reports.map((report) => (
                  <div key={report.report_id} className="glass-card p-4 flex items-center justify-between hover:border-white/10 transition-all">
                    <div className="flex-1 min-w-0">
                      <div className="flex items-center gap-3 mb-1">
                        <h3 className="font-semibold text-white">Report {report.report_id.slice(0, 8)}</h3>
                        <span className="status-badge-info">{report.language.toUpperCase()}</span>
                      </div>
                      <p className="text-sm text-slate-400 line-clamp-1">{report.summary || 'No summary available'}</p>
                      <p className="text-xs text-slate-500 mt-1">
                        Generated: {new Date(report.created_at).toLocaleString()}
                      </p>
                    </div>
                    <button
                      onClick={() => handleDownload(report.report_id)}
                      disabled={downloadingReportId === report.report_id}
                      className="btn-secondary text-xs ml-4 flex items-center gap-1"
                    >
                      {downloadingReportId === report.report_id ? (
                        <span className="w-3.5 h-3.5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                      ) : (
                        '📥'
                      )}
                      Download PDF
                    </button>
                  </div>
                ))}
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}
