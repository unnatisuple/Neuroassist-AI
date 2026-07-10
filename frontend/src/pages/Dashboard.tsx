import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { healthAPI, patientAPI, reportAPI } from '../services/api';
import { motion } from 'framer-motion';
import { useNavigate } from 'react-router-dom';

const fadeUp = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0 },
};

export default function Dashboard() {
  const { doctor } = useAuth();
  const navigate = useNavigate();
  const [stats, setStats] = useState({ patients: 0, reports: 0, modelLoaded: false, dbConnected: false });

  useEffect(() => {
    const fetchStats = async () => {
      try {
        const [healthRes, patientsRes, reportsRes] = await Promise.allSettled([
          healthAPI.check(),
          patientAPI.list(),
          reportAPI.list(),
        ]);

        setStats({
          modelLoaded: healthRes.status === 'fulfilled' ? healthRes.value.data.model_loaded : false,
          dbConnected: healthRes.status === 'fulfilled' ? healthRes.value.data.mongodb_connected : false,
          patients: patientsRes.status === 'fulfilled' ? patientsRes.value.data.length : 0,
          reports: reportsRes.status === 'fulfilled' ? reportsRes.value.data.reports.length : 0,
        });
      } catch { /* Dashboard still shows even if stats fail */ }
    };
    fetchStats();
  }, []);

  const quickActions = [
    { icon: '🧠', label: 'New MRI Analysis', desc: 'Upload and analyze a brain MRI scan', path: '/mri-analysis', color: 'from-brand-600 to-brand-800' },
    { icon: '⚠️', label: 'Risk Assessment', desc: 'Evaluate clinical risk factors', path: '/risk-assessment', color: 'from-amber-600 to-amber-800' },
    { icon: '📄', label: 'Generate Report', desc: 'Create RAG-grounded clinical report', path: '/reports', color: 'from-emerald-600 to-emerald-800' },
    { icon: '💬', label: 'AI Assistant', desc: 'Ask about Alzheimer\'s & dementia', path: '/chatbot', color: 'from-medical-teal to-medical-teal-dark' },
  ];

  return (
    <div className="space-y-8">
      {/* Welcome */}
      <motion.div {...fadeUp} transition={{ delay: 0 }}>
        <h1 className="text-3xl font-extrabold text-white">
          Welcome, Dr. {doctor?.full_name?.split(' ').pop()}
        </h1>
        <p className="text-slate-400 mt-1">Clinical Decision Support Dashboard</p>
      </motion.div>

      {/* System Status */}
      <motion.div {...fadeUp} transition={{ delay: 0.1 }} className="grid grid-cols-4 gap-4">
        <div className="glass-card p-5">
          <p className="text-sm text-slate-400">AI Model</p>
          <div className="flex items-center gap-2 mt-2">
            <div className={`w-2.5 h-2.5 rounded-full ${stats.modelLoaded ? 'bg-emerald-400' : 'bg-amber-400 animate-pulse'}`} />
            <span className="text-white font-semibold">{stats.modelLoaded ? 'Loaded' : 'Not Loaded'}</span>
          </div>
          {!stats.modelLoaded && (
            <p className="text-xs text-amber-400/70 mt-2">Train the model first — no mock predictions will be shown.</p>
          )}
        </div>

        <div className="glass-card p-5">
          <p className="text-sm text-slate-400">Database</p>
          <div className="flex items-center gap-2 mt-2">
            <div className={`w-2.5 h-2.5 rounded-full ${stats.dbConnected ? 'bg-emerald-400' : 'bg-red-400'}`} />
            <span className="text-white font-semibold">{stats.dbConnected ? 'Connected' : 'Disconnected'}</span>
          </div>
        </div>

        <div className="glass-card p-5">
          <p className="text-sm text-slate-400">Patient Records</p>
          <p className="text-2xl font-bold text-white mt-2">{stats.patients}</p>
        </div>

        <div className="glass-card p-5">
          <p className="text-sm text-slate-400">Reports Generated</p>
          <p className="text-2xl font-bold text-white mt-2">{stats.reports}</p>
        </div>
      </motion.div>

      {/* Quick Actions */}
      <motion.div {...fadeUp} transition={{ delay: 0.2 }}>
        <h2 className="section-title mb-4">Quick Actions</h2>
        <div className="grid grid-cols-2 gap-4">
          {quickActions.map(({ icon, label, desc, path, color }) => (
            <button
              key={path}
              onClick={() => navigate(path)}
              className="glass-card p-6 text-left hover:border-white/20 group transition-all duration-300"
            >
              <div className={`w-12 h-12 rounded-xl bg-gradient-to-br ${color} flex items-center justify-center text-2xl mb-4 group-hover:scale-110 transition-transform`}>
                {icon}
              </div>
              <h3 className="text-lg font-bold text-white">{label}</h3>
              <p className="text-sm text-slate-400 mt-1">{desc}</p>
            </button>
          ))}
        </div>
      </motion.div>

      {/* Model info notice */}
      <motion.div {...fadeUp} transition={{ delay: 0.3 }}>
        <div className="glass-card p-5">
          <h3 className="font-semibold text-white mb-2">📋 Model Information</h3>
          <ul className="text-sm text-slate-400 space-y-1.5">
            <li>• <strong className="text-white">4-class classification:</strong> Non-Demented, Very Mild, Mild, Moderate</li>
            <li>• <span className="text-amber-400">⚠ Severe Dementia</span> classification is NOT supported by the current training data</li>
            <li>• All predictions include model version, checksum, and trace ID for auditability</li>
            <li>• XAI visualizations are generated live from actual model activations</li>
          </ul>
        </div>
      </motion.div>
    </div>
  );
}
