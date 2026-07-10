import { Outlet, NavLink, useNavigate } from 'react-router-dom';
import { useAuth } from '../../context/AuthContext';
import { motion } from 'framer-motion';

const navItems = [
  { path: '/dashboard', label: 'Dashboard', icon: '📊' },
  { path: '/mri-analysis', label: 'MRI Analysis', icon: '🧠' },
  { path: '/risk-assessment', label: 'Risk Assessment', icon: '⚠️' },
  { path: '/reports', label: 'Reports', icon: '📄' },
  { path: '/patients', label: 'Patients', icon: '👥' },
  { path: '/chatbot', label: 'AI Assistant', icon: '💬' },
];

export default function Layout() {
  const { doctor, isAdmin, logout } = useAuth();
  const navigate = useNavigate();

  return (
    <div className="min-h-screen flex bg-medical-navy">
      {/* ---- Sidebar ---- */}
      <aside className="w-64 bg-medical-navy-light/50 backdrop-blur-xl border-r border-white/5 flex flex-col fixed h-full z-20">
        {/* Logo */}
        <div className="p-6 border-b border-white/5">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-brand-gradient flex items-center justify-center text-xl">
              🧠
            </div>
            <div>
              <h1 className="text-lg font-bold text-white leading-tight">NeuroAssist AI</h1>
              <p className="text-[10px] text-amber-400 font-medium">INVESTIGATIONAL</p>
            </div>
          </div>
        </div>

        {/* Navigation */}
        <nav className="flex-1 py-4 px-3 space-y-1 overflow-y-auto">
          {navItems.map(({ path, label, icon }) => (
            <NavLink
              key={path}
              to={path}
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-brand-600/20 text-brand-300 border border-brand-500/30'
                    : 'text-slate-400 hover:bg-white/5 hover:text-white'
                }`
              }
            >
              <span className="text-lg">{icon}</span>
              {label}
            </NavLink>
          ))}

          {isAdmin && (
            <NavLink
              to="/admin"
              className={({ isActive }) =>
                `flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all duration-200 ${
                  isActive
                    ? 'bg-amber-600/20 text-amber-300 border border-amber-500/30'
                    : 'text-slate-400 hover:bg-white/5 hover:text-amber-300'
                }`
              }
            >
              <span className="text-lg">⚙️</span>
              Admin Panel
            </NavLink>
          )}
        </nav>

        {/* Doctor profile */}
        <div className="p-4 border-t border-white/5">
          <div className="flex items-center gap-3 mb-3">
            <div className="w-9 h-9 rounded-full bg-brand-600/30 flex items-center justify-center text-sm font-bold text-brand-300">
              {doctor?.full_name?.charAt(0) || 'D'}
            </div>
            <div className="flex-1 min-w-0">
              <p className="text-sm font-medium text-white truncate">{doctor?.full_name}</p>
              <p className="text-xs text-slate-500 truncate">{doctor?.specialization}</p>
            </div>
          </div>
          <button
            onClick={() => { logout(); navigate('/login'); }}
            className="w-full text-xs text-slate-400 hover:text-red-400 transition-colors py-2 rounded-lg hover:bg-red-500/10"
          >
            Sign Out
          </button>
        </div>
      </aside>

      {/* ---- Main content ---- */}
      <main className="flex-1 ml-64">
        {/* Regulatory disclaimer header */}
        <div className="bg-amber-500/5 border-b border-amber-500/20 px-6 py-2">
          <p className="text-[11px] text-amber-400/80 text-center">
            ⚠️ Investigational software. Not a substitute for clinical judgment. Not FDA/CE cleared. Decision-support mode only.
          </p>
        </div>

        {/* Page content */}
        <motion.div
          initial={{ opacity: 0, y: 10 }}
          animate={{ opacity: 1, y: 0 }}
          transition={{ duration: 0.3 }}
          className="page-container"
        >
          <Outlet />
        </motion.div>

        {/* Footer */}
        <footer className="border-t border-white/5 px-6 py-4 text-center">
          <p className="text-[10px] text-slate-600">
            NeuroAssist AI v2.0 · Clinician Portal · Decision Support Mode ·{' '}
            <span className="text-amber-500/60">Investigational — Not for autonomous diagnosis</span>
          </p>
        </footer>
      </main>
    </div>
  );
}
