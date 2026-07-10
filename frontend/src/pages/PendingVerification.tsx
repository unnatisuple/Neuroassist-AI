import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

export default function PendingVerification() {
  const { doctor, logout } = useAuth();

  return (
    <div className="page-container flex items-center justify-center min-h-[60vh]">
      <motion.div
        initial={{ opacity: 0, scale: 0.95 }}
        animate={{ opacity: 1, scale: 1 }}
        className="glass-card p-10 max-w-md text-center"
      >
        <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-amber-500/20 flex items-center justify-center">
          <span className="text-3xl">⏳</span>
        </div>

        <h2 className="text-2xl font-bold text-white mb-3">Account Pending Verification</h2>

        <p className="text-slate-400 mb-6">
          Your medical license (<span className="text-white font-mono">{doctor?.medical_license_number}</span>)
          is being reviewed by a platform administrator. You'll gain access to clinical features
          once your credentials are verified.
        </p>

        <div className="bg-white/5 rounded-xl p-4 text-left space-y-2 mb-6">
          <p className="text-sm text-slate-300">
            <span className="text-slate-500">Name:</span> {doctor?.full_name}
          </p>
          <p className="text-sm text-slate-300">
            <span className="text-slate-500">Status:</span>{' '}
            <span className="status-badge-warning">Pending Verification</span>
          </p>
        </div>

        <button onClick={logout} className="btn-secondary text-sm">
          Sign Out
        </button>
      </motion.div>
    </div>
  );
}
