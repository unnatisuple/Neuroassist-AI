import { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';
import { motion } from 'framer-motion';

export default function Login() {
  const [fullName, setFullName] = useState('');
  const [email, setEmail] = useState('');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const { login } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setLoading(true);

    try {
      await login(fullName, email);
      navigate('/dashboard');
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || 'Authentication failed. Please try again.');
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center bg-medical-navy relative overflow-hidden">
      {/* Background effects */}
      <div className="absolute inset-0">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-brand-600/10 rounded-full blur-3xl" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-medical-teal/10 rounded-full blur-3xl" />
      </div>

      <motion.div
        initial={{ opacity: 0, y: 30 }}
        animate={{ opacity: 1, y: 0 }}
        transition={{ duration: 0.6, ease: 'easeOut' }}
        className="relative z-10 w-full max-w-md px-4"
      >
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-16 h-16 rounded-2xl bg-brand-gradient shadow-neon mb-4">
            <span className="text-3xl">🧠</span>
          </div>
          <h1 className="text-3xl font-extrabold text-white">NeuroAssist AI</h1>
          <p className="text-medical-teal-light font-semibold mt-1">Clinician Portal</p>
          <p className="text-slate-500 text-sm mt-2">
            Secure access for licensed medical professionals only
          </p>
        </div>

        {/* Login form */}
        <div className="glass-card p-8">
          <h2 className="text-xl font-bold text-white mb-6">Clinician Sign In</h2>

          {error && (
            <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 mb-4 text-red-300 text-sm">
              {error}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-5">
            <div>
              <label className="label-text">Full Name</label>
              <input
                type="text"
                value={fullName}
                onChange={(e) => setFullName(e.target.value)}
                className="input-field"
                placeholder="Dr. Jane Doe"
                required
                autoFocus
              />
            </div>

            <div>
              <label className="label-text">Email Address</label>
              <input
                type="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="input-field"
                placeholder="doctor@hospital.org"
                required
              />
            </div>

            <button
              type="submit"
              disabled={loading}
              className="btn-primary w-full"
            >
              {loading ? (
                <span className="flex items-center justify-center gap-2">
                  <span className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                  Authenticating...
                </span>
              ) : (
                'Sign In to Clinician Portal'
              )}
            </button>
          </form>
        </div>

        {/* Disclaimer */}
        <div className="disclaimer-banner mt-6">
          <span>⚠️</span>
          <span>
            Investigational software for authorized clinicians only.
            Not FDA/CE cleared. Not a substitute for clinical judgment.
          </span>
        </div>
      </motion.div>
    </div>
  );
}
