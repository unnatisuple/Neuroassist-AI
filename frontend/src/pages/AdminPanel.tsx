import { useState, useEffect } from 'react';
import { useAuth } from '../context/AuthContext';
import { adminAPI } from '../services/api';

type PendingDoctor = {
  id: string;
  full_name: string;
  email: string;
  specialization: string;
  medical_license_number: string;
  institution: string;
  created_at: string;
};

export default function AdminPanel() {
  const { isAdmin } = useAuth();
  const [pending, setPending] = useState<PendingDoctor[]>([]);
  const [stats, setStats] = useState<Record<string, unknown> | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    const fetch = async () => {
      try {
        const [pendingRes, statsRes] = await Promise.allSettled([
          adminAPI.listPendingDoctors(),
          adminAPI.getStats(),
        ]);
        if (pendingRes.status === 'fulfilled') setPending(pendingRes.value.data.pending_doctors || []);
        if (statsRes.status === 'fulfilled') setStats(statsRes.value.data);
      } catch { /* Admin data may fail */ }
      finally { setLoading(false); }
    };
    fetch();
  }, []);

  const handleVerify = async (doctorId: string, action: 'approve' | 'reject') => {
    try {
      await adminAPI.verifyDoctor({ doctor_id: doctorId, action });
      setPending(prev => prev.filter(d => d.id !== doctorId));
    } catch { alert(`Failed to ${action} doctor.`); }
  };

  if (!isAdmin) {
    return (
      <div className="glass-card p-12 text-center">
        <span className="text-5xl block mb-4">🔒</span>
        <p className="text-white font-semibold">Administrator Access Required</p>
        <p className="text-slate-400 text-sm mt-2">This page is restricted to platform administrators.</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="section-title">Admin Panel</h1>
        <p className="section-subtitle">Doctor verification & platform management</p>
      </div>

      {/* Stats */}
      {stats && (
        <div className="grid grid-cols-4 gap-4">
          {[
            { label: 'Total Doctors', value: (stats.doctors as Record<string, number>)?.total || 0 },
            { label: 'Pending Verification', value: (stats.doctors as Record<string, number>)?.pending || 0 },
            { label: 'Total Predictions', value: stats.predictions as number || 0 },
            { label: 'Total Reports', value: stats.reports as number || 0 },
          ].map(({ label, value }) => (
            <div key={label} className="glass-card p-5">
              <p className="text-sm text-slate-400">{label}</p>
              <p className="text-2xl font-bold text-white mt-1">{value}</p>
            </div>
          ))}
        </div>
      )}

      {/* Pending doctors */}
      <div>
        <h2 className="font-bold text-white text-lg mb-4">Pending Doctor Verifications ({pending.length})</h2>
        {loading ? (
          <div className="flex justify-center py-10">
            <div className="w-8 h-8 border-3 border-brand-500 border-t-transparent rounded-full animate-spin" />
          </div>
        ) : pending.length === 0 ? (
          <div className="glass-card p-8 text-center">
            <p className="text-slate-400">No pending verifications.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {pending.map((doc) => (
              <div key={doc.id} className="glass-card p-5 flex items-center justify-between">
                <div>
                  <h3 className="font-semibold text-white">{doc.full_name}</h3>
                  <p className="text-sm text-slate-400">{doc.email} · {doc.specialization}</p>
                  <p className="text-xs text-slate-500 mt-1">
                    License: <span className="font-mono text-white">{doc.medical_license_number}</span>
                    {doc.institution && ` · ${doc.institution}`}
                  </p>
                </div>
                <div className="flex gap-2">
                  <button onClick={() => handleVerify(doc.id, 'approve')} className="btn-primary text-sm py-2">
                    ✓ Approve
                  </button>
                  <button onClick={() => handleVerify(doc.id, 'reject')} className="btn-danger text-sm py-2">
                    ✗ Reject
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
