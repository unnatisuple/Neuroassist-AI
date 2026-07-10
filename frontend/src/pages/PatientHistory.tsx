import { useState, useEffect } from 'react';
import { patientAPI } from '../services/api';
import { motion } from 'framer-motion';

type Patient = {
  id: string;
  full_name: string;
  age: number;
  gender: string;
  medical_record_number: string;
  created_at: string;
  num_predictions: number;
  num_reports: number;
};

export default function PatientHistory() {
  const [patients, setPatients] = useState<Patient[]>([]);
  const [loading, setLoading] = useState(true);
  const [showForm, setShowForm] = useState(false);
  const [formData, setFormData] = useState({ full_name: '', age: '', gender: 'male', medical_record_number: '', clinical_notes: '' });

  useEffect(() => {
    fetchPatients();
  }, []);

  const fetchPatients = async () => {
    try {
      const res = await patientAPI.list();
      setPatients(res.data || []);
    } catch { /* May fail if DB not connected */ }
    finally { setLoading(false); }
  };

  const handleCreate = async (e: React.FormEvent) => {
    e.preventDefault();
    try {
      await patientAPI.create({
        full_name: formData.full_name,
        age: Number(formData.age),
        gender: formData.gender,
        medical_record_number: formData.medical_record_number,
        clinical_notes: formData.clinical_notes,
      });
      setShowForm(false);
      setFormData({ full_name: '', age: '', gender: 'male', medical_record_number: '', clinical_notes: '' });
      fetchPatients();
    } catch { alert('Failed to create patient record.'); }
  };

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="section-title">Patient Records</h1>
          <p className="section-subtitle">Clinical records managed by you — patients do not have platform access</p>
        </div>
        <button onClick={() => setShowForm(!showForm)} className="btn-primary">
          + New Patient Record
        </button>
      </div>

      {/* Create form */}
      {showForm && (
        <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: 'auto' }}>
          <form onSubmit={handleCreate} className="glass-card p-6 space-y-4">
            <h3 className="font-bold text-white">New Patient Record</h3>
            <div className="grid grid-cols-3 gap-4">
              <div>
                <label className="label-text">Full Name *</label>
                <input value={formData.full_name} onChange={e => setFormData({...formData, full_name: e.target.value})}
                  className="input-field" required />
              </div>
              <div>
                <label className="label-text">Age *</label>
                <input type="number" value={formData.age} onChange={e => setFormData({...formData, age: e.target.value})}
                  className="input-field" min="0" max="150" required />
              </div>
              <div>
                <label className="label-text">Gender *</label>
                <select value={formData.gender} onChange={e => setFormData({...formData, gender: e.target.value})}
                  className="input-field">
                  <option value="male">Male</option>
                  <option value="female">Female</option>
                  <option value="other">Other</option>
                </select>
              </div>
            </div>
            <div>
              <label className="label-text">Medical Record Number</label>
              <input value={formData.medical_record_number} onChange={e => setFormData({...formData, medical_record_number: e.target.value})}
                className="input-field" />
            </div>
            <div>
              <label className="label-text">Clinical Notes</label>
              <textarea value={formData.clinical_notes} onChange={e => setFormData({...formData, clinical_notes: e.target.value})}
                className="input-field min-h-[80px]" />
            </div>
            <div className="flex gap-3">
              <button type="submit" className="btn-primary">Create Record</button>
              <button type="button" onClick={() => setShowForm(false)} className="btn-secondary">Cancel</button>
            </div>
          </form>
        </motion.div>
      )}

      {/* Patient list */}
      {loading ? (
        <div className="flex items-center justify-center py-20">
          <div className="w-8 h-8 border-3 border-brand-500 border-t-transparent rounded-full animate-spin" />
        </div>
      ) : patients.length === 0 ? (
        <div className="glass-card p-12 text-center">
          <span className="text-5xl block mb-4">👥</span>
          <p className="text-white font-semibold">No patient records yet</p>
          <p className="text-slate-400 text-sm mt-2">Create a patient record to track their MRI analyses and reports over time.</p>
        </div>
      ) : (
        <div className="grid grid-cols-2 gap-4">
          {patients.map((p) => (
            <div key={p.id} className="glass-card p-5 hover:border-white/20 cursor-pointer transition-all">
              <div className="flex items-center justify-between mb-3">
                <h3 className="font-semibold text-white">{p.full_name}</h3>
                <span className="text-xs text-slate-500">{p.medical_record_number || 'No MRN'}</span>
              </div>
              <div className="flex gap-4 text-sm text-slate-400">
                <span>Age: {p.age}</span>
                <span>Gender: {p.gender}</span>
              </div>
              <div className="flex gap-4 mt-3">
                <span className="status-badge-info">{p.num_predictions} scans</span>
                <span className="status-badge-success">{p.num_reports} reports</span>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
