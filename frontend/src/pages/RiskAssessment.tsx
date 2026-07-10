import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { riskAPI } from '../services/api';
import { motion } from 'framer-motion';

type RiskResult = {
  assessment_id: string;
  risk_score: number;
  risk_level: string;
  risk_factors: Array<{ factor: string; value: unknown; contribution: number; explanation: string }>;
  protective_factors: Array<{ factor: string; value: unknown; contribution: number; explanation: string }>;
  methodology: string;
  trace_id: string;
};

export default function RiskAssessment() {
  const { register, handleSubmit } = useForm();
  const [result, setResult] = useState<RiskResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const onSubmit = async (data: Record<string, unknown>) => {
    setError('');
    setLoading(true);
    try {
      // Convert checkbox strings to booleans, empty strings to null
      const cleaned = {
        ...data,
        age: Number(data.age),
        mmse_score: data.mmse_score ? Number(data.mmse_score) : null,
        moca_score: data.moca_score ? Number(data.moca_score) : null,
        systolic_bp: data.systolic_bp ? Number(data.systolic_bp) : null,
        diastolic_bp: data.diastolic_bp ? Number(data.diastolic_bp) : null,
        bmi: data.bmi ? Number(data.bmi) : null,
        education_years: data.education_years ? Number(data.education_years) : null,
        has_diabetes: data.has_diabetes === true || data.has_diabetes === 'true',
        family_history_dementia: data.family_history_dementia === true || data.family_history_dementia === 'true',
        depression_history: data.depression_history === true || data.depression_history === 'true',
      };

      const res = await riskAPI.assess(cleaned);
      setResult(res.data);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } } };
      setError(axiosErr.response?.data?.detail || 'Risk assessment failed.');
    } finally {
      setLoading(false);
    }
  };

  const riskColors: Record<string, string> = {
    low: 'text-emerald-400',
    moderate: 'text-yellow-400',
    high: 'text-orange-400',
    very_high: 'text-red-400',
  };

  return (
    <div className="space-y-8">
      <div>
        <h1 className="section-title">Clinical Risk Assessment</h1>
        <p className="section-subtitle">Transparent, rule-based risk index (not ML-based)</p>
      </div>

      <div className="disclaimer-banner">
        <span>ℹ️</span>
        <span>This risk score uses a transparent, rule-based algorithm based on published clinical literature (Livingston et al. 2020). It is NOT an ML prediction.</span>
      </div>

      {error && (
        <div className="bg-red-500/10 border border-red-500/30 rounded-xl px-4 py-3 text-red-300 text-sm">{error}</div>
      )}

      <div className="grid grid-cols-2 gap-8">
        {/* Form */}
        <form onSubmit={handleSubmit(onSubmit)} className="glass-card p-6 space-y-4">
          <h3 className="font-bold text-white">Patient Clinical Data</h3>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">Age *</label>
              <input type="number" {...register('age', { required: true })} className="input-field" min="18" max="120" />
            </div>
            <div>
              <label className="label-text">Gender *</label>
              <select {...register('gender', { required: true })} className="input-field">
                <option value="male">Male</option>
                <option value="female">Female</option>
                <option value="other">Other</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">MMSE Score (0-30)</label>
              <input type="number" {...register('mmse_score')} className="input-field" min="0" max="30" step="0.5" />
            </div>
            <div>
              <label className="label-text">MoCA Score (0-30)</label>
              <input type="number" {...register('moca_score')} className="input-field" min="0" max="30" step="0.5" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">Systolic BP</label>
              <input type="number" {...register('systolic_bp')} className="input-field" min="60" max="250" />
            </div>
            <div>
              <label className="label-text">Diastolic BP</label>
              <input type="number" {...register('diastolic_bp')} className="input-field" min="30" max="150" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">BMI</label>
              <input type="number" {...register('bmi')} className="input-field" min="10" max="60" step="0.1" />
            </div>
            <div>
              <label className="label-text">Education (years)</label>
              <input type="number" {...register('education_years')} className="input-field" min="0" max="30" />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">Cholesterol</label>
              <select {...register('cholesterol_level')} className="input-field">
                <option value="">Unknown</option>
                <option value="normal">Normal</option>
                <option value="borderline">Borderline</option>
                <option value="high">High</option>
              </select>
            </div>
            <div>
              <label className="label-text">Smoking Status</label>
              <select {...register('smoking_status')} className="input-field">
                <option value="">Unknown</option>
                <option value="never">Never</option>
                <option value="former">Former</option>
                <option value="current">Current</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="label-text">Alcohol Consumption</label>
              <select {...register('alcohol_consumption')} className="input-field">
                <option value="">Unknown</option>
                <option value="none">None</option>
                <option value="moderate">Moderate</option>
                <option value="heavy">Heavy</option>
              </select>
            </div>
            <div>
              <label className="label-text">Physical Activity</label>
              <select {...register('physical_activity_level')} className="input-field">
                <option value="">Unknown</option>
                <option value="sedentary">Sedentary</option>
                <option value="light">Light</option>
                <option value="moderate">Moderate</option>
                <option value="active">Active</option>
              </select>
            </div>
          </div>

          <div>
            <label className="label-text">Sleep Quality</label>
            <select {...register('sleep_quality')} className="input-field">
              <option value="">Unknown</option>
              <option value="poor">Poor</option>
              <option value="fair">Fair</option>
              <option value="good">Good</option>
              <option value="excellent">Excellent</option>
            </select>
          </div>

          <div className="space-y-3 pt-2">
            <label className="flex items-center gap-3 text-sm text-slate-300 cursor-pointer">
              <input type="checkbox" {...register('has_diabetes')} className="rounded" />
              Has Diabetes
            </label>
            <label className="flex items-center gap-3 text-sm text-slate-300 cursor-pointer">
              <input type="checkbox" {...register('family_history_dementia')} className="rounded" />
              Family History of Dementia
            </label>
            <label className="flex items-center gap-3 text-sm text-slate-300 cursor-pointer">
              <input type="checkbox" {...register('depression_history')} className="rounded" />
              History of Depression
            </label>
          </div>

          <button type="submit" disabled={loading} className="btn-primary w-full mt-4">
            {loading ? 'Computing Risk Score...' : 'Calculate Risk Assessment'}
          </button>
        </form>

        {/* Results */}
        {result && (
          <motion.div initial={{ opacity: 0, x: 20 }} animate={{ opacity: 1, x: 0 }} className="space-y-4">
            <div className="glass-card-elevated p-6 text-center">
              <p className="text-sm text-slate-400">Risk Score</p>
              <p className={`text-5xl font-extrabold mt-2 ${riskColors[result.risk_level]}`}>
                {result.risk_score}
              </p>
              <p className={`text-lg font-semibold mt-1 ${riskColors[result.risk_level]}`}>
                {result.risk_level.replace('_', ' ').toUpperCase()}
              </p>
              <p className="text-xs text-slate-500 mt-3">{result.methodology}</p>
            </div>

            {result.risk_factors.length > 0 && (
              <div className="glass-card p-6">
                <h3 className="font-bold text-red-400 mb-3">⬆ Risk Factors</h3>
                <div className="space-y-3">
                  {result.risk_factors.map((f, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span className="text-xs font-bold text-red-400 bg-red-500/10 px-2 py-0.5 rounded-full">+{f.contribution}</span>
                      <div>
                        <p className="text-sm text-white font-medium">{f.factor}: {String(f.value)}</p>
                        <p className="text-xs text-slate-400">{f.explanation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}

            {result.protective_factors.length > 0 && (
              <div className="glass-card p-6">
                <h3 className="font-bold text-emerald-400 mb-3">⬇ Protective Factors</h3>
                <div className="space-y-3">
                  {result.protective_factors.map((f, i) => (
                    <div key={i} className="flex items-start gap-3">
                      <span className="text-xs font-bold text-emerald-400 bg-emerald-500/10 px-2 py-0.5 rounded-full">{f.contribution}</span>
                      <div>
                        <p className="text-sm text-white font-medium">{f.factor}: {String(f.value)}</p>
                        <p className="text-xs text-slate-400">{f.explanation}</p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </div>
    </div>
  );
}
