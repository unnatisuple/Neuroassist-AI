import axios from 'axios';

const API_BASE = import.meta.env.VITE_API_BASE_URL || '/api';

const api = axios.create({
  baseURL: API_BASE,
  timeout: 60000,
  headers: { 'Content-Type': 'application/json' },
});

// ---- Request interceptor: attach JWT token ----
api.interceptors.request.use((config) => {
  const token = localStorage.getItem('neuroassist_token');
  if (token) {
    config.headers.Authorization = `Bearer ${token}`;
  }
  return config;
});

// ---- Response interceptor: handle auth errors ----
api.interceptors.response.use(
  (response) => response,
  (error) => {
    if (error.response?.status === 401) {
      localStorage.removeItem('neuroassist_token');
      localStorage.removeItem('neuroassist_refresh');
      localStorage.removeItem('neuroassist_doctor');
      window.location.href = '/login';
    }
    return Promise.reject(error);
  }
);

// ========================================
// Auth API
// ========================================

export const authAPI = {
  login: (data: { full_name: string; email: string }) =>
    api.post('/auth/login', data),

  getProfile: () => api.get('/auth/me'),

  refreshToken: (refresh_token: string) =>
    api.post('/auth/refresh', null, { params: { refresh_token_str: refresh_token } }),
};

// ========================================
// MRI API
// ========================================

export const mriAPI = {
  upload: (file: File, patientId?: string) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/mri/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
      params: patientId ? { patient_id: patientId } : {},
    });
  },

  predict: (fileId: string, patientId?: string) =>
    api.post('/mri/predict', null, {
      params: { file_id: fileId, ...(patientId ? { patient_id: patientId } : {}) },
    }),

  explain: (method: string, predictionId: string) =>
    api.post(`/mri/explain/${method}`, null, {
      params: { prediction_id: predictionId },
    }),

  compare: (predictionId: string) =>
    api.get(`/mri/compare/${predictionId}`),

  list: (patientId?: string) =>
    api.get('/mri/list', { params: patientId ? { patient_id: patientId } : {} }),
};

// ========================================
// Risk Assessment API
// ========================================

export const riskAPI = {
  assess: (data: Record<string, unknown>) =>
    api.post('/risk/assess', data),
};

// ========================================
// Report API
// ========================================

export const reportAPI = {
  generate: (data: {
    prediction_id: string;
    risk_assessment_id?: string;
    patient_id?: string;
    language?: string;
    additional_clinical_notes?: string;
  }) => api.post('/report/generate', data),

  download: (reportId: string) =>
    api.get(`/report/${reportId}/download`, { responseType: 'blob' }),

  list: (patientId?: string) =>
    api.get('/report/list', { params: patientId ? { patient_id: patientId } : {} }),
};

// ========================================
// Chatbot API
// ========================================

export const chatbotAPI = {
  sendMessage: (data: {
    message: string;
    session_id?: string;
    language?: string;
  }) => api.post('/chatbot/message', data),

  listSessions: () => api.get('/chatbot/sessions'),
};

// ========================================
// Patient API
// ========================================

export const patientAPI = {
  create: (data: {
    full_name: string;
    age: number;
    gender: string;
    medical_record_number?: string;
    contact_info?: string;
    clinical_notes?: string;
  }) => api.post('/patients/', data),

  list: () => api.get('/patients/'),

  get: (patientId: string) => api.get(`/patients/${patientId}`),

  getHistory: (patientId: string) =>
    api.get(`/patients/${patientId}/history`),

  update: (patientId: string, data: Record<string, unknown>) =>
    api.put(`/patients/${patientId}`, data),
};

// ========================================
// Admin API
// ========================================

export const adminAPI = {
  listPendingDoctors: () => api.get('/admin/doctors/pending'),

  verifyDoctor: (data: {
    doctor_id: string;
    action: 'approve' | 'reject';
    admin_notes?: string;
  }) => api.post('/admin/doctors/verify', data),

  getAuditLog: (page?: number, pageSize?: number) =>
    api.get('/admin/audit-log', { params: { page, page_size: pageSize } }),

  getStats: () => api.get('/admin/stats'),
};

// ========================================
// Health API
// ========================================

export const healthAPI = {
  check: () => api.get('/health'),
};

export default api;
