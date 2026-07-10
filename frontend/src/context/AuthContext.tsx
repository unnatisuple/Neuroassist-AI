import React, { createContext, useContext, useState, useEffect, useCallback } from 'react';
import { authAPI } from '../services/api';

interface Doctor {
  id: string;
  full_name: string;
  email: string;
  specialization: string;
  medical_license_number: string;
  role: string;
  verification_status: string;
}

interface AuthContextType {
  doctor: Doctor | null;
  token: string | null;
  isAuthenticated: boolean;
  isVerified: boolean;
  isAdmin: boolean;
  isLoading: boolean;
  login: (fullName: string, email: string) => Promise<void>;
  logout: () => void;
}

const AuthContext = createContext<AuthContextType | null>(null);

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [doctor, setDoctor] = useState<Doctor | null>(null);
  const [token, setToken] = useState<string | null>(
    localStorage.getItem('neuroassist_token')
  );
  const [isLoading, setIsLoading] = useState(true);

  const loadProfile = useCallback(async () => {
    if (!token) {
      setIsLoading(false);
      return;
    }

    try {
      const res = await authAPI.getProfile();
      setDoctor(res.data);
    } catch {
      // Token invalid
      localStorage.removeItem('neuroassist_token');
      localStorage.removeItem('neuroassist_refresh');
      setToken(null);
      setDoctor(null);
    } finally {
      setIsLoading(false);
    }
  }, [token]);

  useEffect(() => {
    loadProfile();
  }, [loadProfile]);

  const login = async (fullName: string, email: string) => {
    const res = await authAPI.login({ full_name: fullName, email });
    const data = res.data;

    localStorage.setItem('neuroassist_token', data.access_token);
    localStorage.setItem('neuroassist_refresh', data.refresh_token);
    setToken(data.access_token);
    setDoctor({
      id: data.doctor_id,
      full_name: data.full_name,
      email,
      specialization: 'Neurology',
      medical_license_number: 'LICENSE-AUTO',
      role: data.role,
      verification_status: data.verification_status,
    });
  };

  const logout = () => {
    localStorage.removeItem('neuroassist_token');
    localStorage.removeItem('neuroassist_refresh');
    setToken(null);
    setDoctor(null);
  };

  return (
    <AuthContext.Provider
      value={{
        doctor,
        token,
        isAuthenticated: !!token && !!doctor,
        isVerified: doctor?.verification_status === 'verified',
        isAdmin: doctor?.role === 'admin',
        isLoading,
        login,
        logout,
      }}
    >
      {children}
    </AuthContext.Provider>
  );
}

export function useAuth() {
  const ctx = useContext(AuthContext);
  if (!ctx) throw new Error('useAuth must be used within AuthProvider');
  return ctx;
}
