import { useState, useRef, useEffect } from 'react';
import { chatbotAPI } from '../services/api';
import { motion } from 'framer-motion';

type Message = {
  role: 'user' | 'assistant';
  content: string;
  isOnTopic?: boolean;
  isServiceError?: boolean;
  timestamp: string;
};

export default function Chatbot() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [sessionId, setSessionId] = useState<string>('');
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg = input.trim();
    setInput('');

    setMessages(prev => [...prev, { role: 'user', content: userMsg, timestamp: new Date().toISOString() }]);
    setLoading(true);

    try {
      const res = await chatbotAPI.sendMessage({ message: userMsg, session_id: sessionId || undefined });
      setSessionId(res.data.session_id);
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.data.reply,
        isOnTopic: res.data.is_on_topic,
        timestamp: res.data.timestamp,
      }]);
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string; trace_id?: string } } };
      const detail = axiosErr.response?.data?.detail || 'The medical assistant is temporarily unavailable. Please try again.';
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: detail,
        isServiceError: true,
        timestamp: new Date().toISOString(),
      }]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="h-[calc(100vh-12rem)] flex flex-col">
      <div className="mb-4">
        <h1 className="section-title">AI Medical Assistant</h1>
        <p className="section-subtitle">Specialized in Alzheimer's disease, dementia, MRI interpretation, and platform features</p>
      </div>

      <div className="disclaimer-banner mb-4">
        <span>🩺</span>
        <span>This assistant only answers questions about Alzheimer's, dementia, neuroimaging, and this platform. Off-topic queries are declined.</span>
      </div>

      {/* Messages */}
      <div className="flex-1 overflow-y-auto glass-card p-4 space-y-4">
        {messages.length === 0 && (
          <div className="text-center py-16">
            <span className="text-5xl block mb-4">💬</span>
            <p className="text-white font-semibold">Ask me about Alzheimer's disease</p>
            <p className="text-slate-400 text-sm mt-2">Examples:</p>
            <div className="flex flex-wrap justify-center gap-2 mt-3">
              {[
                'What are the stages of Alzheimer\'s?',
                'How to interpret Grad-CAM heatmaps?',
                'What does a low MMSE score indicate?',
                'Explain the risk factors for dementia',
              ].map((q) => (
                <button
                  key={q}
                  onClick={() => { setInput(q); }}
                  className="text-xs bg-white/5 border border-white/10 px-3 py-1.5 rounded-lg text-slate-300 hover:bg-white/10 transition"
                >
                  {q}
                </button>
              ))}
            </div>
          </div>
        )}

        {messages.map((msg, i) => (
          <motion.div
            key={i}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            <div className={`max-w-[75%] rounded-2xl px-4 py-3 ${
              msg.role === 'user'
                ? 'bg-brand-600/30 border border-brand-500/30 text-white'
                : msg.isServiceError === true
                  ? 'bg-red-500/10 border border-red-500/20 text-red-200 shadow-lg'
                  : msg.isOnTopic === false
                    ? 'bg-amber-500/10 border border-amber-500/20 text-amber-200'
                    : 'bg-white/5 border border-white/10 text-slate-200'
            }`}>
              {msg.role === 'assistant' && msg.isServiceError === true && (
                <span className="text-xs text-red-400 font-bold block mb-1">⚠️ Assistant Temporarily Unavailable</span>
              )}
              {msg.role === 'assistant' && msg.isOnTopic === false && !msg.isServiceError && (
                <span className="text-xs text-amber-400 font-medium block mb-1">Off-topic — redirected</span>
              )}
              <p className="text-sm whitespace-pre-wrap">{msg.content}</p>
            </div>
          </motion.div>
        ))}

        {loading && (
          <div className="flex justify-start">
            <div className="bg-white/5 border border-white/10 rounded-2xl px-4 py-3">
              <div className="flex gap-1.5">
                <div className="w-2 h-2 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '0ms' }} />
                <div className="w-2 h-2 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <div className="w-2 h-2 bg-brand-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}

        <div ref={messagesEnd} />
      </div>

      {/* Input */}
      <div className="mt-4 flex gap-3">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && sendMessage()}
          className="input-field flex-1"
          placeholder="Ask about Alzheimer's, dementia, MRI interpretation..."
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading || !input.trim()} className="btn-primary px-8">
          Send
        </button>
      </div>
    </div>
  );
}
