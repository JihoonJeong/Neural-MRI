import { useState } from 'react';
import { useCollabStore } from '../../store/useCollabStore';

const sectionStyle = {
  padding: '8px 10px',
  fontSize: 'var(--font-size-xs)' as const,
  color: 'var(--text-primary)' as const,
  fontFamily: 'var(--font-primary)' as const,
};

const btnStyle = {
  background: 'rgba(0,255,170,0.10)',
  border: '1px solid rgba(0,255,170,0.25)',
  color: 'var(--accent-active)',
  padding: '4px 10px',
  fontSize: 'var(--font-size-xs)',
  fontFamily: 'var(--font-primary)',
  cursor: 'pointer',
  borderRadius: 3,
} as const;

const inputStyle = {
  background: 'rgba(255,255,255,0.03)',
  border: '1px solid var(--border)',
  color: 'var(--text-data)',
  padding: '4px 8px',
  fontSize: 'var(--font-size-xs)',
  fontFamily: 'var(--font-primary)',
  outline: 'none',
  borderRadius: 3,
  width: '100%',
} as const;

export function CollabPanel() {
  const {
    isInSession,
    sessionId,
    role,
    participants,
    isConnecting,
    error,
    createSession,
    joinSession,
    leaveSession,
  } = useCollabStore();

  const [joinId, setJoinId] = useState('');
  const [displayName, setDisplayName] = useState('');
  const [collapsed, setCollapsed] = useState(false);
  const [copied, setCopied] = useState(false);

  const handleCreate = async () => {
    try {
      await createSession(displayName || 'Host');
    } catch {
      // error is set in store
    }
  };

  const handleJoin = async () => {
    if (!joinId.trim()) return;
    try {
      await joinSession(joinId.trim(), displayName || 'Viewer');
    } catch {
      // error is set in store
    }
  };

  const handleCopy = () => {
    if (sessionId) {
      const url = `${window.location.origin}${window.location.pathname}?session=${sessionId}`;
      navigator.clipboard.writeText(url);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  return (
    <div style={sectionStyle}>
      <div
        style={{ cursor: 'pointer', marginBottom: 4, letterSpacing: '0.5px' }}
        onClick={() => setCollapsed(!collapsed)}
      >
        <span style={{ color: 'var(--text-secondary)', marginRight: 4 }}>
          {collapsed ? '>' : 'v'}
        </span>
        COLLAB
        {isInSession && (
          <span
            style={{
              marginLeft: 6,
              color: '#4ecdc4',
              fontSize: 'var(--font-size-xs)',
            }}
          >
            [{participants.length}]
          </span>
        )}
      </div>

      {!collapsed && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
          {error && (
            <div style={{ color: '#ff4466', fontSize: 'var(--font-size-xs)' }}>
              {error}
            </div>
          )}

          {!isInSession ? (
            <>
              {/* Name input */}
              <input
                value={displayName}
                onChange={(e) => setDisplayName(e.target.value)}
                placeholder="Display name"
                style={inputStyle}
              />

              {/* Create session */}
              <button
                onClick={handleCreate}
                disabled={isConnecting}
                style={{
                  ...btnStyle,
                  opacity: isConnecting ? 0.5 : 1,
                }}
              >
                {isConnecting ? 'CONNECTING...' : 'CREATE SESSION'}
              </button>

              {/* Join session */}
              <div style={{ display: 'flex', gap: 4 }}>
                <input
                  value={joinId}
                  onChange={(e) => setJoinId(e.target.value)}
                  placeholder="Session ID"
                  style={{ ...inputStyle, flex: 1 }}
                  onKeyDown={(e) => e.key === 'Enter' && handleJoin()}
                />
                <button
                  onClick={handleJoin}
                  disabled={isConnecting || !joinId.trim()}
                  style={{
                    ...btnStyle,
                    opacity: isConnecting || !joinId.trim() ? 0.5 : 1,
                  }}
                >
                  JOIN
                </button>
              </div>
            </>
          ) : (
            <>
              {/* Session info */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 4 }}>
                <span style={{ color: 'var(--text-secondary)' }}>ID:</span>
                <code style={{ color: 'var(--text-data)', flex: 1 }}>
                  {sessionId}
                </code>
                <button
                  onClick={handleCopy}
                  style={{
                    ...btnStyle,
                    padding: '2px 6px',
                    background: copied ? 'rgba(0,255,170,0.25)' : btnStyle.background,
                  }}
                >
                  {copied ? 'OK' : 'COPY'}
                </button>
              </div>

              {/* Role indicator */}
              <div
                style={{
                  color: role === 'host' ? '#4ecdc4' : '#f7dc6f',
                  textTransform: 'uppercase',
                  letterSpacing: '0.5px',
                }}
              >
                {role === 'host' ? 'HOSTING' : 'VIEWING'}
              </div>

              {/* Participants */}
              <div>
                {participants.map((p) => (
                  <div
                    key={p.id}
                    style={{
                      display: 'flex',
                      alignItems: 'center',
                      gap: 4,
                      padding: '1px 0',
                    }}
                  >
                    <span
                      style={{
                        display: 'inline-block',
                        width: 6,
                        height: 6,
                        borderRadius: '50%',
                        background: p.color,
                      }}
                    />
                    <span style={{ color: 'var(--text-data)' }}>
                      {p.display_name}
                    </span>
                    {p.role === 'host' && (
                      <span style={{ color: 'var(--text-secondary)', fontSize: '9px' }}>
                        HOST
                      </span>
                    )}
                  </div>
                ))}
              </div>

              {/* Leave/End button */}
              <button
                onClick={leaveSession}
                style={{
                  ...btnStyle,
                  background: 'rgba(255,68,102,0.10)',
                  border: '1px solid rgba(255,68,102,0.3)',
                  color: '#ff4466',
                }}
              >
                {role === 'host' ? 'END SESSION' : 'LEAVE'}
              </button>
            </>
          )}
        </div>
      )}
    </div>
  );
}
