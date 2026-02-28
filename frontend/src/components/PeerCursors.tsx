import { useCollabStore } from '../store/useCollabStore';

export function PeerCursors() {
  const peerCursors = useCollabStore((s) => s.peerCursors);

  const now = Date.now();
  const active = Array.from(peerCursors.values()).filter(
    (c) => now - c.lastUpdated < 5000,
  );

  if (active.length === 0) return null;

  return (
    <>
      {active.map((cursor) => (
        <div
          key={cursor.participantId}
          style={{
            position: 'absolute',
            left: cursor.x,
            top: cursor.y,
            pointerEvents: 'none',
            zIndex: 20,
            transition: 'left 0.1s linear, top 0.1s linear',
          }}
        >
          <svg width="14" height="14" viewBox="0 0 14 14">
            <path
              d="M0 0 L10 7 L4 7 L4 12 Z"
              fill={cursor.color}
              opacity="0.85"
            />
          </svg>
          <span
            style={{
              position: 'absolute',
              left: 12,
              top: 8,
              fontSize: '9px',
              fontFamily: 'var(--font-primary)',
              color: cursor.color,
              whiteSpace: 'nowrap',
              textShadow: '0 1px 3px rgba(0,0,0,0.8)',
            }}
          >
            {cursor.displayName}
          </span>
        </div>
      ))}
    </>
  );
}
