export function ScanLineOverlay() {
  return (
    <div
      className="absolute inset-0 pointer-events-none overflow-hidden"
      style={{ zIndex: 2 }}
    >
      {/* Static scanlines */}
      <div style={{ opacity: 0.04 }}>
        {Array.from({ length: 150 }, (_, i) => (
          <div
            key={i}
            style={{ height: '1px', background: '#fff', marginBottom: '4px' }}
          />
        ))}
      </div>

      {/* Slow scan bar sweeping top to bottom */}
      <div
        style={{
          position: 'absolute',
          left: 0,
          right: 0,
          height: '2px',
          background: 'linear-gradient(90deg, transparent 0%, rgba(0,255,170,0.12) 30%, rgba(0,255,170,0.2) 50%, rgba(0,255,170,0.12) 70%, transparent 100%)',
          boxShadow: '0 0 12px rgba(0,255,170,0.08)',
          animation: 'scan-sweep 8s linear infinite',
        }}
      />

      {/* Subtle brightness flicker */}
      <div
        className="absolute inset-0"
        style={{
          background: 'rgba(255,255,255,0.008)',
          animation: 'crt-flicker 4s ease-in-out infinite',
        }}
      />
    </div>
  );
}
