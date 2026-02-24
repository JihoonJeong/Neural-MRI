import { useScanStore } from '../../store/useScanStore';

export function LogPanel() {
  const logs = useScanStore((s) => s.logs);

  return (
    <div
      className="px-3 py-2 overflow-y-auto"
      style={{
        fontSize: 'var(--font-size-xs)',
        color: 'var(--text-secondary)',
        maxHeight: 80,
        borderTop: '1px solid var(--border)',
      }}
    >
      {logs.slice(-8).map((log, i) => (
        <div key={i} className="mb-0.5">
          <span style={{ color: 'var(--text-primary)' }}>[{log.time}]</span> {log.msg}
        </div>
      ))}
    </div>
  );
}
