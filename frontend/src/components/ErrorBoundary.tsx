import { Component, type ErrorInfo, type ReactNode } from 'react';

interface Props {
  children: ReactNode;
}

interface State {
  hasError: boolean;
  error: Error | null;
}

export class ErrorBoundary extends Component<Props, State> {
  state: State = { hasError: false, error: null };

  static getDerivedStateFromError(error: Error): State {
    return { hasError: true, error };
  }

  componentDidCatch(error: Error, errorInfo: ErrorInfo) {
    console.error('Neural MRI uncaught error:', error, errorInfo);
  }

  render() {
    if (this.state.hasError) {
      return (
        <div
          style={{
            padding: 40,
            fontFamily: 'var(--font-primary, monospace)',
            background: '#0a0c10',
            color: '#c0c8d8',
            height: '100vh',
            display: 'flex',
            flexDirection: 'column',
            alignItems: 'center',
            justifyContent: 'center',
            gap: 16,
          }}
        >
          <div
            style={{
              fontSize: 14,
              color: '#ff4466',
              letterSpacing: '2px',
              textTransform: 'uppercase',
            }}
          >
            NEURAL MRI — ERROR
          </div>
          <pre
            style={{
              fontSize: 12,
              color: '#ff6688',
              background: 'rgba(255,68,102,0.08)',
              border: '1px solid rgba(255,68,102,0.2)',
              padding: '12px 20px',
              borderRadius: 4,
              maxWidth: 600,
              overflow: 'auto',
            }}
          >
            {this.state.error?.message}
          </pre>
          <button
            onClick={() => this.setState({ hasError: false, error: null })}
            style={{
              background: 'rgba(0,255,170,0.12)',
              border: '1px solid rgba(0,255,170,0.3)',
              color: '#00ffaa',
              padding: '6px 20px',
              fontSize: 12,
              fontFamily: 'var(--font-primary, monospace)',
              cursor: 'pointer',
              borderRadius: 4,
              letterSpacing: '1px',
            }}
          >
            RETRY
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}
