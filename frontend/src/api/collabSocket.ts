import type { CollabC2S, CollabS2C } from '../types/collab';

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || `ws://${window.location.host}`;

type MessageHandler = (msg: CollabS2C) => void;

export class CollabSocket {
  private ws: WebSocket | null = null;
  private onMessage: MessageHandler;
  private onDisconnect: (() => void) | null;
  private pingInterval: ReturnType<typeof setInterval> | null = null;

  constructor(onMessage: MessageHandler, onDisconnect?: () => void) {
    this.onMessage = onMessage;
    this.onDisconnect = onDisconnect ?? null;
  }

  connect(sessionId: string): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${WS_BASE}/ws/collab/${sessionId}`);

      this.ws.onopen = () => {
        this.pingInterval = setInterval(() => {
          this.send({ type: 'c2s_ping' });
        }, 30_000);
        resolve();
      };

      this.ws.onmessage = (event) => {
        try {
          const msg: CollabS2C = JSON.parse(event.data);
          this.onMessage(msg);
        } catch {
          // ignore malformed
        }
      };

      this.ws.onerror = () => reject(new Error('Collab WebSocket connection failed'));

      this.ws.onclose = () => {
        if (this.pingInterval) {
          clearInterval(this.pingInterval);
          this.pingInterval = null;
        }
        this.onDisconnect?.();
      };
    });
  }

  send(msg: CollabC2S): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    if (this.pingInterval) {
      clearInterval(this.pingInterval);
      this.pingInterval = null;
    }
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
