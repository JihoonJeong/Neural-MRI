import type { WsMessageIn, WsMessageOut } from '../types/scan';

type MessageHandler = (msg: WsMessageOut) => void;

const WS_BASE = import.meta.env.VITE_WS_BASE_URL || `ws://${window.location.host}`;

export class ScanSocket {
  private ws: WebSocket | null = null;
  private onMessage: MessageHandler;
  private onDisconnect: (() => void) | null;

  constructor(onMessage: MessageHandler, onDisconnect?: () => void) {
    this.onMessage = onMessage;
    this.onDisconnect = onDisconnect ?? null;
  }

  connect(): Promise<void> {
    return new Promise((resolve, reject) => {
      this.ws = new WebSocket(`${WS_BASE}/ws/stream`);

      this.ws.onopen = () => resolve();

      this.ws.onmessage = (event) => {
        try {
          const msg: WsMessageOut = JSON.parse(event.data);
          this.onMessage(msg);
        } catch {
          // ignore malformed messages
        }
      };

      this.ws.onerror = () => reject(new Error('WebSocket connection failed'));

      this.ws.onclose = () => {
        this.onDisconnect?.();
      };
    });
  }

  send(msg: WsMessageIn): void {
    if (this.ws?.readyState === WebSocket.OPEN) {
      this.ws.send(JSON.stringify(msg));
    }
  }

  close(): void {
    this.ws?.close();
    this.ws = null;
  }

  get connected(): boolean {
    return this.ws?.readyState === WebSocket.OPEN;
  }
}
