// const websocketURL = "ws://127.0.0.1:8000/ws/chat/";
const websocketURL = "wss://qwertyweb.xyz:8443/ws/chat/"

class WebSocketHandler {
  private socket: WebSocket;
  private url: string;
  public addMessage: (message: string) => void = () => 0;

  constructor(url: string) {
    this.url = url;
    this.socket = new WebSocket(url);
    this.initialize();
  }

  private initialize() {
    this.socket.onopen = this.onOpen.bind(this);
    this.socket.onmessage = this.onMessage.bind(this);
    this.socket.onerror = this.onError.bind(this);
    this.socket.onclose = this.onClose.bind(this);
  }

  private onOpen(event: Event) {
    console.log("WebSocket connection opened:", event);
  }

  private onMessage(event: MessageEvent) {
    console.log("Message received: ", event.data);
    const data = JSON.parse(event.data);
    this.addMessage(data.message);
  }

  private onError(event: Event) {
    console.error("WebSocket error:", event);
  }

  private onClose(event: CloseEvent) {
    console.log("WebSocket connection closed:", event);
  }

  public sendMessage(message: string, match_id: number, language: string,team_a_player:[], team_b_player:[]) {
    if (this.socket.readyState === WebSocket.OPEN) {
      const msg = { message: message, match_id: match_id, language: language,team_a_player:team_a_player,team_b_player:team_b_player};
      console.log('match_id being sent',match_id)
      const sendMsg = JSON.stringify(msg);
      this.socket.send(sendMsg);
    } else {
      console.error(
        "WebSocket is not open. Ready state:",
        this.socket.readyState
      );
    }
  }

  public closeConnection() {
    this.socket.close();
  }
}

export default WebSocketHandler;

const webSocketConnection = new WebSocketHandler(websocketURL);

export { webSocketConnection };