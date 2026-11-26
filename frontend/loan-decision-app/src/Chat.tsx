import { useState } from "react";
import "./css/chat.css";

type Message = {
  sender: "user" | "system";
  text: string;
};

type Props = {
  messages: any[];
  setMessages: any;
};

export default function Chat({ messages, setMessages }: Props) {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);

  const sendMessage = async () => {
    if (!input.trim() || loading) return; 

    const userMessage: Message = { sender: "user", text: input };
    setMessages((prev: Message[]) => [...prev, userMessage]);
    
    const currentInput = input;
    setInput("");
    setLoading(true);

    try {
      console.log(`🚀 [Frontend] Sending exclusive chat request: "${currentInput}"`);

      const res = await fetch("http://127.0.0.1:8000/chat", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message: currentInput,
        }),
      });

      const data = await res.json();

      const systemMessage: Message = { sender: "system", text: data.response };
      setMessages((prev: Message[]) => [...prev, systemMessage]);

    } catch (error) {
      console.error("Chat Error Details:", error);
      const errorMessage: Message = { sender: "system", text: "Server error or UI binding issue." };
      setMessages((prev: Message[]) => [...prev, errorMessage]);
    } finally {
      setLoading(false); 
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLInputElement>) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  return (
    <div className="chat-card">
      <div className="chat-log">
        {messages.map((msg, index) => (
          <div
            key={index}
            className={`chat-message ${msg.sender === "user" ? "user" : "system"}`}
            style={{ whiteSpace: "pre-line" }}
          >
            {msg.text}
          </div>
        ))}

        {loading && (
          <div className="chat-message system loading-message">
            <div className="banking-spinner"></div>
            <span style={{ fontSize: "12px", color: "#888", marginLeft: "8px" }}>
              Analyzing risk constraints...
            </span>
          </div>
        )}
      </div>

      <div className="chat-input-area">
        <input
          type="text"
          placeholder={loading ? "System is thinking..." : "Type your follow-up risk inquiry here..."}
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyPress}
          disabled={loading}
        />
        <button onClick={sendMessage} disabled={loading}>
          {loading ? "..." : "Send"}
        </button>
      </div>
    </div>
  );
}