import { useState } from "react";
import "./css/App.css";
import Form from "./form";
import Chat from "./chat";

type Message = {
  sender: "user" | "system";
  text: string;
};

export default function App() {
  const [messages, setMessages] = useState<Message[]>([]);

  return (
    <div className="app-container">
      <div className="form-container">
        <Form setMessages={setMessages} />
      </div>  

       <div className="chat-container">
        <Chat messages={messages} setMessages={setMessages} />
      </div>    
    </div>
  );
}