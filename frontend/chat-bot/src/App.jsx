import React, { useState } from "react";
import axios from "axios";
import { TbMessageChatbot } from "react-icons/tb";

const App = () => {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [showForm, setShowForm] = useState(false);

  const sendMessage = async () => {
    if (!input.trim()) return;

    const newMessages = [...messages, { text: input, isUser: true }];
    setMessages(newMessages);

    try {
      const response = await axios.post("http://localhost:8000/chat", { message: input });
      setMessages([...newMessages, { text: response.data.response, isUser: false }]);

      if (response.data.show_form) {
        setShowForm(true);
      } else {
        setShowForm(false);
      }
    } catch (error) {
      console.error("Error:", error);
      setMessages([...newMessages, { text: "Error fetching response", isUser: false }]);
    }

    setInput("");
  };

  const submitForm = async (event) => {
    event.preventDefault();
    const formData = new FormData(event.target);

    try {
      const response = await axios.post("http://localhost:8000/submit_form", formData);
      setMessages([...messages, { text: response.data.message, isUser: false }]);
      setShowForm(false);
    } catch (error) {
      console.error("Error submitting form:", error);
    }
  };

  const[openbot, setOpenbot]=useState(false)

  const handleOpen = ()=>{
    setOpenbot(!openbot)
  }

  return (
    <div className=" fixed right-3 bottom-20">
      {
        openbot && <div className="w-full max-w-md bg-white shadow-lg rounded-lg overflow-hidden">
        <div className="bg-blue-600 text-white text-lg font-semibold text-center py-3 flex justify-between p-3">
          <div>Brihaspathi Chatbot</div><div  className=" cursor-pointer" onClick={handleOpen} >X</div>
        </div>
        <div className="p-4 h-80 overflow-y-auto">
          {messages.map((msg, index) => (
            <div key={index} className={`my-2 px-4 py-2 rounded-lg max-w-xs ${msg.isUser ? "bg-blue-500 text-white ml-auto" : "bg-gray-200 text-gray-800"}`}>
              {msg.text}
            </div>
          ))}
        </div>

        <div className="flex p-2 border-t">
          <input
            type="text"
            className="flex-grow p-2 border rounded-l-lg focus:outline-none"
            placeholder="Ask me anything..."
            value={input}
            onChange={(e) => setInput(e.target.value)}
          />
          <button onClick={sendMessage} className="bg-blue-500 text-white px-4 py-2 rounded-r-lg hover:bg-blue-600">
            Send
          </button>
        </div>

        {showForm && (
          <form onSubmit={submitForm} className="p-4 border-t bg-gray-50">
            <h3 className="text-lg font-semibold mb-2">Request a Quotation</h3>
            <input type="text" name="name" placeholder="Your Name" className="w-full p-2 mb-2 border rounded" required />
            <input type="email" name="email" placeholder="Email Address" className="w-full p-2 mb-2 border rounded" required />
            <input type="tel" name="phone" placeholder="Mobile Number" className="w-full p-2 mb-2 border rounded" required />
            <select name="interest" className="w-full p-2 mb-2 border rounded" required>
              <option value="CCTV/Camera">CCTV/Camera</option>
              <option value="Software">Software</option>
              <option value="Computer/Laptop">Computer/Laptop</option>
              <option value="All">All</option>
            </select>
            <button type="submit" className="w-full bg-green-500 text-white p-2 rounded hover:bg-green-600">
              Submit
            </button>
          </form>
        )}
      </div>
      }

{
  !openbot &&       <div className=" bg-gray-200 p-2 rounded-3xl " onClick={handleOpen}><TbMessageChatbot size={35} className=" cursor-pointer" /></div>

}    </div>
  );
};

export default App;
