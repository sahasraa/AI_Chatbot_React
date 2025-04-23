# 🤖 Brihaspathi ChatBot – FastAPI + MongoDB

This is a chatbot backend built using **FastAPI** and **MongoDB Atlas**. It supports saving and retrieving chat history, downloading brochures, form submissions, and querying a chatbot — all while running securely from your own server (e.g., AWS EC2)

---

## 🔧 Setup & Deployment

Follow these steps to clone, set up, and run the application.

---

### 1. Clone the Repository

```bash
git clone https://github.com/your-username/chat_bot.git
cd chat_bot/backend
```

---

### 2. Set Up Python 3.10 Environment

Install Python 3.10:

```bash
sudo apt install python3.10 python3.10-venv
```

Create and activate the virtual environment:

```bash
python3.10 -m venv venv
source venv/bin/activate
```

---

### 3. Install Python Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

### 4. Run the FastAPI Application

```bash
python app.py
```

The app will be available at: [http://0.0.0.0:8000](http://0.0.0.0:8000)
