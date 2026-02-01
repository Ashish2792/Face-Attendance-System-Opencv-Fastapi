# Face Attendance System (IN / OUT)

A real-time face recognition based attendance system using OpenCV and face_recognition with:

- Punch-In (1) / Punch-Out (0) logic
- Liveness detection (blink + head movement)
- Lighting-aware preprocessing
- Live IN / OUT status overlay
- FastAPI backend
- Excel export support

---

## 🚀 Features
- Real-time face detection & recognition
- Spoof resistance using liveness checks
- State-based attendance (IN ↔ OUT)
- No auto punch-out (event driven)
- Live REST API
- One-click Excel export

---

## 🛠️ Tech Stack
- Python
- OpenCV
- face_recognition (dlib)
- FastAPI
- Pandas
- NumPy

---

## 📂 Project Structure
.
├── face_attendance.py # Core recognition & attendance logic
├── api.py # FastAPI backend
├── requirements.txt
├── README.md
└── .gitignore


---

## ▶️ How to Run

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt
---
### 2️⃣ Register a user

python face_attendance.py --mode register --name Ashish --id 1234

### 3️⃣ Start recognition
python face_attendance.py --mode recognize

#### 4️⃣ Start live API
uvicorn api:app --reload

## 🌐 API Endpoints

/attendance → Full attendance log

/live-status → Users currently IN

/export/excel → Download Excel report

Swagger UI:

http://127.0.0.1:8000/docs

📊 Attendance Logic

1 → Punch-In

0 → Punch-Out

Punch-Out occurs only when the same person is detected again

👨‍💻 Author

Ashish Ubale



---