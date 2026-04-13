from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import asyncio
import json

from inference import load_model, InferenceSession

# ── Startup: load model once into memory ──────────────────
ml = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    print("Loading model...")
    ml["model"], ml["classes"] = load_model(
        "models/asl_bilstm_final.keras",
        "models/label_classes_lstm.npy"
    )
    print(f"Model ready. {len(ml['classes'])} signs loaded.")
    yield
    ml.clear()

app = FastAPI(lifespan=lifespan)

# ── CORS: allow your Angular dev server ───────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],  # add your deployed URL later
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Health check ──────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok", "signs_count": len(ml.get("classes", []))}

# ── WebSocket endpoint ────────────────────────────────────
@app.websocket("/ws/translate")
async def translate(websocket: WebSocket):
    await websocket.accept()
    session = InferenceSession()
    print("Client connected")

    try:
        while True:
            # Receive raw frame bytes from Angular
            frame_bytes = await websocket.receive_bytes()

            # Run inference in a thread (so it doesn't block the event loop)
            result = await asyncio.get_event_loop().run_in_executor(
                None,
                session.process_frame,
                frame_bytes,
                ml["model"],
                ml["classes"]
            )

            # Send prediction back as JSON
            await websocket.send_text(json.dumps(result))

    except WebSocketDisconnect:
        print("Client disconnected")
    finally:
        session.close()