"""OmniVoice synthesis time for one 188-character Indonesian sentence in every current voice.

usage: python3 tts_bench.py
The first request includes the model load. Server on :5004.
"""
import json, time, urllib.request, wave, io
T = "Pengujian ini membaca satu kalimat yang sama dengan empat suara berbeda, lalu mencatat waktu sintesis dan panjang audio yang dihasilkan, supaya hasilnya dapat dibandingkan dengan catatan lama."
T = T[:188]
print("chars", len(T))
voices = [v["id"] for v in json.load(urllib.request.urlopen("http://127.0.0.1:5004/voices")) if v["current"]]
def speak(v):
    req = urllib.request.Request("http://127.0.0.1:5004/speak", json.dumps({"text": T, "voice": v, "language": "Indonesian"}).encode(), {"Content-Type": "application/json"})
    t = time.time(); data = urllib.request.urlopen(req, timeout=600).read(); dt = time.time() - t
    w = wave.open(io.BytesIO(data)); return dt, w.getnframes() / w.getframerate()
dt, a = speak(voices[0]); print("first (load)", round(dt, 2), round(a, 1))
for v in voices:
    dt, a = speak(v); print(v, round(dt, 2), "s for", round(a, 1), "s audio")
