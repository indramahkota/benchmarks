# Benchmark AI lokal pada GPU laptop 6 GiB

Hasil pengujian LLM, ASR, TTS, embedding, reranking, konversi PDF, dan image-to-3D pada satu laptop.

## Perangkat dan versi

| Komponen | Konfigurasi |
| --- | --- |
| Laptop | Acer Nitro AN515-57 |
| CPU | Intel Core i7-11800H, 8 core dan 16 thread |
| RAM | Sekitar 31 GiB, tanpa swap |
| GPU | RTX 3060 Laptop, VRAM 6.144 MiB |
| Sistem | Ubuntu 24.04.5 LTS, driver NVIDIA 580.196.07 |
| Runtime LLM | Ollama 0.32.13, llama.cpp `8df332d`, vLLM 0.30.0, SGLang 0.5.20 |
| Reranker | llama.cpp `5431581` (build CUDA), `bge-reranker-v2-m3` Q8_0 |

## Metode LLM

| Pengaturan | Nilai |
| --- | --- |
| Server | Satu server inference aktif pada satu waktu |
| Prompt | Sekitar 2.790 token, prefix unik agar prefix cache tidak terpakai |
| Keluaran | 256 token, temperature 0, thinking dimatikan |
| Pengulangan | Tiga run per konfigurasi, median; satu warmup tidak dihitung |
| Context | 32.768 token; KV cache q8_0 untuk llama.cpp dan Ollama, bf16 untuk vLLM dan SGLang |
| Suhu awal | GPU paling tinggi 55 °C sebelum setiap request; SGLang 68 °C karena GPU-nya tetap hangat saat idle |

| Metrik | Cara menghitung |
| --- | --- |
| TTFT | Dari request dikirim sampai chunk token pertama diterima |
| Prefill | Jumlah prompt token dibagi TTFT, termasuk overhead sebelum token pertama |
| Decode | Token keluaran setelah token pertama dibagi waktunya |
| Gabungan | Seluruh token keluaran dibagi wall-clock kelompok request, termasuk TTFT |

## Qwen3.5-4B pada empat runtime

Model ini berjalan sepenuhnya di GPU. Kecepatan dalam token per detik.

| Runtime | Request | TTFT | Prefill | Decode per request | Gabungan |
| --- | --- | --- | --- | --- | --- |
| vLLM | 1 | 1,07 s | 2.614 | 78,0 | 58,9 |
| vLLM | 4 | 1,24 s | 2.241 | 66,6 | 201,1 |
| SGLang | 1 | 1,06 s | 2.640 | 77,8 | 58,9 |
| SGLang | 4 | 3,27 s | 1.281 | 73,1 | 116,8 |
| llama.cpp | 1 | 1,40 s | 1.992 | 69,8 | 50,7 |
| llama.cpp | 4 | 5,85 s | 477 | 34,9 | 77,4 |
| Ollama | 1 | 1,70 s | 1.644 | 73,6 | 49,5 |
| Ollama | 4 | 7,13 s | 418 | 73,1 | 64,1 |

| Runtime | Bobot | Konfigurasi |
| --- | --- | --- |
| vLLM | AWQ 4-bit | `--kv-cache-memory-bytes 1500000000 --max-num-seqs 16` |
| SGLang | AWQ 4-bit | `--mem-fraction-static 0.85 --max-running-requests 4 --chunked-prefill-size 1024 --disable-prefill-cuda-graph`; pool KV sekitar 11.549 token |
| llama.cpp | Q4_K_M | `-ngl 99 -ub 2048 --parallel 4 --kv-unified` |
| Ollama | Q4_K_M | `OLLAMA_NUM_PARALLEL=1`, sehingga empat request diproses berurutan |

## Model MoE besar dengan expert di CPU

Satu request, kecepatan dalam token per detik.

| Model dan runtime | TTFT | Prefill | Decode |
| --- | --- | --- | --- |
| Qwen3.6-35B-A3B, llama.cpp `--n-cpu-moe 36 -ub 4096 -t 7` | 4,37 s | 638 | 36,8 |
| Qwen3.6-35B-A3B, Ollama | 14,99 s | 187 | 26,2 |
| Qwen3-Coder-30B-A3B, llama.cpp `--n-cpu-moe 44 -ub 4096 -t 7` | 3,71 s | 751 | 24,6 |
| Qwen3-Coder-30B-A3B, Ollama | 14,88 s | 188 | 20,0 |

Qwen3.6-35B-A3B Q4_K_M berukuran sekitar 20,7 GiB. llama.cpp memakai GGUF Bartowski karena GGUF
milik Ollama tidak dapat dibaca llama.cpp yang diuji. Qwen3-Coder-30B-A3B memakai blob GGUF yang sama
pada kedua runtime. Model besar melalui vLLM dan SGLang tidak diuji, karena bobotnya tidak muat di kartu.

## Simulasi sesi agent

Delapan turn dengan system prompt sekitar 6.000 token dan tiga definisi tool. Setiap turn menambah
sekitar 300 kata dan menghasilkan 128 token; context akhir sekitar 10.320 token. Tool call dinilai
valid bila nama tool ada dan `arguments` berupa JSON yang dapat di-parse; kualitas kode tidak dinilai.

| Model dan runtime | TTFT turn 1 | Median TTFT turn 2–8 | Jumlah TTFT | Decode turn terakhir | Tool call valid |
| --- | --- | --- | --- | --- | --- |
| Qwen3.5-4B, SGLang | 2,50 s | 0,28 s | 4,4 s | 73,9 | 5/5 |
| Qwen3.5-4B, vLLM | 2,45 s | 0,41 s | 5,2 s | 73,5 | 5/5 |
| Qwen3.5-4B, llama.cpp | 3,31 s | 0,39 s | 6,1 s | 64,2 | 5/5 |
| Qwen3.5-4B, Ollama | 3,46 s | 1,00 s | 10,3 s | 68,1 | 5/5 |
| Qwen3-Coder-30B-A3B, llama.cpp | 7,83 s | 2,86 s | 27,9 s | 21,4 | 5/5 |
| Qwen3.6-35B-A3B, llama.cpp | 9,22 s | 3,30 s | 32,3 s | 35,5 | 5/5 |
| Qwen3-Coder-30B-A3B, Ollama | 32,08 s | 4,70 s | 65,0 s | 18,2 | 5/5 |
| Qwen3.6-35B-A3B, Ollama | 33,09 s | 7,63 s | 84,3 s | 25,6 | 5/5 |

Jumlah TTFT tidak mencakup waktu decode.

## Layanan aplikasi

Setiap layanan dijalankan sebagai server HTTP lokal dengan argumen unit systemd-nya, satu per satu.
Layanan yang memakai kunci GPU melepas modelnya bila layanan lain meminta ruang.

### ASR: Qwen3-ASR-1.7B

100 klip pertama FLEURS `id_id` dev, bahasa Indonesian, reranker menetap di kartu. WER dan CER
dihitung dengan `jiwer` setelah teks dijadikan huruf kecil dan tanda baca dihapus.

| Metrik | Hasil |
| --- | --- |
| WER | 6,52 % |
| CER | 2,81 % |
| Model dimuat | 12,8 s |
| Request pertama, termasuk muat | 14,26 s |
| Klip berikutnya | Median 0,86 s, rata-rata 1,01 s |
| Puncak pemakaian kartu | 5.463 MiB |
| Rekaman panjang, 26 klip digabung dengan jeda 0,5 s (324,3 s) | 21,5 s, WER 9,81 %, puncak 5.650 MiB |

Klip rekaman panjang tidak saling berhubungan, jadi WER-nya tidak sebanding dengan klip pendek.

### TTS: OmniVoice

Satu kalimat Indonesia 188 karakter pada empat frozen voice preset. Model dimuat dalam 7,3 s; request
pertama, termasuk muat, 14,0 s.

| Voice | Waktu sintesis | Panjang audio |
| --- | --- | --- |
| `id_ID-female-warm@554ed5a691c3` | 4,61 s | 12,4 s |
| `id_ID-female-young@fa6a0e74fb40` | 4,47 s | 11,2 s |
| `id_ID-male-news@b51966ff90c8` | 4,65 s | 12,3 s |
| `id_ID-male-young@b2cea97a12a4` | 4,59 s | 11,8 s |

Request TTS pertama membuat Qwen3-ASR melepas modelnya atas permintaan peer, lalu OmniVoice dimuat.

### Embedding: bge-m3 di CPU

Tag Ollama `bge-m3` dengan `num_gpu 0` (`size_vram` 0), 16 thread.

| Masukan | Waktu |
| --- | --- |
| Query 40 karakter | 0,178–0,184 s |
| Chunk 1.200 karakter | 0,28–0,54 s |

### Reranker: bge-reranker-v2-m3

Kandidat 1.200 karakter dari tiga set teks: Inggris (median 236 token per kandidat), Indonesia
(292 token), dan campuran (255 token). Median tiga run setelah warmup.

| Kandidat | GPU, Inggris | GPU, Indonesia | GPU, campuran | CPU 16 thread tanpa CUDA, Inggris |
| --- | --- | --- | --- | --- |
| 1 | 19 ms | 21 ms | 16 ms | 340 ms |
| 5 | 72 ms | 100 ms | 90 ms | 1.353 ms |
| 10 | 140 ms | 204 ms | 172 ms | |
| 20 | 313 ms | 323 ms | 344 ms | 5.978 ms |
| 40 | 694 ms | | 635 ms | |

Pada CPU, delapan thread memerlukan 301 ms untuk satu kandidat, hampir sama dengan 16 thread. Set
Indonesia tidak memiliki 40 kandidat penuh.

### Konversi PDF: Docling

`nist-sp800-73pt2.pdf` dengan layout `docling_layout_egret_xlarge`, OCR mati, `malloc_trim`, dan
`MALLOC_ARENA_MAX=2`: 19,4 s pada konversi pertama dan 17,9 s pada konversi kedua, keduanya
menghasilkan 274.794 karakter Markdown.

### Image-to-3D: TRELLIS.2

trellis2cpp dengan patch low-VRAM: SS flow dilepas setelah digunakan, lalu shape flow dimuat saat
dibutuhkan. Masukan berupa potret yang dipotong server menjadi 1.755 px di bagian kepala dan bahu.

| Metrik | Hasil |
| --- | --- |
| Model dimuat | 8,8 s |
| End-to-end | 299 s |
| Generate | 152,5 s |
| Texture bake | 135,8 s |
| Segitiga | 99.999 |
| IoU siluet depan | 0,951 |
| Puncak pemakaian kartu | 4.673 MiB |

IoU siluet hanya menilai kecocokan dari depan.

## Temuan yang perlu diketahui

- **`-ngl 0` bukan CPU murni.** Binary llama.cpp yang dibangun dengan CUDA tetap menitipkan
  perkalian matriks batch ke GPU. Reranker dengan `-ngl 0` terukur sekitar 1,6 s untuk 20 kandidat;
  dengan `CUDA_VISIBLE_DEVICES=` kosong server mencatat "no usable GPU found" dan waktunya 5,98 s.
- **Perbandingan adalah konfigurasi, bukan runtime murni.** Format bobot, kernel, KV cache, dan
  jumlah slot berbeda antar runtime.
- **Beban berkelanjutan lebih lambat.** Angka LLM diukur setelah GPU dingin; beban panjang pada
  laptop ini memicu thermal slowdown dan menurunkan kecepatan.

## Mengulang pengujian

### LLM

| Path | Isi |
| --- | --- |
| `scripts/llm/bench.py`, `agent_bench.py` | Uji throughput dan sesi agent |
| `scripts/llm/run_qwen3.5-4b.sh` | Qwen3.5-4B pada empat runtime |
| `scripts/llm/run_large_moe.sh` | Qwen3.6-35B-A3B dan Qwen3-Coder-30B-A3B pada Ollama dan llama.cpp |
| `scripts/llm/vllm_try.sh`, `sglang_try.sh` | Launcher vLLM dan SGLang |
| `scripts/ollama/` | Modelfile untuk tag Ollama dengan context 32.768 |

Salin isi `scripts/llm/` ke direktori kerja yang berisi `models/`, `envs/` (venv vLLM dan SGLang),
`llama.cpp/`, dan `prompt.txt`. `prompt.txt` adalah teks pengisi prompt, yaitu `functools.py` dari
Python 3.12.3:

```bash
cp "$(python3.12 -c 'import functools; print(functools.__file__)')" prompt.txt
```

Server Ollama dijalankan dengan `OLLAMA_NUM_PARALLEL=1`, `OLLAMA_MAX_LOADED_MODELS=1`,
`OLLAMA_FLASH_ATTENTION=1`, dan `OLLAMA_KV_CACHE_TYPE=q8_0`. Tag dibuat dari Modelfile di
`scripts/ollama/`; blob model yang diuji:

| Tag | Blob |
| --- | --- |
| `qwen3.5-4b-32k` | `sha256-81fb60c7daa80fc1123380b98970b320ae233409f0f71a72ed7b9b0d62f40490` |
| `qwen3.6-32k` | `sha256-f5ee307a2982106a6eb82b62b2c00b575c9072145a759ae4660378acda8dcf2d` |
| `qwen3-coder-32k` | `sha256-9124a16eb1a7429ca76835b5378d3fa99ab4b6c7ceb8c98f6bb1a69050d39c13` |

```bash
ollama create qwen3.5-4b-32k -f scripts/ollama/Modelfile.qwen3.5-4b-32k
BENCH_WORKDIR=<direktori kerja> ./run_qwen3.5-4b.sh
BENCH_WORKDIR=<direktori kerja> QWEN3_CODER_GGUF=<GGUF Qwen3-Coder-30B-A3B> ./run_large_moe.sh
```

Sebelum setiap runtime, pipeline menghentikan llama-server di port 8080, semua proses vLLM dan SGLang,
dan semua model Ollama yang sedang dimuat. Hasilnya ditambahkan ke `results*.jsonl` di direktori kerja.

### Layanan

| Skrip | Server yang dibutuhkan |
| --- | --- |
| `fetch_fleurs.py` | Tidak ada; mengunduh 100 klip FLEURS `id_id` dev |
| `asr_bench.py`, `join_clips.py`, `asr_long.py` | Qwen3-ASR di `:5005` |
| `tts_bench.py` | OmniVoice di `:5004` |
| `embed_bench.py` | Ollama dengan tag `bge-m3` ber-`num_gpu 0` |
| `rerank_bench.py` | Tiga llama-server reranker di `:5002`, `:5012`, dan `:5013`; lihat docstring skrip |
| `docling_bench.py` | docling-serve di `:5003` |

Setiap skrip menjelaskan argumennya di docstring-nya. Uji ASR membutuhkan `jiwer`, dan
`join_clips.py` membutuhkan `numpy` serta `soundfile`.
