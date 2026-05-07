import sys
import os

if sys.stdout is None:
    sys.stdout = open(os.devnull, "w")
if sys.stderr is None:
    sys.stderr = open(os.devnull, "w")

import json
import re
import math
from datetime import datetime
from typing import List, Tuple, Dict, Optional

import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama
from faster_whisper import WhisperModel
from rank_bm25 import BM25Okapi

try:
    from ddgs import DDGS
    HAS_DDGS = True
except ImportError:
    try:
        from duckduckgo_search import DDGS
        HAS_DDGS = True
    except ImportError:
        HAS_DDGS = False

import matplotlib
import os
os.environ["QT_API"] = "pyside6"
matplotlib.use('QtAgg')
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from sklearn.decomposition import PCA

from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QTimer, QThread, Signal, QRectF, QSize, QPoint
from PySide6.QtGui import QFont, QPainter, QPen, QColor, QBrush, QLinearGradient, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QFileDialog, QFrame, QGraphicsOpacityEffect,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem,
    QMainWindow, QPushButton, QScrollArea, QVBoxLayout, QWidget,
    QProgressBar, QTabWidget, QTextEdit, QTableWidget, QTableWidgetItem,
    QHeaderView, QSizePolicy, QStackedWidget, QSplitter, QGraphicsScene, QGraphicsView, QCheckBox
)


# ================= ANIMATED SPINNER ================= #
class ModernSpinner(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(32, 32)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.rotate)

    def showEvent(self, event):
        self.timer.start(12)
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def rotate(self):
        self.angle = (self.angle + 8) % 360
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        gradient = QLinearGradient(0, 0, self.width(), self.height())
        gradient.setColorAt(0, QColor("#00d4ff"))
        gradient.setColorAt(1, QColor("#00b8e6"))
        pen = QPen(QBrush(gradient), 4)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        rect = QRectF(4, 4, 24, 24)
        painter.drawArc(rect, -self.angle * 16, 270 * 16)


# ================= LOADING DOTS ================= #
class LoadingDots(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(60, 20)
        self.dot_offset = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.animate_dots)

    def showEvent(self, event):
        self.timer.start(300)
        super().showEvent(event)

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def animate_dots(self):
        self.dot_offset = (self.dot_offset + 1) % 4
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        color = QColor("#00d4ff")
        for i in range(3):
            x = 10 + i * 20
            size = 6 if (self.dot_offset == i) else 4
            painter.setBrush(QBrush(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPoint(x, 10), size, size)


# ================= FILE PROCESSOR ================= #
class FileProcessor:
    def __init__(self):
        self.whisper = None
        self._ocr_reader = None  # EasyOCR reader, lazily initialised

    def _get_ocr_reader(self):
        """Lazily load EasyOCR (downloads ~100 MB of models on first use)."""
        if self._ocr_reader is None:
            import easyocr
            self._ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
        return self._ocr_reader

    def _ocr_page(self, page) -> str:
        """Render a PDF page to an image and run EasyOCR on it."""
        try:
            reader = self._get_ocr_reader()
            # Render at 2x scale for better OCR accuracy
            pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))
            img_bytes = pix.tobytes("png")
            results = reader.readtext(img_bytes, detail=0, paragraph=True)
            return " ".join(results).strip()
        except Exception as e:
            return ""

    def extract_text_from_pdf(
        self,
        pdf_path: str,
        progress_cb=None,
    ) -> List[Tuple[int, str]]:
        """
        Extract text from every page of a PDF.
        - Pages with native selectable text: fast path (PyMuPDF).
        - Pages that are image-only / scanned: OCR via EasyOCR.
        progress_cb(str) is called with status messages if provided.
        """
        doc = fitz.open(pdf_path)
        n = len(doc)
        pages = []
        ocr_needed = []  # page indices that need OCR

        # ---- Pass 1: fast text extraction ----
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            # FYP Enhancement: Multi-Modal Table Extraction
            try:
                tabs = page.find_tables()
                for tab in tabs:
                    text += "\n\n[Extracted Table]\n" + tab.to_markdown()
            except Exception:
                pass

            if text:
                pages.append((i + 1, text))
            else:
                ocr_needed.append(i)  # image-based page — queue for OCR

        # ---- Pass 2: OCR for image-based pages ----
        if ocr_needed:
            if progress_cb:
                progress_cb(f"Detected {len(ocr_needed)} image page(s) — starting OCR...")
            for count, page_idx in enumerate(ocr_needed, 1):
                if progress_cb:
                    progress_cb(f"OCR page {count}/{len(ocr_needed)} of {os.path.basename(pdf_path)}...")
                page = doc.load_page(page_idx)
                ocr_text = self._ocr_page(page)
                if ocr_text:
                    pages.append((page_idx + 1, ocr_text))

        doc.close()
        # Sort by page number (OCR pages are appended after text pages)
        pages.sort(key=lambda x: x[0])

        if not pages:
            raise ValueError(
                "No text could be extracted from this PDF even after OCR. "
                "The file may be corrupted, encrypted, or have very low image quality."
            )

        return pages

    def extract_text_from_audio(self, audio_path: str) -> List[Tuple[int, str]]:
        if self.whisper is None:
            self.whisper = WhisperModel("base", device="cpu", compute_type="int8")
        segments, info = self.whisper.transcribe(audio_path, beam_size=5)
        full_text = " ".join([segment.text for segment in segments])
        return [(1, full_text)]

    def chunk_text(self, pages, source_name="unknown", chunk_size=300, overlap=80):
        chunks = []
        for page_num, text in pages:
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk:
                    # FYP Enhancement: Hierarchical Chunking (store full parent page text)
                    chunks.append({
                        "page": page_num, 
                        "text": chunk, 
                        "source": source_name,
                        "parent_text": text  # The parent chunk!
                    })
        return chunks


# ================= RAG SYSTEM ================= #
class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []
        self.documents = {}
        self.bm25 = None
        self.pca = None
        self.pca_coords = []
        self.doc_paths = {}

    def add_document(self, doc_name, chunks, doc_path=None):
        for c in chunks:
            c["source"] = doc_name
        self.documents[doc_name] = chunks
        if doc_path:
            self.doc_paths[doc_name] = doc_path
        self._rebuild_index()

    def remove_document(self, doc_name):
        if doc_name in self.documents:
            del self.documents[doc_name]
        self._rebuild_index()

    def _rebuild_index(self):
        self.chunks = []
        for doc_chunks in self.documents.values():
            self.chunks.extend(doc_chunks)
        if not self.chunks:
            self.index = None
            self.bm25 = None
            self.pca = None
            self.pca_coords = []
            return
        
        texts = [c["text"] for c in self.chunks]
        emb = self.embedder.encode(texts, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))
        tokenized = [t.lower().split() for t in texts]
        self.bm25 = BM25Okapi(tokenized)

        # FYP Enhancement: PCA for Vector Space Visualization
        if len(self.chunks) >= 2:
            try:
                self.pca = PCA(n_components=2)
                self.pca_coords = self.pca.fit_transform(emb).tolist()
            except Exception:
                self.pca = None
                self.pca_coords = []
        else:
            self.pca = None
            self.pca_coords = []

    def search(self, query, k=5, doc_filter="All Documents"):
        if not self.chunks or self.index is None:
            return [], None

        # Filter chunks and embeddings if a specific document is selected
        if doc_filter and doc_filter != "All Documents":
            filtered_indices = [i for i, c in enumerate(self.chunks) if c["source"] == doc_filter]
            if not filtered_indices:
                return [], None
            
            # Create a temporary index for the filtered chunks
            texts = [self.chunks[i]["text"] for i in filtered_indices]
            emb = self.embedder.encode(texts, show_progress_bar=False)
            emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
            temp_index = faiss.IndexFlatIP(emb.shape[1])
            temp_index.add(emb.astype("float32"))
            
            q_emb = self.embedder.encode([query])
            q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
            
            query_pca = self.pca.transform(q_emb)[0].tolist() if self.pca else [0, 0]
            
            scores, idxs = temp_index.search(q_emb_norm.astype("float32"), min(k, len(filtered_indices)))
            
            results = []
            for score, local_idx in zip(scores[0], idxs[0]):
                if local_idx != -1:
                    global_idx = filtered_indices[local_idx]
                    r = self.chunks[global_idx].copy()
                    r["score"] = float(score)
                    r["index"] = int(global_idx)
                    results.append(r)
            return results, query_pca

        q_emb = self.embedder.encode([query])
        q_emb_norm = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)
        
        # Get query PCA coords
        query_pca = self.pca.transform(q_emb)[0].tolist() if self.pca else [0, 0]

        faiss_scores, faiss_idx = self.index.search(q_emb_norm.astype("float32"), min(k*2, len(self.chunks)))
        tokenized_query = query.lower().split()
        bm25_scores = self.bm25.get_scores(tokenized_query)
        bm25_ranked = np.argsort(bm25_scores)[::-1][:min(k*2, len(self.chunks))]
        
        rrf_scores = {}
        RRF_K = 60
        for rank, idx in enumerate(faiss_idx[0]):
            if 0 <= idx < len(self.chunks):
                rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (RRF_K + rank + 1)
        for rank, idx in enumerate(bm25_ranked):
            rrf_scores[int(idx)] = rrf_scores.get(int(idx), 0) + 1.0 / (RRF_K + rank + 1)
            
        sorted_results = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        results = []
        for idx, rrf_score in sorted_results:
            if idx < len(self.chunks):
                r = self.chunks[idx].copy()
                r["score"] = float(rrf_score)
                r["index"] = int(idx)
                faiss_pos = np.where(faiss_idx[0] == idx)[0]
                r["faiss_score"] = float(faiss_scores[0][faiss_pos[0]]) if len(faiss_pos) > 0 else 0.0
                results.append(r)
                
        return results, query_pca

    def has_documents(self):
        # A document only counts if it actually produced chunks (i.e. has indexable text)
        return bool(self.chunks)

    def get_document_names(self):
        return list(self.documents.keys())

    def get_document_chunk_count(self, doc_name):
        return len(self.documents.get(doc_name, []))

    def total_chunks(self):
        return len(self.chunks)

# ================= SEMANTIC CACHE ================= #
class SemanticCache:
    def __init__(self, embedder):
        self.embedder = embedder
        self.index = None
        self.cache_data = []

    def add(self, query, answer, sources, is_web):
        emb = self.embedder.encode([query])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        if self.index is None:
            self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))
        self.cache_data.append({
            "query": query, "answer": answer, 
            "sources": sources, "is_web": is_web
        })

    def search(self, query, threshold=0.92):
        if self.index is None: return None
        emb = self.embedder.encode([query])
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
        scores, indices = self.index.search(emb.astype("float32"), 1)
        if scores[0][0] >= threshold:
            return self.cache_data[indices[0][0]]
        return None

    def clear(self):
        self.index = None
        self.cache_data = []

# ================= WEB SEARCH FALLBACK ================= #
class WebSearchFallback:
    @staticmethod
    def search(query, max_results=5):
        if not HAS_DDGS:
            return []
        try:
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=max_results))
            return [{"title": r.get("title", ""), "text": r.get("body", ""),
                     "url": r.get("href", ""), "source": "🌐 Web", "page": "—"} for r in results]
        except Exception:
            return []


# ================= LOCAL LLM ================= #
class LocalLLM:
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = Llama(model_path=path, n_ctx=4096, n_threads=4, n_gpu_layers=35, verbose=False)

    def generate(self, messages, max_tokens=256):
        result = ""
        try:
            for chunk in self.model.create_chat_completion(
                    messages=messages, max_tokens=max_tokens, temperature=0.3, stream=True):
                delta = chunk["choices"][0]["delta"]
                if "content" in delta:
                    result += delta["content"]
        except Exception:
            pass
        return result

    def stream(self, messages, max_tokens=1024):
        for chunk in self.model.create_chat_completion(
                messages=messages, max_tokens=max_tokens, temperature=0.6, top_p=0.9, stream=True):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]


# ================= THREADS ================= #
class MultiProcessingThread(QThread):
    progress_signal = Signal(str)
    finished_signal = Signal(bool, str, list)

    def __init__(self, paths, rag):
        super().__init__()
        self.paths = paths
        self.rag = rag

    def run(self):
        try:
            proc = FileProcessor()
            total = 0
            names = []
            for path in self.paths:
                name = os.path.basename(path)
                self.progress_signal.emit(f"Processing: {name}...")
                if path.lower().endswith(('.mp3', '.wav', '.m4a')):
                    pages = proc.extract_text_from_audio(path)
                else:
                    pages = proc.extract_text_from_pdf(
                        path,
                        progress_cb=lambda msg: self.progress_signal.emit(msg)
                    )
                chunks = proc.chunk_text(pages, source_name=name)
                self.rag.add_document(name, chunks, doc_path=path)
                total += len(chunks)
                names.append(name)
            self.finished_signal.emit(True, f"Indexed {len(self.paths)} file(s), {total} chunks.", names)
        except Exception as e:
            self.finished_signal.emit(False, str(e), [])


class ModelLoadThread(QThread):
    progress_signal = Signal(int, str)
    finished_signal = Signal(bool, str)

    def __init__(self, llm, path):
        super().__init__()
        self.llm = llm
        self.path = path

    def run(self):
        try:
            self.progress_signal.emit(20, "Initializing model...")
            self.msleep(200)
            self.progress_signal.emit(40, "Loading weights...")
            self.llm.load(self.path)
            self.progress_signal.emit(80, "Optimizing for inference...")
            self.msleep(300)
            self.progress_signal.emit(100, "Model ready!")
            self.msleep(200)
            self.finished_signal.emit(True, "Model loaded successfully!")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


class SearchPipelineThread(QThread):
    result_signal = Signal(str, list, bool, list)  # query, results, is_web, query_pca

    def __init__(self, llm, rag, query, chat_history, model_loaded, use_hyde=False, doc_filter="All Documents", allow_web=False):
        super().__init__()
        self.llm = llm
        self.rag = rag
        self.query = query
        self.chat_history = chat_history
        self.model_loaded = model_loaded
        self.use_hyde = use_hyde
        self.doc_filter = doc_filter
        self.allow_web = allow_web

    def run(self):
        search_query = self.query
        
        # Reformulation
        if self.model_loaded and self.llm.model and len(self.chat_history) >= 2:
            try:
                history_text = ""
                for msg in self.chat_history[-6:]:
                    role = "User" if msg["role"] == "user" else "Assistant"
                    history_text += f"{role}: {msg['content'][:200]}\n"
                msgs = [
                    {"role": "system", "content": "Rewrite the user's latest question as a standalone search query using conversation context. Output ONLY the query."},
                    {"role": "user", "content": f"Conversation:\n{history_text}\nLatest: {self.query}\n\nStandalone query:"}
                ]
                reformulated = self.llm.generate(msgs, max_tokens=100).strip()
                if reformulated and len(reformulated) < 500:
                    search_query = reformulated
            except Exception:
                pass

        # FYP Enhancement: HyDE (Hypothetical Document Embeddings)
        if self.use_hyde and self.model_loaded and self.llm.model:
            try:
                hyde_msgs = [
                    {"role": "system", "content": "Generate a brief, highly factual hypothetical answer to the user's question to be used for semantic search. Do not include conversational filler."},
                    {"role": "user", "content": search_query}
                ]
                hypothetical_answer = self.llm.generate(hyde_msgs, max_tokens=150).strip()
                if hypothetical_answer:
                    search_query = search_query + " " + hypothetical_answer
            except Exception:
                pass

        results = []
        is_web = False
        query_pca = [0, 0]

        has_indexed_docs = self.rag.has_documents()  # True only when chunks actually exist

        if self.rag.has_documents():
            results, query_pca = self.rag.search(search_query, doc_filter=self.doc_filter)

        # Web fallback — ONLY when no documents are indexed at all AND user enabled it.
        if not has_indexed_docs and self.allow_web:
            web = WebSearchFallback.search(self.query)
            if web:
                results = web
                is_web = True

        # If docs are indexed but search returned nothing, emit empty results so the
        # LLM can respond with "the context doesn't cover this topic" rather than
        # confabulating from web data the user didn't ask for.
        self.result_signal.emit(self.query, results, is_web, query_pca)


class StreamThread(QThread):
    token_signal = Signal(str)
    finished_signal = Signal()

    def __init__(self, llm, messages):
        super().__init__()
        self.llm = llm
        self.messages = messages
        self._is_stopped = False

    def stop(self):
        self._is_stopped = True

    def run(self):
        try:
            for t in self.llm.stream(self.messages):
                if self._is_stopped:
                    break
                self.token_signal.emit(t)
        except Exception as e:
            self.token_signal.emit(f"\n[Error: {str(e)}]")
        finally:
            self.finished_signal.emit()


class EvalThread(QThread):
    progress_signal = Signal(int, int)
    result_signal = Signal(list)
    finished_signal = Signal()

    def __init__(self, llm, rag, embedder, questions):
        super().__init__()
        self.llm = llm
        self.rag = rag
        self.embedder = embedder
        self.questions = questions

    def run(self):
        results = []
        for i, q in enumerate(self.questions):
            self.progress_signal.emit(i + 1, len(self.questions))
            search_results, _ = self.rag.search(q)
            context = "\n".join([r["text"][:300] for r in search_results]) if search_results else ""
            msgs = [{"role": "system", "content": f"Answer based on context:\n{context}"},
                    {"role": "user", "content": q}]
            try:
                answer = self.llm.generate(msgs, max_tokens=512)
            except Exception:
                answer = "[Error generating answer]"

            try:
                q_emb = self.embedder.encode([q])[0]
                a_emb = self.embedder.encode([answer])[0]
                relevancy = float(np.dot(q_emb, a_emb) / (np.linalg.norm(q_emb) * np.linalg.norm(a_emb) + 1e-9))
            except Exception:
                relevancy = 0.0

            ctx_score = float(np.mean([r.get("score", 0) for r in search_results])) if search_results else 0.0

            try:
                faith_msgs = [
                    {"role": "system", "content": "Rate how well the answer is supported by context. Reply with ONLY a number 0.0-1.0."},
                    {"role": "user", "content": f"Context: {context[:800]}\nAnswer: {answer[:400]}\nScore:"}
                ]
                faith_text = self.llm.generate(faith_msgs, max_tokens=10)
                match = re.search(r'([01]\.?\d*)', faith_text)
                faithfulness = min(max(float(match.group(1)), 0), 1) if match else 0.5
            except Exception:
                faithfulness = 0.5

            results.append({"question": q, "answer": answer.strip(),
                           "faithfulness": faithfulness, "relevancy": relevancy, "context_score": ctx_score})
        self.result_signal.emit(results)
        self.finished_signal.emit()


STYLESHEET = """
* { font-family: 'Inter', 'Segoe UI', sans-serif; }
QMainWindow { background-color: #0b1117; }

/* Sidebar Styling */
QFrame#sidebar { background-color: #0d161e; border-right: 1px solid #1c2c3e; }
QLabel#sidebarTitle { color: #00d4ff; font-size: 20px; font-weight: 800; letter-spacing: 2px; padding: 20px 0; border-bottom: 1px solid #1c2c3e; margin-bottom: 10px; }
QLabel#sectionLabel { color: #8a9ba8; font-size: 11px; font-weight: 700; text-transform: uppercase; letter-spacing: 1.5px; padding: 15px 10px 5px 10px; }

/* Components */
QComboBox { background-color: #162431; color: #e1e8ed; border: 1px solid #243b53; border-radius: 6px; padding: 10px 15px; font-size: 13px; }
QComboBox:hover { border: 1px solid #00d4ff; }
QComboBox::drop-down { border: none; width: 30px; }
QComboBox::down-arrow { image: none; border-left: 4px solid transparent; border-right: 4px solid transparent; border-top: 5px solid #8a9ba8; width: 0; height: 0; margin-right: 10px; }
QComboBox QAbstractItemView { background-color: #162431; color: #e1e8ed; selection-background-color: #00d4ff; selection-color: #0b1117; outline: none; border: 1px solid #243b53; }

QPushButton { background-color: #162431; color: #e1e8ed; border: 1px solid #243b53; border-radius: 6px; padding: 10px 20px; font-size: 13px; font-weight: 600; }
QPushButton:hover { background-color: #1c2c3e; border: 1px solid #00d4ff; }
QPushButton#newChatButton { background-color: #00d4ff; color: #0b1117; border: none; font-weight: 800; margin: 10px; }
QPushButton#newChatButton:hover { background-color: #00b8e6; }
QPushButton#attachButton { background-color: #162431; color: #00d4ff; border: 1px solid #00d4ff; font-weight: 800; }
QPushButton#sendButton { background-color: #00d4ff; color: #0b1117; border: none; border-radius: 8px; font-weight: 800; }
QPushButton#sendButton:hover { background-color: #00b8e6; }

QListWidget { background-color: transparent; border: none; outline: none; }
QListWidget::item { background-color: #162431; color: #e1e8ed; border-radius: 8px; margin-bottom: 8px; padding: 12px; }
QListWidget::item:selected { background-color: rgba(0, 212, 255, 0.15); border: 1px solid #00d4ff; color: #00d4ff; }

/* Chat Container */
QFrame#chatContainer { background-color: #0b1117; }
QFrame#chatHeader { background-color: #0b1117; border-bottom: 1px solid #1c2c3e; }
QLabel#chatTitle { color: #f5f8fa; font-size: 22px; font-weight: 800; letter-spacing: 1px; }

/* Chat Bubbles */
QFrame#userBubble { background-color: #162431; border-radius: 12px; border: 1px solid #243b53; padding: 12px; margin-left: 50px; }
QFrame#userBubble QLabel { color: #f5f8fa; font-size: 14px; line-height: 1.4; }
QFrame#botBubble { background-color: rgba(28, 44, 62, 0.4); border-radius: 12px; border: 1px solid #243b53; padding: 12px; margin-right: 50px; }
QFrame#botBubble QLabel { color: #e1e8ed; font-size: 14px; line-height: 1.5; }

/* Input Area */
QFrame#inputFrame { background-color: #0b1117; border-top: 1px solid #1c2c3e; padding: 20px; }
QLineEdit { background-color: #162431; color: #f5f8fa; border: 1px solid #243b53; border-radius: 8px; padding: 12px 15px; font-size: 14px; }
QLineEdit:focus { border: 1px solid #00d4ff; }

/* Tabs & Scrolls */
QTabWidget::pane { border: none; background: transparent; }
QTabBar::tab { background: #0d161e; color: #8a9ba8; padding: 10px 15px; font-size: 12px; font-weight: 700; border-bottom: 2px solid transparent; }
QTabBar::tab:selected { color: #00d4ff; border-bottom: 2px solid #00d4ff; }

QProgressBar { background-color: #162431; border: none; border-radius: 4px; height: 6px; text-align: center; }
QProgressBar::chunk { background-color: #00d4ff; border-radius: 4px; }

QScrollBar:vertical { border: none; background: #0b1117; width: 8px; margin: 0; }
QScrollBar::handle:vertical { background: #1c2c3e; min-height: 20px; border-radius: 4px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
"""



class RagBot(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rag = RAGSystem()
        self.semantic_cache = SemanticCache(self.rag.embedder)
        self.llm = LocalLLM()
        self.model_loaded = False
        self.model_loading = False   # True while the model thread is running
        self.current_chat_id = ""
        self.animations = []
        self.current_bot_label = None
        self.current_answer = ""
        self.chat_history = []
        self.current_sources = []
        self.is_web_source = False
        self.pending_message = ""
        self.setWindowTitle("RAGBOT AI (FYP EDITION)")
        self.resize(1400, 900)
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("chats"):
            os.makedirs("chats")
        self.setStyleSheet(STYLESHEET)
        self.setup_ui()

    def setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # ---- LEFT SIDEBAR ---- #
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(340)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(16, 16, 16, 16)
        sidebar_layout.setSpacing(12)

        title_label = QLabel("RAGBOT")
        title_label.setObjectName("sidebarTitle")
        title_label.setAlignment(Qt.AlignCenter)
        sidebar_layout.addWidget(title_label)

        # Model Selection
        model_section_label = QLabel("AI Model")
        model_section_label.setObjectName("sectionLabel")
        sidebar_layout.addWidget(model_section_label)

        self.model_dropdown = QComboBox()
        self.model_dropdown.addItem("Select LLM Model...")
        self.populate_models()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_selected)
        sidebar_layout.addWidget(self.model_dropdown)

        self.model_progress = QProgressBar()
        self.model_progress.setVisible(False)
        sidebar_layout.addWidget(self.model_progress)

        self.progress_label = QLabel("")
        self.progress_label.setObjectName("progressLabel")
        self.progress_label.setAlignment(Qt.AlignCenter)
        self.progress_label.setVisible(False)
        sidebar_layout.addWidget(self.progress_label)

        # FYP Enhancement: HyDE Toggle
        self.hyde_checkbox = QCheckBox("Enable HyDE (Hypothetical Doc Embeddings)")
        self.hyde_checkbox.setStyleSheet("color: #8a9ba8; font-size: 11px;")
        self.hyde_checkbox.setToolTip("Generates a hypothetical answer before searching for better semantic matching.")
        sidebar_layout.addWidget(self.hyde_checkbox)

        self.web_search_checkbox = QCheckBox("Enable Web Search Fallback")
        self.web_search_checkbox.setStyleSheet("color: #8a9ba8; font-size: 11px;")
        self.web_search_checkbox.setChecked(False) 
        sidebar_layout.addWidget(self.web_search_checkbox)

        # ---- TABBED SIDEBAR SECTIONS ---- #
        self.sidebar_tabs = QTabWidget()
        sidebar_layout.addWidget(self.sidebar_tabs)

        # Tab 1: Chat History
        chat_tab = QWidget()
        chat_tab_layout = QVBoxLayout(chat_tab)
        chat_tab_layout.setContentsMargins(0, 8, 0, 0)
        chat_tab_layout.setSpacing(8)

        new_chat_btn = QPushButton("➕ New Chat")
        new_chat_btn.setObjectName("newChatButton")
        new_chat_btn.clicked.connect(self.start_new_chat)
        chat_tab_layout.addWidget(new_chat_btn)

        self.history_list = QListWidget()
        self.history_list.itemClicked.connect(self.load_selected_chat)
        chat_tab_layout.addWidget(self.history_list)
        self.sidebar_tabs.addTab(chat_tab, "💬 Chats")

        # Tab 2: Document Manager
        doc_tab = QWidget()
        doc_tab_layout = QVBoxLayout(doc_tab)
        doc_tab_layout.setContentsMargins(0, 8, 0, 0)
        doc_tab_layout.setSpacing(8)

        add_files_btn = QPushButton("📁 Add Files")
        add_files_btn.setObjectName("newChatButton")
        add_files_btn.clicked.connect(self.select_attachment)
        doc_tab_layout.addWidget(add_files_btn)

        self.doc_status_label = QLabel("No documents indexed")
        self.doc_status_label.setObjectName("progressLabel")
        self.doc_status_label.setAlignment(Qt.AlignCenter)
        doc_tab_layout.addWidget(self.doc_status_label)

        self.doc_list = QListWidget()
        doc_tab_layout.addWidget(self.doc_list)
        self.sidebar_tabs.addTab(doc_tab, "📁 Docs")

        # Tab 3: Evaluation Dashboard
        eval_tab = QWidget()
        eval_tab_layout = QVBoxLayout(eval_tab)
        eval_tab_layout.setContentsMargins(0, 8, 0, 0)
        eval_tab_layout.setSpacing(8)

        eval_info = QLabel("Enter test questions (one per line):")
        eval_info.setObjectName("progressLabel")
        eval_tab_layout.addWidget(eval_info)

        self.eval_input = QTextEdit()
        self.eval_input.setPlaceholderText("What is machine learning?\nExplain neural networks.")
        self.eval_input.setMaximumHeight(120)
        eval_tab_layout.addWidget(self.eval_input)

        self.eval_btn = QPushButton("▶ Run Evaluation")
        self.eval_btn.clicked.connect(self.run_evaluation)
        eval_tab_layout.addWidget(self.eval_btn)

        self.eval_progress_label = QLabel("")
        self.eval_progress_label.setObjectName("progressLabel")
        self.eval_progress_label.setAlignment(Qt.AlignCenter)
        self.eval_progress_label.setVisible(False)
        eval_tab_layout.addWidget(self.eval_progress_label)

        self.eval_table = QTableWidget(0, 4)
        self.eval_table.setHorizontalHeaderLabels(["Question", "Faith.", "Relev.", "Ctx"])
        self.eval_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.Stretch)
        for col in range(1, 4):
            self.eval_table.horizontalHeader().setSectionResizeMode(col, QHeaderView.ResizeToContents)
        eval_tab_layout.addWidget(self.eval_table)

        self.eval_avg_label = QLabel("")
        self.eval_avg_label.setObjectName("progressLabel")
        self.eval_avg_label.setAlignment(Qt.AlignCenter)
        eval_tab_layout.addWidget(self.eval_avg_label)
        self.sidebar_tabs.addTab(eval_tab, "📊 Eval")

        # FYP Enhancement: Tab 4 Vector Space Map
        space_tab = QWidget()
        space_layout = QVBoxLayout(space_tab)
        space_layout.setContentsMargins(0, 8, 0, 0)
        
        self.figure = Figure(facecolor='#1f2937')
        self.canvas = FigureCanvasQTAgg(self.figure)
        space_layout.addWidget(self.canvas)
        
        self.ax = self.figure.add_subplot(111)
        self.ax.set_facecolor('#1f2937')
        self.ax.tick_params(colors='white')
        for spine in self.ax.spines.values():
            spine.set_color('#4b5563')
        self.ax.set_title("Semantic Vector Space", color='white', pad=10)
        
        self.sidebar_tabs.addTab(space_tab, "🌌 Map")

        main_layout.addWidget(sidebar)

        # ---- SPLITTER FOR MAIN VIEW (Chat + PDF Viewer) ---- #
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setHandleWidth(4)
        self.main_splitter.setStyleSheet("QSplitter::handle { background-color: #374151; }")
        
        # ---- RIGHT CHAT AREA ---- #
        chat_container = QFrame()
        chat_container.setObjectName("chatContainer")
        chat_layout = QVBoxLayout(chat_container)
        chat_layout.setContentsMargins(0, 0, 0, 0)
        chat_layout.setSpacing(0)

        chat_header = QFrame()
        chat_header.setObjectName("chatHeader")
        chat_header_layout = QHBoxLayout(chat_header)
        chat_header_layout.setContentsMargins(32, 18, 32, 18)
        self.chat_title = QLabel("INTELLISEARCH RAG")
        self.chat_title.setObjectName("chatTitle")
        chat_header_layout.addWidget(self.chat_title)
        chat_header_layout.addStretch()

        # Doc Filter Dropdown
        self.doc_filter_dropdown = QComboBox()
        self.doc_filter_dropdown.addItem("All Documents")
        self.doc_filter_dropdown.setFixedWidth(200)
        chat_header_layout.addWidget(self.doc_filter_dropdown)
        
        chat_layout.addWidget(chat_header)

        # Chat Messages Area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll_content = QWidget()
        self.messages_layout = QVBoxLayout(scroll_content)
        self.messages_layout.setContentsMargins(24, 24, 24, 24)
        self.messages_layout.setSpacing(16)
        self.messages_layout.addStretch()
        self.chat_scroll.setWidget(scroll_content)
        chat_layout.addWidget(self.chat_scroll)

        # Loading indicator
        self.loading_frame = QFrame()
        self.loading_frame.setObjectName("loadingFrame")
        self.loading_frame.setVisible(False)
        loading_layout = QHBoxLayout(self.loading_frame)
        loading_layout.setContentsMargins(16, 12, 16, 12)
        self.loading_spinner = ModernSpinner()
        loading_layout.addWidget(self.loading_spinner)
        self.loading_label = QLabel("Processing...")
        loading_layout.addWidget(self.loading_label)
        self.loading_dots = LoadingDots()
        loading_layout.addWidget(self.loading_dots)
        loading_layout.addStretch()
        chat_layout.addWidget(self.loading_frame)

        # Input Area
        input_frame = QFrame()
        input_frame.setObjectName("inputFrame")
        input_layout = QVBoxLayout(input_frame)
        input_layout.setContentsMargins(24, 16, 24, 20)
        input_layout.setSpacing(12)

        self.attachment_chip = QFrame()
        self.attachment_chip.setObjectName("attachmentChip")
        self.attachment_chip.setVisible(False)
        chip_layout = QHBoxLayout(self.attachment_chip)
        chip_layout.setContentsMargins(12, 6, 12, 6)
        self.attachment_label = QLabel("")
        chip_layout.addWidget(self.attachment_label)
        chip_layout.addStretch()
        clear_attach_btn = QPushButton("✕")
        clear_attach_btn.setFixedSize(24, 24)
        clear_attach_btn.clicked.connect(self.clear_attachment)
        chip_layout.addWidget(clear_attach_btn)
        input_layout.addWidget(self.attachment_chip)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(12)

        self.attachment_button = QPushButton("📎  Attach File")
        self.attachment_button.setObjectName("attachButton")
        self.attachment_button.setFixedHeight(52)
        self.attachment_button.clicked.connect(self.select_attachment)
        bottom_row.addWidget(self.attachment_button)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)
        bottom_row.addWidget(self.input_field)

        self.send_button = QPushButton("Send ➤")
        self.send_button.setObjectName("sendButton")
        self.send_button.setFixedHeight(52)
        self.send_button.clicked.connect(self.send_message)
        bottom_row.addWidget(self.send_button)

        self.stop_button = QPushButton("⬛ Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setFixedHeight(52)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_generation)
        bottom_row.addWidget(self.stop_button)

        input_layout.addLayout(bottom_row)
        chat_layout.addWidget(input_frame)

        self.main_splitter.addWidget(chat_container)

        # ---- FYP Enhancement: PDF VIEWER PANEL ---- #
        self.pdf_container = QFrame()
        self.pdf_container.setObjectName("chatContainer") # Reuse dark theme
        pdf_layout = QVBoxLayout(self.pdf_container)
        pdf_layout.setContentsMargins(0,0,0,0)
        pdf_layout.setSpacing(0)
        
        pdf_header = QFrame()
        pdf_header.setObjectName("chatHeader")
        pdf_header_layout = QHBoxLayout(pdf_header)
        pdf_header_layout.setContentsMargins(20, 18, 20, 18)
        self.pdf_title = QLabel("PDF VIEWER")
        self.pdf_title.setObjectName("chatTitle")
        self.pdf_title.setStyleSheet("font-size: 18px;")
        pdf_header_layout.addWidget(self.pdf_title)
        pdf_header_layout.addStretch()
        close_pdf_btn = QPushButton("✕")
        close_pdf_btn.setFixedSize(30, 30)
        close_pdf_btn.setStyleSheet("background: #374151; color: white;")
        close_pdf_btn.clicked.connect(lambda: self.pdf_container.hide())
        pdf_header_layout.addWidget(close_pdf_btn)
        pdf_layout.addWidget(pdf_header)

        self.pdf_scroll = QScrollArea()
        self.pdf_scroll.setBackgroundRole(self.pdf_scroll.backgroundRole())
        self.pdf_scroll.setAlignment(Qt.AlignCenter)
        self.pdf_label = QLabel()
        self.pdf_label.setAlignment(Qt.AlignCenter)
        self.pdf_scroll.setWidget(self.pdf_label)
        pdf_layout.addWidget(self.pdf_scroll)
        
        self.main_splitter.addWidget(self.pdf_container)
        self.pdf_container.hide() # Hidden by default
        
        # Add splitter to main layout
        main_layout.addWidget(self.main_splitter)

        self.populate_history()


    # ---- TOAST ---- #
    def show_toast(self, message, duration=3000):
        toast = QFrame(self)
        toast.setObjectName("toastFrame")
        toast_layout = QHBoxLayout(toast)
        toast_layout.setContentsMargins(16, 10, 16, 10)
        icon = QLabel("✅")
        icon.setStyleSheet("font-size: 16px; background: transparent;")
        toast_layout.addWidget(icon)
        lbl = QLabel(message)
        lbl.setStyleSheet("background: transparent;")
        toast_layout.addWidget(lbl)
        toast.adjustSize()
        margin = 20
        x = self.width() - toast.width() - margin - 10
        toast.move(x, margin)
        toast.show()
        toast.raise_()
        effect = QGraphicsOpacityEffect(toast)
        toast.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(500)
        anim.setStartValue(1.0)
        anim.setEndValue(0.0)
        anim.finished.connect(toast.deleteLater)
        QTimer.singleShot(duration, anim.start)

    # ---- MODEL MANAGEMENT ---- #
    def populate_models(self):
        models_dir = "models"
        if os.path.exists(models_dir):
            for f in os.listdir(models_dir):
                if f.endswith('.gguf'):
                    self.model_dropdown.addItem(f)
        # Auto-load if there is exactly one model available
        if self.model_dropdown.count() == 2:  # placeholder + 1 model
            self.model_dropdown.setCurrentIndex(1)  # triggers on_model_selected → load

    def on_model_selected(self, index):
        if index == 0: return
        self.load_selected_model()

    def load_selected_model(self):
        selected = self.model_dropdown.currentText()
        if selected == "Select LLM Model..." or not selected:
            self.add_message("Please select a valid model from the dropdown.", role="system")
            return
        model_path = os.path.join("models", selected)
        if not os.path.exists(model_path):
            self.add_message(f"Model file not found: {model_path}", role="system")
            return
        self.model_loading = True
        self.model_dropdown.setEnabled(False)
        self.model_progress.setVisible(True)
        self.model_progress.setValue(0)
        self.progress_label.setVisible(True)
        self.progress_label.setText("Initializing...")
        self.model_thread = ModelLoadThread(self.llm, model_path)
        self.model_thread.progress_signal.connect(self.on_model_progress)
        self.model_thread.finished_signal.connect(self.on_model_loaded)
        self.model_thread.start()

    def on_model_progress(self, value, message):
        self.model_progress.setValue(value)
        self.progress_label.setText(message)

    def on_model_loaded(self, success, message):
        self.model_loading = False
        self.model_dropdown.setEnabled(True)
        if success:
            self.model_loaded = True
            self.show_toast(f"{message} You can now start chatting!")
        else:
            self.add_message(f"❌ Error loading model: {message}", role="system")
        QTimer.singleShot(1500, lambda: self.model_progress.setVisible(False))
        QTimer.singleShot(1500, lambda: self.progress_label.setVisible(False))

    # ---- CHAT HISTORY ---- #
    def populate_history(self):
        self.history_list.clear()
        chats_dir = "chats"
        if not os.path.exists(chats_dir): return
        files = sorted(
            [f for f in os.listdir(chats_dir) if f.endswith('.json')],
            key=lambda x: os.path.getmtime(os.path.join(chats_dir, x)),
            reverse=True
        )
        for f in files:
            display_name = f.replace("Chat_", "").replace(".json", "")
            file_path = os.path.join(chats_dir, f)
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(6)
            name_label = QLabel(display_name)
            name_label.setStyleSheet("color: #f3f4f6; font-size: 13px; background: transparent;")
            name_label.setWordWrap(False)
            row_layout.addWidget(name_label, stretch=1)
            del_btn = QPushButton("🗑")
            del_btn.setFixedSize(26, 26)
            del_btn.setStyleSheet("""
                QPushButton { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4);
                    border-radius: 6px; font-size: 12px; color: #ef4444; padding: 0px; }
                QPushButton:hover { background: rgba(239,68,68,0.5); border: 1px solid #ef4444; }
            """)
            del_btn.clicked.connect(lambda checked, fp=file_path: self.delete_chat_by_path(fp))
            row_layout.addWidget(del_btn)
            item = QListWidgetItem()
            item.setData(Qt.UserRole, file_path)
            item.setSizeHint(QSize(280, 60))
            self.history_list.addItem(item)
            self.history_list.setItemWidget(item, row_widget)

    def start_new_chat(self):
        self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.chat_history.clear()
        self.clear_messages()
        self.clear_attachment()
        self.current_sources = []
        self.add_message("Chat history cleared. Start a new conversation!", role="system")
        self.update_vector_map()

    def save_chat(self):
        if not self.current_chat_id:
            self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join("chats", f"Chat_{self.current_chat_id}.json")
        is_new = not os.path.exists(filepath)
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, indent=4)
            if is_new: self.populate_history()
        except Exception as e:
            print(f"Failed to save chat: {e}")

    def load_selected_chat(self, item):
        file_path = item.data(Qt.UserRole)
        if file_path and os.path.exists(file_path):
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    history = json.load(f)
                filename = os.path.basename(file_path)
                self.current_chat_id = filename.replace("Chat_", "").replace(".json", "")
                self.chat_history.clear()
                self.clear_messages()
                self.add_message("📂 Loaded Conversation", role="system")
                for msg in history:
                    role = "user" if msg["role"] == "user" else "bot"
                    self.add_message(msg["content"], role=role)
                    if msg.get("sources"):
                        self.add_citations(msg["sources"], msg.get("is_web", False))
                    self.chat_history.append(msg)
            except Exception as e:
                self.add_message(f"Failed to load chat: {str(e)}", role="system")

    def delete_chat_by_path(self, filepath):
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                fname = os.path.basename(filepath)
                chat_id = fname.replace("Chat_", "").replace(".json", "")
                if self.current_chat_id == chat_id:
                    self.current_chat_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                    self.chat_history.clear()
                    self.clear_messages()
                    self.add_message("Chat deleted. Start a new conversation!", role="system")
                self.populate_history()
            except Exception as e:
                self.add_message(f"Failed to delete chat: {str(e)}", role="system")

    # ---- DOCUMENT MANAGER ---- #
    def select_attachment(self):
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Documents or Audio", "", "Files (*.pdf *.mp3 *.wav *.m4a)"
        )
        if not file_paths: return
        self.attachment_label.setText(f"📄 Indexing {len(file_paths)} file(s)...")
        self.attachment_chip.show()
        self.attachment_button.setEnabled(False)
        self.input_field.setEnabled(False)
        self.send_button.setEnabled(False)
        self.show_loading(f"Processing {len(file_paths)} file(s)...")
        self.proc_thread = MultiProcessingThread(file_paths, self.rag)
        self.proc_thread.progress_signal.connect(lambda msg: self.loading_label.setText(msg))
        self.proc_thread.finished_signal.connect(self.on_files_indexed)
        self.proc_thread.start()

    def on_files_indexed(self, success, msg, doc_names):
        self.hide_loading()
        self.attachment_button.setEnabled(True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)
        if success:
            self.semantic_cache.clear() # Clear cache on new upload
            self.attachment_label.setText(f"✅ {len(doc_names)} file(s) indexed")
            self.add_message(f"✅ {msg}", role="system")
            self.refresh_doc_list()
            self.update_doc_filter()
            self.update_vector_map()
        else:
            self.clear_attachment()
            self.add_message(f"Error: {msg}", role="system")

    def update_doc_filter(self):
        current = self.doc_filter_dropdown.currentText()
        self.doc_filter_dropdown.clear()
        self.doc_filter_dropdown.addItem("All Documents")
        names = self.rag.get_document_names()
        self.doc_filter_dropdown.addItems(names)
        index = self.doc_filter_dropdown.findText(current)
        if index >= 0:
            self.doc_filter_dropdown.setCurrentIndex(index)

    def refresh_doc_list(self):
        self.doc_list.clear()
        names = self.rag.get_document_names()
        total = self.rag.total_chunks()
        self.doc_status_label.setText(f"{len(names)} document(s) • {total} chunks")
        for name in names:
            count = self.rag.get_document_chunk_count(name)
            row_widget = QWidget()
            row_widget.setStyleSheet("background: transparent;")
            row_layout = QHBoxLayout(row_widget)
            row_layout.setContentsMargins(4, 2, 4, 2)
            row_layout.setSpacing(6)
            lbl = QLabel(f"📄 {name} ({count} chunks)")
            lbl.setStyleSheet("color: #f3f4f6; font-size: 12px; background: transparent;")
            lbl.setWordWrap(False)
            row_layout.addWidget(lbl, stretch=1)
            del_btn = QPushButton("🗑")
            del_btn.setFixedSize(26, 26)
            del_btn.setStyleSheet("""
                QPushButton { background: rgba(239,68,68,0.15); border: 1px solid rgba(239,68,68,0.4);
                    border-radius: 6px; font-size: 12px; color: #ef4444; padding: 0px; }
                QPushButton:hover { background: rgba(239,68,68,0.5); border: 1px solid #ef4444; }
            """)
            del_btn.clicked.connect(lambda checked, n=name: self.remove_document(n))
            row_layout.addWidget(del_btn)
            item = QListWidgetItem()
            item.setSizeHint(QSize(280, 60))
            self.doc_list.addItem(item)
            self.doc_list.setItemWidget(item, row_widget)

    def remove_document(self, doc_name):
        self.rag.remove_document(doc_name)
        self.refresh_doc_list()
        self.update_doc_filter()
        self.update_vector_map()
        self.add_message(f"🗑 Removed: {doc_name}", role="system")
        if not self.rag.has_documents():
            self.clear_attachment()

    def clear_attachment(self):
        self.attachment_label.setText("")
        self.attachment_chip.hide()

    # ---- CHAT LOGIC ---- #
    def send_message(self):
        message = self.input_field.text().strip()
        if not message: return
        if self.model_loading:
            self.add_message("⏳ Model is still loading, please wait a moment...", role="system")
            return
        if not self.model_loaded:
            self.add_message("⚠️ Please select and load an LLM model from the sidebar dropdown.", role="system")
            return
        self.add_message(message, role="user")
        self.input_field.clear()
        self.input_field.setEnabled(False)
        self.send_button.hide()
        self.stop_button.show()
        self.pending_message = message
        self.current_sources = []
        self.is_web_source = False

        # FYP Enhancement: Intelligent Fast Routing
        if re.match(r'^[\d\s\+\-\*\/\(\)\.]+$', message) and any(c.isdigit() for c in message):
            try:
                ans = str(eval(message))
                self.show_loading("Calculating (Fast Router) ⚡...")
                QTimer.singleShot(500, lambda: self.finish_fast_path(f"The result is: **{ans}**", [], False))
                return
            except Exception:
                pass
        
        # FYP Enhancement: Semantic Caching (DISABLED)
        
        self.show_loading("Searching & reformulating...")
        self.chat_title.setText("Searching..." if self.web_search_checkbox.isChecked() else "RAG Analysis...")
        doc_filter = self.doc_filter_dropdown.currentText()
        self.search_pipe = SearchPipelineThread(
            self.llm, self.rag, message, list(self.chat_history), self.model_loaded, 
            use_hyde=self.hyde_checkbox.isChecked(),
            doc_filter=doc_filter,
            allow_web=self.web_search_checkbox.isChecked()
        )
        self.search_pipe.result_signal.connect(self.on_search_complete)
        self.search_pipe.start()

    def on_search_complete(self, reformulated_query, results, is_web, query_pca):
        self.chat_title.setText("INTELLISEARCH RAG")
        self.current_sources = results
        self.is_web_source = is_web
        self.show_loading("Generating response...")
        
        # FYP Enhancement: Update vector map with query point
        self.update_vector_map(query_pca)

        context = ""
        if results:
            if is_web:
                context = "\n".join([f"[Web: {r.get('title','')}] {r['text']}" for r in results[:5]])
            else:
                # FYP: Provide parent text context if available (Hierarchical Chunking)
                context = "\n".join([f"[Page {r.get('page','?')} of {r.get('source','doc')}] {r.get('parent_text', r['text'])}" for r in results[:5]])

        messages = []
        if context:
            src_type = "web search results" if is_web else "the uploaded documents"
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Answer based on the following context from {src_type}. "
                           f"If the answer is not in the context, say so. Cite the source page/document when possible.\n\nContext:\n{context}"
            })
        else:
            messages.append({"role": "system", "content": "You are a helpful, intelligent AI assistant."})

        messages.extend(self.chat_history[-10:])
        messages.append({"role": "user", "content": self.pending_message})
        self.chat_history.append({"role": "user", "content": self.pending_message})
        self.save_chat()

        self.current_answer = ""
        self.add_message("", role="bot")
        self.stream_thread = StreamThread(self.llm, messages)
        self.stream_thread.token_signal.connect(self.on_token_received)
        self.stream_thread.finished_signal.connect(self.on_stream_finished)
        self.stream_thread.start()

    def finish_fast_path(self, answer, sources, is_web):
        self.hide_loading()
        self.current_answer = answer
        self.current_sources = sources
        self.is_web_source = is_web
        self.add_message(answer, role="bot")
        self.on_stream_finished()

    def stop_generation(self):
        if hasattr(self, 'stream_thread') and self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping...")

    def on_token_received(self, token):
        self.hide_loading()
        self.current_answer += token
        if self.current_bot_label:
            self.current_bot_label.setText(self.current_answer.replace("\n", "  \n"))
            self.scroll_to_bottom()

    def on_stream_finished(self):
        self.hide_loading()
        if self.current_answer.strip():
            entry = {"role": "assistant", "content": self.current_answer.strip()}
            if self.current_sources:
                entry["sources"] = [{"source": s.get("source",""), "page": s.get("page",""),
                                     "score": s.get("score",0), "title": s.get("title",""),
                                     "url": s.get("url",""), "text": s.get("text","")[:150],
                                     "parent_text": s.get("parent_text","")[:300]}
                                    for s in self.current_sources[:5]]
                entry["is_web"] = self.is_web_source
            self.chat_history.append(entry)
            self.save_chat()
            if self.current_sources:
                self.add_citations(self.current_sources, self.is_web_source)
                
            # Save to semantic cache
            self.semantic_cache.add(
                self.pending_message,
                self.current_answer.strip(),
                self.current_sources,
                self.is_web_source
            )

        self.input_field.setEnabled(True)
        self.stop_button.hide()
        self.stop_button.setEnabled(True)
        self.stop_button.setText("⬛ Stop")
        self.send_button.show()
        self.input_field.setFocus()
        self.current_bot_label = None

    # ---- SOURCE CITATIONS & PDF VIEWER ---- #
    def add_citations(self, sources, is_web=False):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        citation_frame = QFrame()
        citation_frame.setObjectName("citationFrame")
        c_layout = QVBoxLayout(citation_frame)
        c_layout.setContentsMargins(12, 8, 12, 8)
        c_layout.setSpacing(4)

        header = QLabel("🌐 Sources from Web Search:" if is_web else "📄 Sources from Documents (Click to View):")
        header.setStyleSheet("color: #9ca3af; font-size: 13px; font-weight: 600; background: transparent;")
        c_layout.addWidget(header)

        for i, src in enumerate(sources[:5]):
            if is_web:
                url = src.get("url", "")
                txt = f"  {i+1}. <a href='{url}' style='color: #667eea; text-decoration: none;'>{src.get('title', 'Web Result')}</a>"
            else:
                doc = src.get('source', 'Unknown')
                page = src.get('page', 1)
                idx = src.get('index', -1)
                txt = f"  {i+1}. <a href='pdf:{doc}:{page}:{idx}' style='color: #667eea; text-decoration: none;'>{doc} — Page {page}</a> (score: {src.get('score', 0):.3f})"
            lbl = QLabel(txt)
            lbl.setOpenExternalLinks(is_web)
            if not is_web:
                lbl.linkActivated.connect(self.open_pdf_viewer)
            lbl.setStyleSheet("color: #6b7280; font-size: 13px; background: transparent;")
            lbl.setWordWrap(True)
            c_layout.addWidget(lbl)

        citation_frame.setMaximumWidth(800)
        row_layout.addWidget(citation_frame)
        row_layout.addStretch()
        insert_index = self.messages_layout.count() - 1
        self.messages_layout.insertWidget(insert_index, row)
        self.scroll_to_bottom()

    def open_pdf_viewer(self, link):
        if not link.startswith("pdf:"): return
        parts = link.split(":")
        if len(parts) < 3: return
        doc_name = parts[1]
        page_num = int(parts[2])
        chunk_idx = int(parts[3]) if len(parts) > 3 else -1

        # Search for the actual path
        full_path = self.rag.doc_paths.get(doc_name)
        if not full_path and os.path.exists(doc_name):
            full_path = doc_name
        
        # FYP Enhancement: Render PDF page using PyMuPDF
        if full_path:
            try:
                doc = fitz.open(full_path)
                page = doc.load_page(page_num - 1)  # 0-indexed
                
                # FYP Enhancement: XAI Highlighting (Highlight retrieved chunk text)
                if chunk_idx != -1 and chunk_idx < len(self.rag.chunks):
                    chunk_text = self.rag.chunks[chunk_idx]["text"]
                    import string
                    # Highlight long keywords from chunk to show the LLM's focus area
                    words = [w.strip(string.punctuation) for w in chunk_text.split() if len(w) > 5][:15]
                    for w in words:
                        quads = page.search_for(w)
                        for q in quads:
                            annot = page.add_highlight_annot(q)
                            annot.set_colors(stroke=(1, 1, 0)) # Yellow XAI highlighting
                            annot.update()
                            
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2))  # Zoom 2x for clarity
                img = QImage(pix.samples, pix.width, pix.height, pix.stride, QImage.Format_RGB888)
                self.pdf_label.setPixmap(QPixmap.fromImage(img))
                self.pdf_label.resize(img.width(), img.height())
                self.pdf_title.setText(f"📄 {doc_name} (Page {page_num})")
                self.pdf_container.show()
                # Ensure splitter gives space
                sizes = self.main_splitter.sizes()
                if sizes[1] == 0:
                    total = sum(sizes)
                    self.main_splitter.setSizes([int(total*0.6), int(total*0.4)])
            except Exception as e:
                self.show_toast(f"Error opening PDF: {e}")
        else:
            self.show_toast(f"Could not locate original file for {doc_name}")

    # ---- VECTOR SPACE MAP ---- #
    def update_vector_map(self, query_pca=None):
        self.ax.clear()
        self.ax.set_facecolor('#1f2937')
        self.ax.set_title("Semantic Vector Space (PCA)", color='white', pad=10)
        
        if self.rag.pca_coords:
            coords = np.array(self.rag.pca_coords)
            self.ax.scatter(coords[:, 0], coords[:, 1], c='#667eea', alpha=0.6, s=50, label='Document Chunks')
            
            if query_pca and query_pca != [0, 0]:
                self.ax.scatter([query_pca[0]], [query_pca[1]], c='#fbbf24', marker='*', s=200, edgecolors='white', label='Latest Query')
                
            self.ax.legend(loc='best', framealpha=0.2, labelcolor='white')
        else:
            self.ax.text(0.5, 0.5, "Index multiple chunks\nto visualize space.", 
                         ha='center', va='center', color='#9ca3af', transform=self.ax.transAxes)
            
        self.canvas.draw()

    # ---- EVALUATION DASHBOARD ---- #
    def run_evaluation(self):
        if not self.model_loaded:
            self.add_message("⚠️ Load an LLM model first.", role="system")
            return
        if not self.rag.has_documents():
            self.add_message("⚠️ Index at least one document first.", role="system")
            return
        text = self.eval_input.toPlainText().strip()
        if not text:
            self.add_message("⚠️ Enter test questions in the Eval tab.", role="system")
            return
        questions = [q.strip() for q in text.split("\n") if q.strip()]
        if not questions: return

        self.eval_btn.setEnabled(False)
        self.eval_btn.setText("Running...")
        self.eval_progress_label.setVisible(True)
        self.eval_table.setRowCount(0)
        self.eval_avg_label.setText("")

        self.eval_thread = EvalThread(self.llm, self.rag, self.rag.embedder, questions)
        self.eval_thread.progress_signal.connect(lambda cur, tot: self.eval_progress_label.setText(f"Evaluating {cur}/{tot}..."))
        self.eval_thread.result_signal.connect(self.on_eval_results)
        self.eval_thread.finished_signal.connect(self.on_eval_finished)
        self.eval_thread.start()

    def on_eval_results(self, results):
        self.eval_table.setRowCount(len(results))
        f_sum, r_sum, c_sum = 0, 0, 0
        for i, r in enumerate(results):
            q_item = QTableWidgetItem(r["question"][:40])
            q_item.setToolTip(f"Q: {r['question']}\nA: {r['answer'][:200]}")
            self.eval_table.setItem(i, 0, q_item)
            self.eval_table.setItem(i, 1, QTableWidgetItem(f"{r['faithfulness']:.2f}"))
            self.eval_table.setItem(i, 2, QTableWidgetItem(f"{r['relevancy']:.2f}"))
            self.eval_table.setItem(i, 3, QTableWidgetItem(f"{r['context_score']:.3f}"))
            f_sum += r["faithfulness"]; r_sum += r["relevancy"]; c_sum += r["context_score"]
            for col, val in [(1, r["faithfulness"]), (2, r["relevancy"]), (3, r["context_score"])]:
                item = self.eval_table.item(i, col)
                item.setBackground(QColor(16, 185, 129, 40) if val >= 0.7 else QColor(251, 191, 36, 40) if val >= 0.4 else QColor(239, 68, 68, 40))
        n = len(results)
        self.eval_avg_label.setText(f"Averages — Faith: {f_sum/n:.2f} | Relev: {r_sum/n:.2f} | Ctx: {c_sum/n:.3f}")

    def on_eval_finished(self):
        self.eval_btn.setEnabled(True)
        self.eval_btn.setText("▶ Run Evaluation")
        self.eval_progress_label.setVisible(False)

    # ---- UI HELPERS ---- #
    def show_loading(self, message):
        self.loading_label.setText(message)
        self.loading_frame.setVisible(True)

    def hide_loading(self):
        self.loading_frame.setVisible(False)

    def clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            if item.widget(): item.widget().deleteLater()

    def add_message(self, text, role="bot"):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)
        bubble_layout.setContentsMargins(*(16,8,16,8) if role=="system" else (18,14,18,14))
        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        label.setTextFormat(Qt.MarkdownText)
        label.setOpenExternalLinks(True)
        if role == "system": label.setAlignment(Qt.AlignCenter)
        bubble_layout.addWidget(label)
        if role == "user":
            bubble.setObjectName("userBubble")
            bubble.setMaximumWidth(800)
            row_layout.addStretch(); row_layout.addWidget(bubble)
        elif role == "system":
            bubble.setObjectName("systemBubble")
            bubble.setMaximumWidth(600)
            row_layout.addStretch(); row_layout.addWidget(bubble); row_layout.addStretch()
        else:
            bubble.setObjectName("botBubble")
            bubble.setMaximumWidth(1400)
            row_layout.addWidget(bubble); row_layout.addStretch()
            self.current_bot_label = label
        self.messages_layout.insertWidget(self.messages_layout.count() - 1, row)
        self.animate_message(row)
        self.scroll_to_bottom()

    def animate_message(self, widget):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)
        anim = QPropertyAnimation(effect, b"opacity", self)
        anim.setDuration(250); anim.setStartValue(0.0); anim.setEndValue(1.0)
        anim.finished.connect(lambda: widget.setGraphicsEffect(None))
        self.animations.append(anim); anim.start()

    def scroll_to_bottom(self):
        QTimer.singleShot(10, lambda: self.chat_scroll.verticalScrollBar().setValue(self.chat_scroll.verticalScrollBar().maximum()))




if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RagBot()
    window.show()
    sys.exit(app.exec())
