
import sys
from typing import List, Tuple

import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QTextEdit, QLineEdit, QFileDialog, QLabel,
    QSplitter, QMessageBox, QProgressBar
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QTextCursor, QIcon, QPixmap


# ================= PDF PROCESSOR ================= #

class PDFProcessor:
    def extract_text(self, pdf_path: str) -> List[Tuple[int, str]]:
        doc = fitz.open(pdf_path)
        pages = []
        for i, page in enumerate(doc):
            text = page.get_text().strip()
            if text:
                pages.append((i + 1, text))
        doc.close()
        return pages

    def chunk_text(self, pages, chunk_size=300, overlap=80):
        chunks = []
        for page_num, text in pages:
            words = text.split()
            for i in range(0, len(words), chunk_size - overlap):
                chunk = " ".join(words[i:i + chunk_size])
                if chunk:
                    chunks.append({"page": page_num, "text": chunk})
        return chunks


# ================= RAG SYSTEM ================= #

class RAGSystem:
    def __init__(self):
        self.embedder = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.chunks = []

    def build(self, chunks):
        self.chunks = chunks
        texts = [c["text"] for c in chunks]

        emb = self.embedder.encode(texts, show_progress_bar=True)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))

    def search(self, query, k=3):
        q = self.embedder.encode([query])
        q = q / np.linalg.norm(q, axis=1, keepdims=True)

        scores, idx = self.index.search(q.astype("float32"), k)
        results = []
        for i, s in zip(idx[0], scores[0]):
            if i < len(self.chunks):
                r = self.chunks[i].copy()
                r["score"] = float(s)
                results.append(r)
        return results


# ================= LOCAL LLM ================= #

class LocalLLM:
    def __init__(self):
        self.model = None

    def load(self, path):
        self.model = Llama(
            model_path=path,
            n_ctx=2048,
            n_threads=4,
            n_gpu_layers=35,   # RTX 3050 sweet spot
            verbose=False
        )

    def stream(self, prompt, max_tokens=400):
        for chunk in self.model(
            prompt,
            max_tokens=max_tokens,
            temperature=0.6,
            top_p=0.9,
            stream=True
        ):
            token = chunk["choices"][0]["text"]
            if token:
                yield token


# ================= THREADS ================= #

class PDFThread(QThread):
    finished = pyqtSignal(bool, str)

    def __init__(self, paths, rag):
        super().__init__()
        self.paths = paths
        self.rag = rag

    def run(self):
        try:
            proc = PDFProcessor()
            all_chunks = []

            for path in self.paths:
                pages = proc.extract_text(path)
                chunks = proc.chunk_text(pages)
                all_chunks.extend(chunks)

            self.rag.build(all_chunks)
            self.finished.emit(True, f"{len(all_chunks)} chunks indexed from {len(self.paths)} PDFs")

        except Exception as e:
            self.finished.emit(False, str(e))

class StreamThread(QThread):
    token = pyqtSignal(str)
    finished = pyqtSignal()

    def __init__(self, llm, prompt):
        super().__init__()
        self.llm = llm
        self.prompt = prompt

    def run(self):
        for t in self.llm.stream(self.prompt):
            self.token.emit(t)
        self.finished.emit()


# ================= CHAT WIDGET ================= #

class ChatWidget(QWidget):
    def __init__(self, title, placeholder, allow_upload=False, upload_cb=None):
        super().__init__()
        layout = QVBoxLayout()

        # Header
        header = QHBoxLayout()
        label = QLabel(title)
        label.setFont(QFont("Segoe UI", 12, QFont.Bold))
        header.addWidget(label)
        header.addStretch()

        if allow_upload:
            upload_btn = QPushButton("+")
            upload_btn.setFixedSize(60, 60)  # Bigger button
            upload_btn.setFont(QFont("Segoe UI", 18, QFont.Bold))  # Bigger +
            upload_btn.setCursor(Qt.PointingHandCursor)
            upload_btn.setStyleSheet("""
                QPushButton {
                    background-color: #111827;
                    color: white;
                    border: 1px solid #374151;
                    border-radius: 22px;
                }
                QPushButton:hover {
                    background-color: #1f2937;
                }
            """)
            upload_btn.setToolTip("Upload PDF")
            upload_btn.clicked.connect(upload_cb)
            header.addWidget(upload_btn)

        layout.addLayout(header)

        # Chat display
        self.display = QTextEdit()
        self.display.setReadOnly(True)
        layout.addWidget(self.display)

        # Input row
        row = QHBoxLayout()
        self.input = QLineEdit()
        self.input.setPlaceholderText(placeholder)
        self.send = QPushButton("Send")

        row.addWidget(self.input)
        row.addWidget(self.send)

        layout.addLayout(row)
        self.setLayout(layout)

    def add(self, who, text):
        self.display.append(f"<b>{who}:</b> {text}")
        self.display.verticalScrollBar().setValue(
            self.display.verticalScrollBar().maximum()
        )

    def update_last(self, text):
        cursor = self.display.textCursor()
        cursor.movePosition(QTextCursor.End)
        cursor.select(QTextCursor.BlockUnderCursor)
        cursor.removeSelectedText()
        cursor.insertHtml(f"<b>Assistant:</b> {text}")
        self.display.setTextCursor(cursor)

    def get(self):
        t = self.input.text().strip()
        self.input.clear()
        return t

    def clear(self):
        self.display.clear()

# ================= MAIN WINDOW ================= #

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.rag = RAGSystem()
        self.llm = LocalLLM()
        self.pdf_loaded = False
        self.model_loaded = False
        self.current_answer = ""
        self.setWindowIcon(QIcon("rag_logo.ico"))
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("RAG PDF Analyzer")
        self.resize(1400, 800)

        central = QWidget()
        layout = QVBoxLayout()

        # ---------- HEADER ----------
        btn_model = QPushButton("Load LLM")
        btn_clear = QPushButton("Clear")

        btn_model.clicked.connect(self.load_model)
        btn_clear.clicked.connect(self.clear_all)

        top = QHBoxLayout()

        icon = QLabel()
        icon.setPixmap(
            QPixmap("rag_logo.png").scaled(56, 56, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        )

        title = QLabel("RAG PDF Analyzer")
        title.setFont(QFont("Segoe UI", 16, QFont.Bold))

        top.addWidget(icon)
        top.addWidget(title)
        top.addStretch()

        layout.addLayout(top)


        title.setFont(QFont("Segoe UI", 16, QFont.Bold))
        top.addWidget(title)
        top.addStretch()
        top.addWidget(btn_model)
        top.addWidget(btn_clear)

        divider = QLabel()
        divider.setFixedHeight(1)
        divider.setStyleSheet("background-color: #1e293b;")
        layout.addWidget(divider)

        self.status = QLabel("Ready")
        layout.addWidget(self.status)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        # ---------- SPLIT ----------
        split = QSplitter(Qt.Horizontal)

        self.pdf_chat = ChatWidget(
            "PDF Q&A",
            "Ask a question or upload PDFs…",
            allow_upload=True,
            upload_cb=self.open_pdf_dialog
        )

        self.pdf_chat.send.clicked.connect(self.ask_pdf)

        self.help_chat = ChatWidget("AI Assistant", "Ask anything…")
        self.help_chat.send.clicked.connect(self.ask_help)

        split.addWidget(self.pdf_chat)
        split.addWidget(self.help_chat)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 1)

        layout.addWidget(split, 1)

        central.setLayout(layout)
        self.setCentralWidget(central)

        # ---------- STYLES ----------
        self.setStyleSheet("""
            QMainWindow { background-color: #000000; }

            QTextEdit, QLineEdit {
                background-color: #000000;
                color: #ffffff;
                border: 1px solid #2a2a2a;
                border-radius: 8px;
                padding: 10px;
                font-size: 17px;
                font-family: "Segoe UI";
            }

            QPushButton {
                background-color: #1a1a1a;
                color: white;
                border-radius: 8px;
                padding: 8px 16px;
            }

            QPushButton:hover {
                background-color: #2a2a2a;
            }

            QLabel {
                color: white;
            }

            QSplitter::handle {
                background-color: #1a1a1a;
            }

            QProgressBar {
                background-color: #111;
                color: white;
                border-radius: 6px;
            }

            QProgressBar::chunk {
                background-color: #ffffff;
            }
        """)


    # -------- ACTIONS -------- #

    def open_pdf_dialog(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select PDF files",
            "",
            "PDF Files (*.pdf)"
        )

        if not paths:
            return

        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        # Process PDFs in background
        self.thread = PDFThread(paths, self.rag)
        self.thread.finished.connect(self.pdf_done)
        self.thread.start()

    def load_pdf(self):
        path, _ = QFileDialog.getOpenFileName(self, "PDF", "", "*.pdf")
        if not path:
            return
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.thread = PDFThread(path, self.rag)
        self.thread.finished.connect(self.pdf_done)
        self.open_pdf_dialog()
        self.thread.start()

    def pdf_done(self, ok, msg):
        self.progress.setVisible(False)
        if ok:
            self.pdf_loaded = True
            self.status.setText(msg)
            self.pdf_chat.add("System", "PDF ready ✅")
        else:
            QMessageBox.critical(self, "Error", msg)

    def load_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "GGUF Model", "", "*.gguf")
        if not path:
            return
        self.llm.load(path)
        self.model_loaded = True
        self.status.setText("Model loaded")

    def ask_pdf(self):
        if not (self.pdf_loaded and self.model_loaded):
            QMessageBox.warning(self, "Warning", "Load PDF and model first")
            return

        q = self.pdf_chat.get()
        if not q:
            return

        self.pdf_chat.add("You", q)
        self.pdf_chat.add("Assistant", "")

        results = self.rag.search(q)
        context = "\n".join([f"(Page {r['page']}) {r['text']}" for r in results])

        prompt = f"""
Answer ONLY from the context.
If not found say: Not found in the document.

Context:
{context}

Question: {q}
Answer:
"""

        self.current_answer = ""
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self.pdf_stream = StreamThread(self.llm, prompt)
        self.pdf_stream.token.connect(self.on_pdf_token)
        self.pdf_stream.finished.connect(self.finish_pdf)
        self.pdf_stream.start()

    def on_pdf_token(self, token):
        self.current_answer += token
        self.pdf_chat.update_last(self.current_answer)

    def finish_pdf(self):
        self.progress.setVisible(False)
        self.pdf_chat.add("System", "📌 Answer completed")

    def ask_help(self):
        if not self.model_loaded:
            return
        q = self.help_chat.get()
        if not q:
            return

        self.help_chat.add("You", q)
        self.help_chat.add("Assistant", "")

        self.current_answer = ""
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)

        self.help_stream = StreamThread(self.llm, q)
        self.help_stream.token.connect(self.on_help_token)
        self.help_stream.finished.connect(lambda: self.progress.setVisible(False))
        self.help_stream.start()

    def on_help_token(self, token):
        self.current_answer += token
        self.help_chat.update_last(self.current_answer)

    def clear_all(self):
        self.pdf_chat.clear()
        self.help_chat.clear()


# ================= ENTRY ================= #

def main():
    app = QApplication(sys.argv)

    app.setWindowIcon(QIcon("rag_logo.ico"))  # 👈 TASKBAR + ALT+TAB
    
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
