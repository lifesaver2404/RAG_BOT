import os
import sys
import json
from datetime import datetime
from typing import List, Tuple

import fitz
import faiss
import numpy as np

from sentence_transformers import SentenceTransformer
from llama_cpp import Llama

from PySide6.QtCore import Qt, QEasingCurve, QPropertyAnimation, QTimer, QThread, Signal
from PySide6.QtGui import QFont
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QGraphicsOpacityEffect,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)


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

        emb = self.embedder.encode(texts, show_progress_bar=False)
        emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)

        self.index = faiss.IndexFlatIP(emb.shape[1])
        self.index.add(emb.astype("float32"))

    def search(self, query, k=3):
        if self.index is None:
            return []

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
            n_ctx=4096,
            n_threads=4,
            n_gpu_layers=35,
            verbose=False
        )

    def stream(self, messages, max_tokens=1024):
        for chunk in self.model.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.6,
                top_p=0.9,
                stream=True
        ):
            delta = chunk["choices"][0]["delta"]
            if "content" in delta:
                yield delta["content"]


# ================= THREADS ================= #

class PDFThread(QThread):
    finished_signal = Signal(bool, str)

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
            self.finished_signal.emit(True, f"{len(all_chunks)} chunks indexed successfully.")
        except Exception as e:
            self.finished_signal.emit(False, str(e))


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


# ================= MAIN UI ================= #

class RagBot(QMainWindow):
    def __init__(self):
        super().__init__()

        # Core Backend
        self.rag = RAGSystem()
        self.llm = LocalLLM()
        self.model_loaded = False
        self.pdf_loaded = False

        # UI State
        self.attached_file = None
        self.current_chat_id = ""
        self.animations = []
        self.current_bot_label = None
        self.current_answer = ""
        self.chat_history = []

        self.setWindowTitle("RAGBOT AI")
        self.resize(1200, 800)

        # Ensure the directories exist
        if not os.path.exists("models"):
            os.makedirs("models")
        if not os.path.exists("chats"):
            os.makedirs("chats")

        self.setStyleSheet(
            """
            QMainWindow {
                background-color: #111213;
            }
            QFrame#card {
                background-color: #1B1E23;
                border-radius: 20px;
                border: 1px solid #2A2F36;
            }
            QScrollArea#chatScroll {
                border: none;
                background: transparent;
            }
            QWidget#messagesContainer {
                background-color: #1F2329;
                border-radius: 12px;
            }
            QFrame#userBubble {
                background-color: #2E7D32;
                border-radius: 12px;
            }
            QFrame#botBubble {
                background-color: #2A2F36;
                border-radius: 12px;
            }
            QFrame#systemBubble {
                background-color: transparent;
                border: 1px solid #2C313A;
                border-radius: 14px;
            }
            QFrame#userBubble QLabel,
            QFrame#botBubble QLabel {
                color: #E5E7EB;
                font-size: 15px; 
            }
            QFrame#systemBubble QLabel {
                color: #6B7280;
                font-size: 12px;
                font-weight: 500;
            }
            QLineEdit {
                background-color: #1F2329;
                border: 1px solid #2C313A;
                border-radius: 12px;
                padding: 12px;
                color: white;
                font-size: 14px;
            }
            QPushButton {
                background-color: #2E7D32;
                border-radius: 10px;
                padding: 10px 20px;
                color: white;
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #37993C;
            }
            QPushButton:disabled {
                background-color: #1A471C;
                color: #888888;
            }
            QPushButton#attachButton {
                background-color: #2C313A;
                padding: 10px 16px;
            }
            QPushButton#attachButton:hover {
                background-color: #3A414D;
            }
            QPushButton#stopButton {
                background-color: #991B1B; 
            }
            QPushButton#stopButton:hover {
                background-color: #B91C1C;
            }
            QFrame#attachmentChip {
                background-color: #262C34;
                border: 1px solid #3A414D;
                border-radius: 14px;
            }
            QLabel#attachmentLabel {
                color: #D1D5DB;
            }
            QPushButton#removeAttachmentButton {
                background-color: transparent;
                color: #9CA3AF;
                padding: 0px;
                border-radius: 0px;
            }
            QPushButton#removeAttachmentButton:hover {
                color: #FFFFFF;
                background-color: transparent;
            }
            QComboBox {
                background-color: #1F2329;
                border: 1px solid #2C313A;
                border-radius: 8px;
                padding: 8px;
                color: white;
                font-weight: bold;
            }
            QComboBox::drop-down {
                border: none;
            }
            QComboBox QAbstractItemView {
                background-color: #1F2329;
                color: white;
                selection-background-color: #2E7D32;
                border-radius: 8px;
            }
            QPushButton#sidebarBtn {
                background-color: transparent;
                text-align: left;
                padding: 10px;
                border-radius: 8px;
                color: #E5E7EB;
                font-size: 15px;
                font-weight: bold;
            }
            QPushButton#sidebarBtn:hover {
                background-color: #2C313A;
                color: white;
            }

            /* --- NEW: Delete Button Styling --- */
            QPushButton#deleteBtn {
                background-color: transparent;
                text-align: left;
                padding: 10px;
                border-radius: 8px;
                color: #EF4444; /* Bright Red */
                font-size: 14px;
                font-weight: bold;
            }
            QPushButton#deleteBtn:hover {
                background-color: #7F1D1D; /* Dark Red Background on Hover */
                color: white;
            }

            QListWidget#historyList {
                background-color: transparent;
                border: none;
                color: #D1D5DB;
                font-size: 14px;
                font-weight: 500;
            }
            QListWidget#historyList::item {
                padding: 10px;
                border-radius: 8px;
                margin-bottom: 2px;
            }
            QListWidget#historyList::item:hover {
                background-color: #2C313A;
                color: white;
            }
            QListWidget#historyList::item:selected {
                background-color: #2E7D32;
                color: white;
                font-weight: bold;
            }
            """
        )

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # ---------- SIDEBAR ----------
        self.sidebar_container = QWidget()
        self.sidebar_container.setFixedWidth(200)
        sidebar_layout = QVBoxLayout(self.sidebar_container)
        sidebar_layout.setContentsMargins(0, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        self.new_chat_btn = QPushButton("+ New Chat")
        self.new_chat_btn.setObjectName("sidebarBtn")
        self.new_chat_btn.clicked.connect(self.start_new_chat)
        sidebar_layout.addWidget(self.new_chat_btn)

        model_label = QLabel("LLM Model:")
        model_label.setStyleSheet("color: #E5E7EB; font-size: 13px; font-weight: bold; margin-top: 10px;")
        sidebar_layout.addWidget(model_label)

        self.model_dropdown = QComboBox()
        self.populate_models()
        self.model_dropdown.currentIndexChanged.connect(self.on_model_selected)
        sidebar_layout.addWidget(self.model_dropdown)

        history_label = QLabel("Recent Chats:")
        history_label.setStyleSheet("color: #E5E7EB; font-size: 13px; font-weight: bold; margin-top: 15px;")
        sidebar_layout.addWidget(history_label)

        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.itemClicked.connect(self.load_selected_chat)
        self.populate_history()
        sidebar_layout.addWidget(self.history_list)

        sidebar_layout.addStretch()  # Pushes the delete button to the very bottom

        # --- NEW: Delete Chat Button ---
        self.delete_chat_btn = QPushButton("🗑 Delete Current Chat")
        self.delete_chat_btn.setObjectName("deleteBtn")
        self.delete_chat_btn.clicked.connect(self.delete_current_chat)
        sidebar_layout.addWidget(self.delete_chat_btn)

        # ---------- MAIN CARD ----------
        self.card = QFrame()
        self.card.setObjectName("card")
        card_layout = QVBoxLayout(self.card)

        title = QLabel("RAGBOT AI")
        title.setFont(QFont("Inter", 18, QFont.Bold))
        title.setStyleSheet("color: #E6E8EB;")
        card_layout.addWidget(title)

        self.session_label = QLabel("Status: Waiting for LLM Model...")
        self.session_label.setStyleSheet("color: #9CA3AF;")
        card_layout.addWidget(self.session_label)

        # Chat Area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setObjectName("chatScroll")
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setFrameShape(QFrame.NoFrame)

        self.messages_container = QWidget()
        self.messages_container.setObjectName("messagesContainer")
        self.messages_layout = QVBoxLayout(self.messages_container)
        self.messages_layout.setContentsMargins(15, 15, 15, 15)
        self.messages_layout.setSpacing(12)
        self.messages_layout.addStretch()
        self.chat_scroll.setWidget(self.messages_container)
        card_layout.addWidget(self.chat_scroll)

        # Attachment Chip
        self.attachment_chip = QFrame()
        self.attachment_chip.setObjectName("attachmentChip")
        chip_layout = QHBoxLayout(self.attachment_chip)
        chip_layout.setContentsMargins(10, 4, 10, 4)
        chip_layout.setSpacing(6)

        self.attachment_label = QLabel("")
        self.attachment_label.setObjectName("attachmentLabel")
        chip_layout.addWidget(self.attachment_label)

        self.remove_attachment_button = QPushButton("x")
        self.remove_attachment_button.setObjectName("removeAttachmentButton")
        self.remove_attachment_button.setFixedWidth(20)
        self.remove_attachment_button.clicked.connect(self.clear_attachment)
        chip_layout.addWidget(self.remove_attachment_button)

        self.attachment_chip.hide()
        card_layout.addWidget(self.attachment_chip)

        # Input Area
        input_layout = QHBoxLayout()
        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type your message here...")
        self.input_field.returnPressed.connect(self.send_message)

        self.attachment_button = QPushButton("Attach PDF")
        self.attachment_button.setObjectName("attachButton")
        self.attachment_button.setToolTip("Attach and index a PDF file")
        self.attachment_button.clicked.connect(self.select_attachment)

        self.send_button = QPushButton("Send")
        self.send_button.clicked.connect(self.send_message)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.clicked.connect(self.stop_generation)
        self.stop_button.hide()

        input_layout.addWidget(self.attachment_button)
        input_layout.addWidget(self.input_field)
        input_layout.addWidget(self.send_button)
        input_layout.addWidget(self.stop_button)
        card_layout.addLayout(input_layout)

        main_layout.addWidget(self.sidebar_container)
        main_layout.addWidget(self.card)

        self.start_new_chat()

    # -------- UI INTERACTIONS -------- #

    def start_new_chat(self):
        self.current_chat_id = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.clear_messages()
        self.input_field.clear()
        self.chat_history.clear()
        self.history_list.clearSelection()
        self.add_message("Welcome to RAGBOT! Please select an LLM model from the dropdown to begin.", role="system")

    def populate_models(self):
        self.model_dropdown.blockSignals(True)
        self.model_dropdown.clear()
        self.model_dropdown.addItem("-- Select Model --")

        for file in os.listdir("models"):
            if file.endswith(".gguf"):
                self.model_dropdown.addItem(file)
        self.model_dropdown.blockSignals(False)

    def on_model_selected(self, index):
        if index == 0:
            return

        model_name = self.model_dropdown.currentText()
        model_path = os.path.join("models", model_name)
        self.load_model(model_path)

    def load_model(self, path):
        self.add_message(f"Loading model: {os.path.basename(path)}...", role="system")
        QApplication.processEvents()

        try:
            self.llm.load(path)
            self.model_loaded = True
            self.session_label.setText(f"Model: {os.path.basename(path)}")
            self.add_message("Model loaded successfully! You can now chat or attach PDFs.", role="system")
        except Exception as e:
            self.add_message(f"Failed to load model: {str(e)}", role="system")

    # -------- CHAT SAVING / LOADING / DELETING -------- #

    def populate_history(self):
        self.history_list.clear()
        if not os.path.exists("chats"):
            return

        chat_files = [f for f in os.listdir("chats") if f.endswith(".json")]
        chat_files.sort(key=lambda x: os.path.getmtime(os.path.join("chats", x)), reverse=True)

        for file in chat_files:
            filepath = os.path.join("chats", file)
            title = file.replace("Chat_", "").replace(".json", "")

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    history = json.load(f)
                    for msg in history:
                        if msg["role"] == "user":
                            title = msg["content"][:25] + ("..." if len(msg["content"]) > 25 else "")
                            break
            except Exception:
                pass

            item = QListWidgetItem(title)
            item.setData(Qt.UserRole, filepath)
            self.history_list.addItem(item)

    def save_chat(self):
        if not self.chat_history:
            return

        filepath = os.path.join("chats", f"Chat_{self.current_chat_id}.json")
        is_new_file = not os.path.exists(filepath)

        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(self.chat_history, f, indent=4)

            if is_new_file:
                self.populate_history()

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
                self.add_message("Loaded Conversation", role="system")

                for msg in history:
                    role = "user" if msg["role"] == "user" else "bot"
                    self.add_message(msg["content"], role=role)
                    self.chat_history.append(msg)

            except Exception as e:
                self.add_message(f"Failed to load chat: {str(e)}", role="system")

    def delete_current_chat(self):
        """Deletes the JSON file for the current chat and resets the view."""
        if not self.current_chat_id:
            return

        filepath = os.path.join("chats", f"Chat_{self.current_chat_id}.json")
        if os.path.exists(filepath):
            try:
                os.remove(filepath)
                # Refresh the sidebar and clear the screen
                self.populate_history()
                self.start_new_chat()
                self.add_message("Chat history deleted.", role="system")
            except Exception as e:
                self.add_message(f"Failed to delete chat: {str(e)}", role="system")
        else:
            # If the file doesn't exist yet (e.g., just clicked New Chat)
            self.start_new_chat()

    # -------- RAG LOGIC -------- #

    def select_attachment(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Select PDF Document", "", "PDF Files (*.pdf)")
        if file_path:
            self.attached_file = file_path
            self.attachment_label.setText(f"Indexing: {os.path.basename(file_path)}...")
            self.attachment_chip.show()

            self.attachment_button.setEnabled(False)
            self.input_field.setEnabled(False)
            self.send_button.setEnabled(False)

            self.pdf_thread = PDFThread([file_path], self.rag)
            self.pdf_thread.finished_signal.connect(self.on_pdf_indexed)
            self.pdf_thread.start()

    def on_pdf_indexed(self, success, msg):
        self.attachment_button.setEnabled(True)
        self.input_field.setEnabled(True)
        self.send_button.setEnabled(True)

        if success:
            self.pdf_loaded = True
            self.attachment_label.setText(os.path.basename(self.attached_file))
            self.add_message(f"PDF Indexed: {msg}", role="system")
        else:
            self.clear_attachment()
            self.add_message(f"Error reading PDF: {msg}", role="system")

    def clear_attachment(self):
        self.attached_file = None
        self.pdf_loaded = False
        self.attachment_label.setText("")
        self.attachment_chip.hide()

    # -------- CHAT LOGIC -------- #

    def send_message(self):
        message = self.input_field.text().strip()
        if not message:
            return

        if not self.model_loaded:
            self.add_message("Please select an LLM model from the sidebar first.", role="system")
            return

        self.add_message(message, role="user")
        self.input_field.clear()

        self.input_field.setEnabled(False)
        self.send_button.hide()
        self.stop_button.show()

        context = ""
        if self.pdf_loaded:
            results = self.rag.search(message)
            context = "\n".join([f"(Page {r['page']}) {r['text']}" for r in results])

        messages = []
        if context:
            messages.append({
                "role": "system",
                "content": f"You are a helpful assistant. Answer ONLY based on the context provided. If the answer is not in the context, say so.\n\nContext:\n{context}"
            })
        else:
            messages.append({
                "role": "system",
                "content": "You are a helpful, intelligent AI assistant."
            })

        messages.extend(self.chat_history[-10:])
        messages.append({"role": "user", "content": message})

        self.chat_history.append({"role": "user", "content": message})
        self.save_chat()

        self.current_answer = ""
        self.add_message("", role="bot")

        self.stream_thread = StreamThread(self.llm, messages)
        self.stream_thread.token_signal.connect(self.on_token_received)
        self.stream_thread.finished_signal.connect(self.on_stream_finished)
        self.stream_thread.start()

    def stop_generation(self):
        if hasattr(self, 'stream_thread') and self.stream_thread.isRunning():
            self.stream_thread.stop()
            self.stop_button.setEnabled(False)
            self.stop_button.setText("Stopping...")

    def on_token_received(self, token):
        self.current_answer += token
        if self.current_bot_label:
            formatted_text = self.current_answer.replace("\n", "  \n")
            self.current_bot_label.setText(formatted_text)
            self.scroll_to_bottom()

    def on_stream_finished(self):
        if self.current_answer.strip():
            self.chat_history.append({"role": "assistant", "content": self.current_answer.strip()})
            self.save_chat()

        self.input_field.setEnabled(True)
        self.stop_button.hide()
        self.stop_button.setEnabled(True)
        self.stop_button.setText("Stop")
        self.send_button.show()

        self.input_field.setFocus()
        self.current_bot_label = None

    # -------- UI HELPERS -------- #

    def clear_messages(self):
        while self.messages_layout.count() > 1:
            item = self.messages_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

    def add_message(self, text, role="bot"):
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)

        bubble = QFrame()
        bubble_layout = QVBoxLayout(bubble)

        if role == "system":
            bubble_layout.setContentsMargins(16, 6, 16, 6)
        else:
            bubble_layout.setContentsMargins(16, 12, 16, 12)

        label = QLabel(text)
        label.setWordWrap(True)
        label.setTextInteractionFlags(Qt.TextSelectableByMouse)

        label.setTextFormat(Qt.MarkdownText)
        label.setOpenExternalLinks(True)

        if role == "system":
            label.setAlignment(Qt.AlignCenter)

        bubble_layout.addWidget(label)

        if role == "user":
            bubble.setObjectName("userBubble")
            bubble.setMaximumWidth(800)
            row_layout.addStretch()
            row_layout.addWidget(bubble)
        elif role == "system":
            bubble.setObjectName("systemBubble")
            bubble.setMaximumWidth(600)
            row_layout.addStretch()
            row_layout.addWidget(bubble)
            row_layout.addStretch()
        else:
            bubble.setObjectName("botBubble")
            bubble.setMaximumWidth(1400)
            row_layout.addWidget(bubble)
            row_layout.addStretch()
            self.current_bot_label = label

        insert_index = self.messages_layout.count() - 1
        self.messages_layout.insertWidget(insert_index, row)
        self.animate_message(row)
        self.scroll_to_bottom()

    def animate_message(self, widget):
        effect = QGraphicsOpacityEffect(widget)
        widget.setGraphicsEffect(effect)

        animation = QPropertyAnimation(effect, b"opacity", self)
        animation.setDuration(200)
        animation.setStartValue(0.0)
        animation.setEndValue(1.0)
        animation.setEasingCurve(QEasingCurve.OutCubic)

        def cleanup():
            widget.setGraphicsEffect(None)
            if animation in self.animations:
                self.animations.remove(animation)

        animation.finished.connect(cleanup)
        self.animations.append(animation)
        animation.start()

    def scroll_to_bottom(self):
        def _scroll():
            scrollbar = self.chat_scroll.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

        QTimer.singleShot(10, _scroll)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = RagBot()
    window.show()
    sys.exit(app.exec())