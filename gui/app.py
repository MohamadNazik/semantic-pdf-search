"""
gui/app.py
===========
Module 10 — Desktop GUI (Tkinter)

SemanticSearchApp provides a research-oriented desktop interface with:
  * PDF folder selection
  * Embedding model selection (MiniLM / MPNet / SciBERT)
  * Index build (with progress feedback)
  * Natural-language and keyword search
  * Ranked result list showing filename, score, and snippet
  * Double-click to open the source PDF
"""

import logging
import os
import subprocess
import sys
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, scrolledtext, ttk
from typing import List, Optional

# Ensure project root is on sys.path when this file is run directly
_HERE = Path(__file__).resolve().parent.parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import config
from evaluation.logger       import ExperimentLogger
from extraction.extractor    import DocumentRegistry, HybridExtractor
from indexing.faiss_index    import FAISSIndex, ChunkMeta
from preprocessing.preprocessor import TextPreprocessor
from embeddings.embedder     import EmbeddingEngine
from search.semantic_search  import SemanticSearch
from search.keyword_search   import KeywordSearch

logger = logging.getLogger(__name__)


# ── Colour palette ────────────────────────────────────────────────────────────
COLORS = {
    "bg":          "#1e1e2e",
    "surface":     "#313244",
    "accent":      "#89b4fa",
    "accent2":     "#cba6f7",
    "text":        "#cdd6f4",
    "subtext":     "#a6adc8",
    "green":       "#a6e3a1",
    "red":         "#f38ba8",
    "yellow":      "#f9e2af",
    "border":      "#45475a",
}

FONT_BODY  = ("Segoe UI", 10)
FONT_BOLD  = ("Segoe UI", 10, "bold")
FONT_MONO  = ("Consolas", 9)
FONT_TITLE = ("Segoe UI", 14, "bold")


class SemanticSearchApp:
    """Main Tkinter application for the semantic search prototype."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Semantic PDF Search — Research Prototype")
        self.root.geometry("1050x720")
        self.root.configure(bg=COLORS["bg"])
        self.root.resizable(True, True)

        # ── runtime state ─────────────────────────────────────────────────────
        self._pdf_folder:    Optional[Path] = None
        self._registry:      Optional[DocumentRegistry] = None
        self._semantic:      Optional[SemanticSearch] = None
        self._keyword:       Optional[KeywordSearch]  = None
        self._exp_logger     = ExperimentLogger(config.LOGS_DIR)
        self._current_run_id: Optional[str] = None
        self._result_rows:   List[dict] = []   # currently displayed results

        # ── build UI ──────────────────────────────────────────────────────────
        self._build_ui()
        self._refresh_status("Ready. Select a PDF folder to begin.")

    # ─────────────────────────────────────────────────────────────────────────
    # UI construction
    # ─────────────────────────────────────────────────────────────────────────

    def _build_ui(self):
        # ── header ────────────────────────────────────────────────────────────
        header = tk.Frame(self.root, bg=COLORS["surface"], height=56)
        header.pack(fill="x", side="top")
        tk.Label(
            header,
            text="Semantic PDF Search",
            font=FONT_TITLE,
            bg=COLORS["surface"],
            fg=COLORS["accent"],
        ).pack(side="left", padx=18, pady=10)

        tk.Label(
            header,
            text="Research Prototype",
            font=FONT_BODY,
            bg=COLORS["surface"],
            fg=COLORS["subtext"],
        ).pack(side="left", pady=10)

        # ── main body (left panel + right panel) ──────────────────────────────
        body = tk.Frame(self.root, bg=COLORS["bg"])
        body.pack(fill="both", expand=True, padx=0, pady=0)

        left  = tk.Frame(body, bg=COLORS["bg"], width=280)
        left.pack(side="left", fill="y", padx=(10, 0), pady=8)
        left.pack_propagate(False)

        right = tk.Frame(body, bg=COLORS["bg"])
        right.pack(side="left", fill="both", expand=True, padx=8, pady=8)

        self._build_left_panel(left)
        self._build_right_panel(right)

        # ── status bar ────────────────────────────────────────────────────────
        self._status_var = tk.StringVar(value="Ready")
        status_bar = tk.Label(
            self.root,
            textvariable=self._status_var,
            font=FONT_MONO,
            bg=COLORS["surface"],
            fg=COLORS["subtext"],
            anchor="w",
            padx=10,
        )
        status_bar.pack(fill="x", side="bottom", ipady=3)

    # ── left panel ────────────────────────────────────────────────────────────

    def _build_left_panel(self, parent):
        def section(label):
            f = tk.LabelFrame(
                parent, text=label, font=FONT_BOLD,
                bg=COLORS["bg"], fg=COLORS["accent"],
                bd=1, relief="groove",
            )
            f.pack(fill="x", pady=(0, 8))
            return f

        # ── Folder selection ──────────────────────────────────────────────────
        sf = section("1. PDF Folder")
        self._folder_var = tk.StringVar(value="(none)")
        tk.Label(sf, textvariable=self._folder_var, font=FONT_MONO,
                 bg=COLORS["bg"], fg=COLORS["text"],
                 wraplength=240, justify="left").pack(fill="x", padx=6, pady=(2, 0))
        self._btn_browse = tk.Button(
            sf, text="Browse…", font=FONT_BODY,
            bg=COLORS["accent"], fg=COLORS["bg"], relief="flat",
            command=self._browse_folder,
        )
        self._btn_browse.pack(fill="x", padx=6, pady=5)

        # ── Model selection ───────────────────────────────────────────────────
        mf = section("2. Embedding Model")
        self._model_var = tk.StringVar(value="minilm")
        for key, info in config.EMBEDDING_MODELS.items():
            short = key.upper()
            tk.Radiobutton(
                mf, text=f"{short}  —  {info['description']}",
                variable=self._model_var, value=key,
                font=FONT_BODY,
                bg=COLORS["bg"], fg=COLORS["text"],
                selectcolor=COLORS["surface"],
                activebackground=COLORS["bg"],
                command=self._on_model_change,
            ).pack(anchor="w", padx=6, pady=1)

        # ── Chunking settings ─────────────────────────────────────────────────
        cf = section("3. Chunking")
        self._chunking_var = tk.BooleanVar(value=config.USE_CHUNKING)
        tk.Checkbutton(
            cf, text="Enable chunking",
            variable=self._chunking_var,
            font=FONT_BODY,
            bg=COLORS["bg"], fg=COLORS["text"],
            selectcolor=COLORS["surface"],
            activebackground=COLORS["bg"],
        ).pack(anchor="w", padx=6)

        row = tk.Frame(cf, bg=COLORS["bg"])
        row.pack(fill="x", padx=6, pady=3)
        tk.Label(row, text="Chunk size (words):", font=FONT_BODY,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left")
        self._chunk_size_var = tk.IntVar(value=config.CHUNK_SIZE)
        tk.Spinbox(row, from_=50, to=1000, width=6,
                   textvariable=self._chunk_size_var,
                   bg=COLORS["surface"], fg=COLORS["text"]).pack(side="right")

        row2 = tk.Frame(cf, bg=COLORS["bg"])
        row2.pack(fill="x", padx=6, pady=3)
        tk.Label(row2, text="Overlap (words):", font=FONT_BODY,
                 bg=COLORS["bg"], fg=COLORS["text"]).pack(side="left")
        self._chunk_overlap_var = tk.IntVar(value=config.CHUNK_OVERLAP)
        tk.Spinbox(row2, from_=0, to=200, width=6,
                   textvariable=self._chunk_overlap_var,
                   bg=COLORS["surface"], fg=COLORS["text"]).pack(side="right")

        # ── Build index ───────────────────────────────────────────────────────
        bf = section("4. Build Index")
        self._progress = ttk.Progressbar(bf, mode="indeterminate", length=240)
        self._progress.pack(fill="x", padx=6, pady=(4, 0))
        self._btn_build = tk.Button(
            bf, text="Build Index", font=FONT_BOLD,
            bg=COLORS["green"], fg=COLORS["bg"], relief="flat",
            command=self._build_index_async,
        )
        self._btn_build.pack(fill="x", padx=6, pady=6)

        # ── Index info ────────────────────────────────────────────────────────
        inf = section("Index Info")
        self._index_info_var = tk.StringVar(value="No index loaded.")
        tk.Label(
            inf, textvariable=self._index_info_var,
            font=FONT_MONO, bg=COLORS["bg"], fg=COLORS["subtext"],
            justify="left", wraplength=250,
        ).pack(fill="x", padx=6, pady=4)

    # ── right panel ───────────────────────────────────────────────────────────

    def _build_right_panel(self, parent):
        # ── search bar ────────────────────────────────────────────────────────
        srow = tk.Frame(parent, bg=COLORS["bg"])
        srow.pack(fill="x", pady=(0, 6))

        self._query_var = tk.StringVar()
        entry = tk.Entry(
            srow, textvariable=self._query_var, font=FONT_BODY,
            bg=COLORS["surface"], fg=COLORS["text"],
            insertbackground=COLORS["text"],
            relief="flat", bd=4,
        )
        entry.pack(side="left", fill="x", expand=True, ipady=5)
        entry.bind("<Return>", lambda _e: self._run_search())

        self._search_type_var = tk.StringVar(value="semantic")
        tk.Radiobutton(
            srow, text="Semantic", variable=self._search_type_var,
            value="semantic", font=FONT_BODY,
            bg=COLORS["bg"], fg=COLORS["text"],
            selectcolor=COLORS["surface"],
            activebackground=COLORS["bg"],
        ).pack(side="left", padx=(8, 2))
        tk.Radiobutton(
            srow, text="Keyword", variable=self._search_type_var,
            value="keyword", font=FONT_BODY,
            bg=COLORS["bg"], fg=COLORS["text"],
            selectcolor=COLORS["surface"],
            activebackground=COLORS["bg"],
        ).pack(side="left", padx=2)

        self._topk_var = tk.IntVar(value=config.DEFAULT_TOP_K)
        tk.Label(srow, text="Top-K:", font=FONT_BODY,
                 bg=COLORS["bg"], fg=COLORS["subtext"]).pack(side="left", padx=(8, 2))
        tk.Spinbox(srow, from_=1, to=50, width=4,
                   textvariable=self._topk_var,
                   bg=COLORS["surface"], fg=COLORS["text"]).pack(side="left")

        btn_search = tk.Button(
            srow, text="Search", font=FONT_BOLD,
            bg=COLORS["accent2"], fg=COLORS["bg"], relief="flat",
            command=self._run_search,
        )
        btn_search.pack(side="left", padx=(8, 0), ipadx=8, ipady=2)

        # ── results table ─────────────────────────────────────────────────────
        cols = ("Rank", "Score", "Filename", "Snippet")
        self._tree = ttk.Treeview(
            parent, columns=cols, show="headings", height=18,
        )

        style = ttk.Style()
        style.theme_use("clam")
        style.configure(
            "Treeview",
            background=COLORS["surface"],
            foreground=COLORS["text"],
            fieldbackground=COLORS["surface"],
            rowheight=28,
            font=FONT_BODY,
        )
        style.configure("Treeview.Heading",
                         background=COLORS["border"],
                         foreground=COLORS["accent"],
                         font=FONT_BOLD)
        style.map("Treeview",
                  background=[("selected", COLORS["accent"])],
                  foreground=[("selected", COLORS["bg"])])

        for col, w in zip(cols, [50, 70, 240, 550]):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, stretch=(col == "Snippet"))

        vsb = ttk.Scrollbar(parent, orient="vertical", command=self._tree.yview)
        self._tree.configure(yscrollcommand=vsb.set)

        self._tree.pack(side="left", fill="both", expand=True)
        vsb.pack(side="left", fill="y")

        self._tree.bind("<Double-1>", self._on_result_double_click)

        # ── snippet detail ────────────────────────────────────────────────────
        tk.Label(parent, text="Snippet detail (double-click a result):",
                 font=FONT_BODY, bg=COLORS["bg"], fg=COLORS["subtext"]
                 ).pack(anchor="w", pady=(6, 0))
        self._detail_text = scrolledtext.ScrolledText(
            parent, height=5, font=FONT_MONO,
            bg=COLORS["surface"], fg=COLORS["text"],
            relief="flat", wrap="word",
        )
        self._detail_text.pack(fill="x", pady=(2, 0))

    # ─────────────────────────────────────────────────────────────────────────
    # Event handlers
    # ─────────────────────────────────────────────────────────────────────────

    def _browse_folder(self):
        folder = filedialog.askdirectory(title="Select PDF Folder")
        if folder:
            self._pdf_folder = Path(folder)
            self._folder_var.set(str(self._pdf_folder))
            self._refresh_status(f"Folder selected: {self._pdf_folder}")

    def _on_model_change(self):
        self._semantic = None   # force re-load on next search
        self._refresh_status(
            f"Model changed to {self._model_var.get()}. "
            "Rebuild the index or load an existing one."
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Index building (runs in background thread)
    # ─────────────────────────────────────────────────────────────────────────

    def _build_index_async(self):
        if self._pdf_folder is None:
            messagebox.showwarning("No folder", "Please select a PDF folder first.")
            return
        self._btn_build.config(state="disabled")
        self._progress.start(12)
        threading.Thread(target=self._build_index, daemon=True).start()

    def _build_index(self):
        try:
            self._refresh_status("Building index…")
            model_key     = self._model_var.get()
            use_chunking  = self._chunking_var.get()
            chunk_size    = self._chunk_size_var.get()
            chunk_overlap = self._chunk_overlap_var.get()

            # ── registry ──────────────────────────────────────────────────────
            registry = DocumentRegistry(config.REGISTRY_FILE)
            registry.scan_folder(self._pdf_folder, reset=True)
            if not registry.documents:
                self._root_call(
                    messagebox.showwarning, "No PDFs",
                    "No PDF files found in the selected folder."
                )
                return
            self._registry = registry
            self._refresh_status(
                f"Registered {len(registry)} PDF(s). Extracting text…"
            )

            # ── extraction + preprocessing ────────────────────────────────────
            extractor    = HybridExtractor(
                min_text_length = config.OCR_MIN_TEXT_LENGTH,
                tesseract_cmd   = config.TESSERACT_CMD,
                ocr_dpi         = config.OCR_DPI,
                ocr_language    = config.OCR_LANGUAGE,
            )
            preprocessor = TextPreprocessor(
                use_chunking  = use_chunking,
                chunk_size    = chunk_size,
                chunk_overlap = chunk_overlap,
            )
            embedder = EmbeddingEngine(
                model_key  = model_key,
                models_dir = config.MODELS_DIR,
            )

            all_chunks:  list = []
            chunk_metas: list = []
            doc_texts:   list = []   # for keyword index

            for doc_entry in registry:
                self._refresh_status(
                    f"Extracting: {doc_entry['filename']}"
                )
                try:
                    ext_result = extractor.extract_document(doc_entry["filepath"])
                except Exception as exc:
                    logger.error("Extraction failed for %s: %s",
                                 doc_entry["filename"], exc)
                    continue

                chunks = preprocessor.process(
                    ext_result["full_text"], doc_entry["doc_id"]
                )
                for chunk in chunks:
                    all_chunks.append(chunk.text)
                    chunk_metas.append(ChunkMeta(
                        faiss_id    = len(chunk_metas),
                        doc_id      = doc_entry["doc_id"],
                        chunk_index = chunk.chunk_index,
                        filename    = doc_entry["filename"],
                        filepath    = doc_entry["filepath"],
                        text        = chunk.text,
                    ))

                doc_texts.append({
                    "doc_id":   doc_entry["doc_id"],
                    "filename": doc_entry["filename"],
                    "filepath": doc_entry["filepath"],
                    "text":     ext_result["full_text"],
                })

            if not all_chunks:
                self._root_call(
                    messagebox.showwarning, "No text",
                    "Could not extract any text from the selected PDFs."
                )
                return

            self._refresh_status(
                f"Generating embeddings for {len(all_chunks)} chunk(s)…"
            )

            # ── embeddings ────────────────────────────────────────────────────
            import time
            t_embed = time.perf_counter()
            embeddings = embedder.encode(all_chunks)
            embed_time = time.perf_counter() - t_embed

            # ── FAISS index ───────────────────────────────────────────────────
            faiss_idx = FAISSIndex(
                model_key   = model_key,
                indexes_dir = config.INDEXES_DIR,
            )
            faiss_idx.build(embeddings, chunk_metas)
            faiss_idx.save()

            # ── keyword index ─────────────────────────────────────────────────
            kw = KeywordSearch()
            kw.build(doc_texts)
            self._keyword = kw

            # ── semantic search object ────────────────────────────────────────
            self._semantic = SemanticSearch(
                model_key   = model_key,
                indexes_dir = config.INDEXES_DIR,
                models_dir  = config.MODELS_DIR,
                snippet_chars = config.SNIPPET_CONTEXT_CHARS,
            )
            self._semantic.load_index()

            # ── log experiment ────────────────────────────────────────────────
            run_id = ExperimentLogger.make_run_id(model_key)
            self._current_run_id = run_id
            self._exp_logger.log_experiment(
                run_id             = run_id,
                model_key          = model_key,
                model_name         = config.EMBEDDING_MODELS[model_key]["name"],
                search_type        = "semantic",
                num_documents      = len(registry),
                num_chunks         = len(all_chunks),
                embedding_dim      = embedder.dimension,
                indexing_time_s    = faiss_idx.build_time,
                embedding_time_s   = embed_time,
                use_chunking       = use_chunking,
                chunk_size         = chunk_size,
                chunk_overlap      = chunk_overlap,
                index_size_vectors = faiss_idx.total_vectors,
            )

            info = (
                f"Model: {model_key.upper()}\n"
                f"Documents: {len(registry)}\n"
                f"Chunks: {len(all_chunks)}\n"
                f"Embedding dim: {embedder.dimension}\n"
                f"Embed time: {embed_time:.1f}s"
            )
            self._root_call(self._index_info_var.set, info)
            self._refresh_status(
                f"Index built — {len(all_chunks)} chunks indexed. Ready to search."
            )

        except Exception as exc:
            logger.exception("Index build failed.")
            self._root_call(
                messagebox.showerror, "Build Failed", str(exc)
            )
            self._refresh_status("Index build failed. See logs.")

        finally:
            self._root_call(self._progress.stop)
            self._root_call(self._btn_build.config, state="normal")

    # ─────────────────────────────────────────────────────────────────────────
    # Search
    # ─────────────────────────────────────────────────────────────────────────

    def _run_search(self):
        query = self._query_var.get().strip()
        if not query:
            return

        search_type = self._search_type_var.get()
        top_k       = self._topk_var.get()

        if search_type == "semantic":
            if self._semantic is None or not self._semantic.is_ready:
                messagebox.showwarning(
                    "No index",
                    "Please build the index first."
                )
                return
            try:
                result = self._semantic.search(query, top_k=top_k)
            except Exception as exc:
                messagebox.showerror("Search Error", str(exc))
                return
        else:
            if self._keyword is None or not self._keyword.is_built:
                messagebox.showwarning("No index", "Please build the index first.")
                return
            try:
                result = self._keyword.search(query, top_k=top_k)
            except Exception as exc:
                messagebox.showerror("Search Error", str(exc))
                return

        # Log search
        if self._current_run_id:
            self._exp_logger.log_search(
                run_id        = self._current_run_id,
                model_key     = self._model_var.get(),
                search_type   = search_type,
                query         = query,
                top_k         = top_k,
                query_time_ms = result.get("query_time_ms", 0.0),
                num_results   = len(result.get("results", [])),
            )

        self._display_results(result)
        self._refresh_status(
            f"'{query}' → {len(result['results'])} result(s)  "
            f"({result.get('query_time_ms', 0):.1f} ms)"
        )

    def _display_results(self, result: dict):
        # Clear tree
        for item in self._tree.get_children():
            self._tree.delete(item)
        self._detail_text.delete("1.0", tk.END)

        self._result_rows = result.get("results", [])

        for hit in self._result_rows:
            snippet_preview = hit.get("snippet", "")[:120].replace("\n", " ")
            self._tree.insert(
                "", "end",
                iid=str(hit["rank"]),
                values=(
                    hit["rank"],
                    f"{hit['score']:.4f}",
                    hit["filename"],
                    snippet_preview,
                ),
            )

    def _on_result_double_click(self, _event):
        sel = self._tree.selection()
        if not sel:
            return
        rank = int(sel[0])
        hit  = next((r for r in self._result_rows if r["rank"] == rank), None)
        if not hit:
            return

        # Show full snippet
        self._detail_text.delete("1.0", tk.END)
        self._detail_text.insert(
            tk.END,
            f"[{hit['filename']}]\n\n{hit.get('snippet', '')}"
        )

        # Open the PDF
        filepath = hit.get("filepath", "")
        if filepath and Path(filepath).exists():
            try:
                os.startfile(filepath)
            except Exception as exc:
                messagebox.showerror("Cannot open file", str(exc))
        else:
            messagebox.showwarning("File not found", filepath)

    # ─────────────────────────────────────────────────────────────────────────
    # Helpers
    # ─────────────────────────────────────────────────────────────────────────

    def _refresh_status(self, msg: str):
        self._root_call(self._status_var.set, msg)
        logger.info(msg)

    def _root_call(self, fn, *args, **kwargs):
        """Thread-safe call to update the Tkinter main thread."""
        if kwargs:
            self.root.after(0, lambda: fn(*args, **kwargs))
        else:
            self.root.after(0, fn, *args)

    # ─────────────────────────────────────────────────────────────────────────
    # Entry point
    # ─────────────────────────────────────────────────────────────────────────

    @classmethod
    def run(cls):
        root = tk.Tk()
        app  = cls(root)
        root.mainloop()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
    )
    SemanticSearchApp.run()
