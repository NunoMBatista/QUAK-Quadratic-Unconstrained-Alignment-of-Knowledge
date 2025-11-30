from __future__ import annotations

import copy
import random
import threading
import tkinter as tk
import tkinter.font as tkfont
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch

from . import storage
from .graph_model import GraphModel, GraphNode
from .manual_pipeline import EmbeddingMode, ManualPipelineRunner, PipelineResult, AlignmentRow
from src.kg_construction.nlp_pipeline import NLPPipeline, RelationTriple
from src.kg_construction.llm_based_pipeline import LLMBasedPipeline
from . import storage
import urllib.request
from html import unescape
from src.config import DEFAULT_SIMILARITY_THRESHOLD, QUBO_NODE_WEIGHT, QUBO_STRUCTURE_WEIGHT
from src.quak import QUAK_ASCII_MASCOT, QUAK_WORDMARK

CANVAS_WIDTH = 420
CANVAS_HEIGHT = 320
NODE_RADIUS = 18
APP_FONT_FAMILY = "Courier New"


def app_font(size: int, *modifiers: str) -> Tuple[Any, ...]:
    if modifiers:
        return (APP_FONT_FAMILY, size, *modifiers)
    return (APP_FONT_FAMILY, size)


def retro_button(master: tk.Misc, **kwargs: Any) -> tk.Button:
    options: Dict[str, Any] = {
        "font": app_font(10),
        "bg": RETRO_BTN_BG,
        "fg": RETRO_TEXT,
        "activebackground": RETRO_BTN_ACTIVE,
        "activeforeground": RETRO_TEXT,
        "relief": "raised",
        "bd": 3,
        "highlightthickness": 1,
        "highlightbackground": RETRO_BORDER_LIGHT,
        "highlightcolor": RETRO_BORDER_LIGHT,
        "cursor": "hand2",
        "takefocus": True,
        "padx": 6,
        "pady": 2,
    }
    options.update(kwargs)
    normal_bg = options.get("bg", RETRO_BTN_BG)
    button = tk.Button(master, **options)

    def _hover(_: tk.Event) -> None:  # type: ignore[override]
        button.configure(bg=RETRO_BTN_HOVER)

    def _leave(_: tk.Event) -> None:  # type: ignore[override]
        button.configure(bg=normal_bg)

    button.bind("<Enter>", _hover)
    button.bind("<Leave>", _leave)
    return button


def configure_retro_style(style: ttk.Style) -> None:
    style.configure("TFrame", background=RETRO_PANEL_BG)
    style.configure("TLabel", background=RETRO_PANEL_BG, foreground=RETRO_TEXT)
    style.configure("TEntry", fieldbackground=RETRO_CANVAS_BG, foreground=RETRO_TEXT)
    style.configure("TCombobox", fieldbackground=RETRO_CANVAS_BG, foreground=RETRO_TEXT)
    style.configure("Retro.TFrame", background=RETRO_PANEL_BG, borderwidth=2, relief="groove")
    style.configure(
        "Retro.TLabelframe",
        background=RETRO_PANEL_BG,
        borderwidth=2,
        relief="groove",
    )
    style.configure(
        "Retro.TLabelframe.Label",
        background=RETRO_PANEL_BG,
        foreground=RETRO_TEXT,
        font=app_font(10, "bold"),
    )
    style.configure(
        "Treeview",
        background=RETRO_CANVAS_BG,
        fieldbackground=RETRO_CANVAS_BG,
        foreground=RETRO_TEXT,
        bordercolor=RETRO_BORDER_DARK,
        rowheight=22,
    )
    style.configure(
        "Treeview.Heading",
        background=RETRO_BTN_BG,
        foreground=RETRO_TEXT,
        font=app_font(10, "bold"),
    )
    style.map("Treeview", background=[("selected", RETRO_SELECTION_BG)])
    style.map("Treeview", foreground=[("selected", "#ffffff")])


RETRO_BG = "#c0c0c0"
RETRO_PANEL_BG = "#d4d0c8"
RETRO_CANVAS_BG = "#ffffff"
RETRO_TEXT = "#000000"
RETRO_BTN_BG = "#dfdfdf"
RETRO_BTN_HOVER = "#f5f5f5"
RETRO_BTN_ACTIVE = "#bdbdbd"
RETRO_BORDER_DARK = "#7b7b7b"
RETRO_BORDER_LIGHT = "#ffffff"
RETRO_SELECTION_BG = "#0a246a"


@dataclass
class GraphPanelState:
    model: GraphModel
    color: str


class GraphPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, color: str) -> None:
        super().__init__(master, padding=8, style="Retro.TFrame")
        self.panel_state = GraphPanelState(model=GraphModel(title), color=color)
        self._drag_node: Optional[str] = None
        self._drag_offset = (0.0, 0.0)
        self._node_order: List[str] = []
        self._edge_order: List[int] = []

        ttk.Label(self, text=title, font=app_font(14, "bold"), foreground="#000000").pack(
            anchor="w", pady=(0, 6)
        )

        self.canvas = tk.Canvas(
            self,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg=RETRO_CANVAS_BG,
            highlightbackground=RETRO_BORDER_DARK,
            highlightthickness=1,
            relief="sunken",
            borderwidth=2,
        )
        self.canvas.pack(fill="x", pady=(0, 8))
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        button_row = ttk.Frame(self, style="Retro.TFrame")
        button_row.pack(fill="x", pady=4)
        retro_button(button_row, text="Add Node", command=self._prompt_add_node).pack(side="left", padx=2)
        retro_button(button_row, text="Edit Node", command=self._prompt_edit_node).pack(side="left", padx=2)
        retro_button(button_row, text="Add Edge", command=self._prompt_add_edge).pack(side="left", padx=2)
        retro_button(button_row, text="Save", command=self._save_graph).pack(side="left", padx=2)
        retro_button(button_row, text="Load", command=self._load_graph).pack(side="left", padx=2)
        retro_button(button_row, text="Clear", command=self._clear_graph).pack(side="left", padx=2)
        retro_button(button_row, text="Delete Selected", command=self._delete_selected).pack(side="right")

        lists_frame = ttk.Frame(self, style="Retro.TFrame")
        lists_frame.pack(fill="both", expand=True)
        self.node_list = tk.Listbox(
            lists_frame,
            height=6,
            background=RETRO_CANVAS_BG,
            foreground=RETRO_TEXT,
            font=app_font(10),
            selectbackground=RETRO_SELECTION_BG,
            selectforeground="#ffffff",
            relief="sunken",
            borderwidth=2,
            highlightbackground=RETRO_BORDER_DARK,
            highlightcolor=RETRO_BORDER_DARK,
        )
        self.edge_list = tk.Listbox(
            lists_frame,
            height=6,
            background=RETRO_CANVAS_BG,
            foreground=RETRO_TEXT,
            font=app_font(10),
            selectbackground=RETRO_SELECTION_BG,
            selectforeground="#ffffff",
            relief="sunken",
            borderwidth=2,
            highlightbackground=RETRO_BORDER_DARK,
            highlightcolor=RETRO_BORDER_DARK,
        )
        ttk.Label(lists_frame, text="Nodes").grid(row=0, column=0, sticky="w")
        ttk.Label(lists_frame, text="Edges").grid(row=0, column=1, sticky="w")
        self.node_list.grid(row=1, column=0, sticky="nsew", padx=(0, 4))
        self.edge_list.grid(row=1, column=1, sticky="nsew")
        self.node_list.bind("<Double-Button-1>", self._on_node_list_double_click)
        lists_frame.columnconfigure(0, weight=1)
        lists_frame.columnconfigure(1, weight=1)

        self._refresh_ui()

    # ------------------------------------------------------------------
    def _prompt_add_node(self) -> None:
        dialog = tk.Toplevel(self)
        dialog.title("Add Node")
        dialog.transient(self.winfo_toplevel())
        dialog.configure(bg=RETRO_BG)

        ttk.Label(dialog, text="Label").grid(row=0, column=0, sticky="w")
        label_entry = ttk.Entry(dialog, width=30)
        label_entry.grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(dialog, text="Description (optional)").grid(row=1, column=0, sticky="w")
        desc_entry = ttk.Entry(dialog, width=30)
        desc_entry.grid(row=1, column=1, padx=6, pady=4)

        def submit() -> None:
            label = label_entry.get().strip()
            if not label:
                messagebox.showwarning("Missing label", "Please provide a label for the node.")
                return
            position = (
                random.randint(NODE_RADIUS + 10, CANVAS_WIDTH - NODE_RADIUS - 10),
                random.randint(NODE_RADIUS + 10, CANVAS_HEIGHT - NODE_RADIUS - 10),
            )
            node = self.panel_state.model.add_node(label, desc_entry.get().strip(), position)
            self._refresh_ui()
            dialog.destroy()

        retro_button(dialog, text="Add", command=submit).grid(row=2, column=0, columnspan=2, pady=8)
        dialog.grab_set()
        label_entry.focus_set()

    def _prompt_add_edge(self) -> None:
        nodes = self.panel_state.model.list_nodes()
        if len(nodes) < 2:
            messagebox.showwarning("Add more nodes", "You need at least two nodes before creating an edge.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Add Edge")
        dialog.transient(self.winfo_toplevel())
        dialog.configure(bg=RETRO_BG)

        ttk.Label(dialog, text="Source").grid(row=0, column=0, sticky="w")
        ttk.Label(dialog, text="Target").grid(row=1, column=0, sticky="w")
        ttk.Label(dialog, text="Relation").grid(row=2, column=0, sticky="w")

        node_labels = [node.label for node in nodes]
        source_var = tk.StringVar(value=node_labels[0])
        target_var = tk.StringVar(value=node_labels[1])
        relation_var = tk.StringVar(value="related_to")

        ttk.Combobox(dialog, textvariable=source_var, values=node_labels, state="readonly").grid(
            row=0, column=1, padx=6, pady=4
        )
        ttk.Combobox(dialog, textvariable=target_var, values=node_labels, state="readonly").grid(
            row=1, column=1, padx=6, pady=4
        )
        ttk.Entry(dialog, textvariable=relation_var).grid(row=2, column=1, padx=6, pady=4)

        lookup = {node.label: node.id for node in nodes}

        def submit() -> None:
            relation = relation_var.get().strip()
            if not relation:
                messagebox.showwarning("Missing relation", "Provide a relation label.")
                return
            src = lookup[source_var.get()]
            dst = lookup[target_var.get()]
            if src == dst:
                messagebox.showwarning("Invalid edge", "Choose two different nodes.")
                return
            self.panel_state.model.add_edge(src, dst, relation)
            self._refresh_ui()
            dialog.destroy()

        retro_button(dialog, text="Add", command=submit).grid(row=3, column=0, columnspan=2, pady=8)
        dialog.grab_set()

    def _save_graph(self) -> None:
        initial = storage.HANDMADE_DIR / f"{self.panel_state.model.name.lower().replace(' ', '_')}.json"
        path = filedialog.asksaveasfilename(
            title="Save Graph",
            defaultextension=".json",
            initialdir=storage.HANDMADE_DIR,
            initialfile=initial.name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        storage.save_graph(self.panel_state.model, filename=Path(path))
        messagebox.showinfo("Graph saved", f"Saved graph to {path}")

    def _load_graph(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Graph",
            initialdir=storage.HANDMADE_DIR,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            model = storage.load_graph(Path(path))
        except Exception as exc:  # pragma: no cover - GUI feedback
            messagebox.showerror("Failed to load", str(exc))
            return
        self.panel_state.model = model
        self._refresh_ui()

    def _clear_graph(self) -> None:
        if not messagebox.askyesno(
            "Clear graph", f"Remove every node/edge from {self.panel_state.model.name}?"
        ):
            return
        self.panel_state.model.clear()
        self._refresh_ui()

    # ------------------------------------------------------------------
    def _refresh_ui(self) -> None:
        self._draw_graph()
        self._refresh_lists()

    def _draw_graph(self) -> None:
        self.canvas.delete("all")
        nodes = self.panel_state.model.list_nodes()
        node_lookup = {node.id: node for node in nodes}
        for edge in self.panel_state.model.list_edges():
            src = node_lookup.get(edge.source)
            dst = node_lookup.get(edge.target)
            if not src or not dst:
                continue
            self.canvas.create_line(*src.position, *dst.position, fill="#3b82f6", width=2)
            mid_x = (src.position[0] + dst.position[0]) / 2
            mid_y = (src.position[1] + dst.position[1]) / 2
            self.canvas.create_text(mid_x, mid_y, text=edge.relation, fill="#000000", font=app_font(9))
        for node in nodes:
            x, y = node.position
            self.canvas.create_oval(
                x - NODE_RADIUS,
                y - NODE_RADIUS,
                x + NODE_RADIUS,
                y + NODE_RADIUS,
                fill=self.panel_state.color,
                outline="#f9fafb",
                width=2,
            )
            self.canvas.create_text(x, y, text=node.label[:6], fill="#111827", font=app_font(10, "bold"))

    def _refresh_lists(self) -> None:
        selection = self.node_list.curselection()
        selected_id: Optional[str] = None
        if selection:
            index = selection[0]
            if 0 <= index < len(self._node_order):
                selected_id = self._node_order[index]

        edge_selection = self.edge_list.curselection()
        selected_edge_key: Optional[int] = None
        if edge_selection:
            edge_index = edge_selection[0]
            if 0 <= edge_index < len(self._edge_order):
                selected_edge_key = self._edge_order[edge_index]

        self.node_list.delete(0, tk.END)
        self._node_order = []
        for idx, node in enumerate(self.panel_state.model.list_nodes()):
            desc = f" - {node.description}" if node.description else ""
            self.node_list.insert(tk.END, f"{node.label}{desc}")
            self._node_order.append(node.id)
            if selected_id and node.id == selected_id:
                self.node_list.selection_set(idx)

        self.edge_list.delete(0, tk.END)
        self._edge_order = []
        for edge_idx, edge in enumerate(self.panel_state.model.list_edges()):
            src = self.panel_state.model.nodes.get(edge.source)
            dst = self.panel_state.model.nodes.get(edge.target)
            src_label = src.label if src else edge.source
            dst_label = dst.label if dst else edge.target
            list_index = self.edge_list.size()
            self.edge_list.insert(tk.END, f"{src_label} --{edge.relation}--> {dst_label}")
            self._edge_order.append(edge_idx)
            if selected_edge_key is not None and edge_idx == selected_edge_key:
                self.edge_list.selection_set(list_index)

    def _prompt_edit_node(self) -> None:
        node_id = self._get_selected_node_id()
        if not node_id:
            messagebox.showinfo("Select a node", "Choose a node from the list to edit.")
            return
        self._open_node_editor(node_id)

    def _get_selected_node_id(self) -> Optional[str]:
        selection = self.node_list.curselection()
        if not selection:
            return None
        index = selection[0]
        if 0 <= index < len(self._node_order):
            return self._node_order[index]
        return None

    def _get_selected_edge_index(self) -> Optional[int]:
        selection = self.edge_list.curselection()
        if not selection:
            return None
        index = selection[0]
        if 0 <= index < len(self._edge_order):
            return self._edge_order[index]
        return None

    def _on_node_list_double_click(self, _: tk.Event) -> None:  # type: ignore[override]
        node_id = self._get_selected_node_id()
        if node_id:
            self._open_node_editor(node_id)

    def _delete_selected(self) -> None:
        node_id = self._get_selected_node_id()
        if node_id:
            node = self.panel_state.model.nodes.get(node_id)
            label = node.label if node else node_id
            if messagebox.askyesno("Delete node", f"Remove node '{label}' and its edges?"):
                self.panel_state.model.remove_node(node_id)
                self._refresh_ui()
            return

        edge_index = self._get_selected_edge_index()
        if edge_index is not None:
            edges = self.panel_state.model.list_edges()
            edge_desc = "the selected edge"
            if 0 <= edge_index < len(edges):
                edge = edges[edge_index]
                src = self.panel_state.model.nodes.get(edge.source)
                dst = self.panel_state.model.nodes.get(edge.target)
                src_label = src.label if src else edge.source
                dst_label = dst.label if dst else edge.target
                edge_desc = f"{src_label} --{edge.relation}--> {dst_label}"
            if messagebox.askyesno("Delete edge", f"Remove {edge_desc}?"):
                self.panel_state.model.remove_edge(edge_index)
                self._refresh_ui()
            return

        messagebox.showinfo("Select an item", "Select a node or edge to delete.")

    def _open_node_editor(self, node_id: str) -> None:
        node = self.panel_state.model.nodes.get(node_id)
        if not node:
            messagebox.showerror("Missing node", "Could not find the selected node for editing.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Edit Node")
        dialog.transient(self.winfo_toplevel())
        dialog.configure(bg=RETRO_BG)
        ttk.Label(dialog, text="Label").grid(row=0, column=0, sticky="w")
        label_var = tk.StringVar(value=node.label)
        ttk.Entry(dialog, textvariable=label_var, width=30).grid(row=0, column=1, padx=6, pady=4)
        ttk.Label(dialog, text="Description").grid(row=1, column=0, sticky="w")
        desc_var = tk.StringVar(value=node.description)
        ttk.Entry(dialog, textvariable=desc_var, width=30).grid(row=1, column=1, padx=6, pady=4)

        def save() -> None:
            new_label = label_var.get().strip()
            if not new_label:
                messagebox.showwarning("Missing label", "Provide a label for the node.")
                return
            self.panel_state.model.update_node(node_id, label=new_label, description=desc_var.get())
            self._refresh_ui()
            dialog.destroy()

        button_row = ttk.Frame(dialog, style="Retro.TFrame")
        button_row.grid(row=2, column=0, columnspan=2, pady=8)
        retro_button(button_row, text="Save", command=save).pack(side="left", padx=4)
        retro_button(button_row, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
        dialog.grab_set()

    # ------------------------------------------------------------------
    def _hit_test(self, x: float, y: float) -> Optional[str]:
        for node in self.panel_state.model.list_nodes():
            nx, ny = node.position
            if (x - nx) ** 2 + (y - ny) ** 2 <= NODE_RADIUS ** 2:
                return node.id
        return None

    def _on_canvas_press(self, event: tk.Event) -> None:  # type: ignore[override]
        node_id = self._hit_test(event.x, event.y)
        if node_id:
            node = self.panel_state.model.nodes[node_id]
            self._drag_node = node_id
            self._drag_offset = (event.x - node.position[0], event.y - node.position[1])

    def _on_canvas_drag(self, event: tk.Event) -> None:  # type: ignore[override]
        if not self._drag_node:
            return
        node = self.panel_state.model.nodes[self._drag_node]
        new_x = min(max(event.x - self._drag_offset[0], NODE_RADIUS), CANVAS_WIDTH - NODE_RADIUS)
        new_y = min(max(event.y - self._drag_offset[1], NODE_RADIUS), CANVAS_HEIGHT - NODE_RADIUS)
        node.position = (new_x, new_y)
        self._draw_graph()

    def _on_canvas_release(self, _: tk.Event) -> None:  # type: ignore[override]
        self._drag_node = None

    # ------------------------------------------------------------------
    @property
    def model(self) -> GraphModel:
        return self.panel_state.model

    def set_model(self, model: GraphModel) -> None:
        self.panel_state.model = model
        self._refresh_ui()


class ResultsPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, ascii_art: Optional[str] = None) -> None:
        super().__init__(master, padding=8, style="Retro.TFrame")

        # Header with collapse toggle
        header = ttk.Frame(self, style="Retro.TFrame")
        header.pack(fill="x")
        ttk.Label(header, text="Alignment Results", font=app_font(14, "bold")).pack(side="left", anchor="w")
        # keep references for sash manipulation when collapsing
        self._header = header
        self._prev_sashpos: Optional[int] = None
        self._collapsed = False
        self._collapse_text = tk.StringVar(value="▾")

        def _toggle() -> None:
            self._collapsed = not self._collapsed
            self._collapse_text.set("▸" if self._collapsed else "▾")
            if self._collapsed:
                # hide content
                self._content_frame.pack_forget()
                # try to locate a PanedWindow parent to adjust its sash so
                # the bottom pane becomes as small as the header
                try:
                    parent = getattr(self, "master", None)
                    grand = getattr(parent, "master", None)
                    if isinstance(grand, ttk.PanedWindow):
                        paned: ttk.PanedWindow = grand  # type: ignore[assignment]
                        paned.update_idletasks()
                        total_h = paned.winfo_height()
                        header_h = self._header.winfo_height() or 24
                        # enforce a minimum height for the collapsed bottom pane (header + some padding)
                        min_h = header_h + 36
                        try:
                            paned.paneconfigure(parent, minsize=min_h)
                        except Exception:
                            pass
                        # remember previous sash position (if any)
                        try:
                            self._prev_sashpos = paned.sashpos(0)
                        except Exception:
                            self._prev_sashpos = None
                        newpos = max(0, total_h - min_h)
                        try:
                            paned.sashpos(0, newpos)
                        except Exception:
                            pass
                    else:
                        # fallback: try pack_configure to shrink
                        try:
                            self.pack_configure(fill="x", expand=False, side="bottom")
                        except tk.TclError:
                            pass
                except Exception:
                    pass
            else:
                # restore expanded content and sizing
                self._content_frame.pack(fill="both", expand=True)
                try:
                    parent = getattr(self, "master", None)
                    grand = getattr(parent, "master", None)
                    if isinstance(grand, ttk.PanedWindow):
                        try:
                            # restore sash position if we have it
                            if self._prev_sashpos is not None:
                                grand.sashpos(0, self._prev_sashpos)
                            # restore a reasonable minsize for the pane
                            grand.paneconfigure(parent, minsize=80)
                        except Exception:
                            pass
                except Exception:
                    pass

        retro_button(header, textvariable=self._collapse_text, command=_toggle).pack(side="right")

        # Content goes into a frame that can be collapsed
        self._content_frame = ttk.Frame(self, style="Retro.TFrame")
        self._content_frame.pack(fill="both", expand=True)

        tree_row = ttk.Frame(self._content_frame, style="Retro.TFrame")
        tree_row.pack(fill="x", pady=(0, 4))

        alignments_frame = ttk.Frame(tree_row, style="Retro.TFrame")
        alignments_frame.pack(side="left", fill="both", expand=True)
        nn_columns = [
            {"id": "wiki", "label": "Wiki Entity", "width": 160},
            {"id": "arxiv", "label": "arXiv Entity", "width": 160},
            {"id": "similarity", "label": "Similarity", "width": 120, "anchor": "center"},
        ]
        qubo_columns = [
            {"id": "wiki", "label": "Wiki Entity", "width": 140},
            {"id": "arxiv", "label": "arXiv Entity", "width": 140},
            {"id": "similarity", "label": "Similarity", "width": 90, "anchor": "center"},
            {"id": "structure", "label": "Structure", "width": 90, "anchor": "center"},
            {"id": "total", "label": "Total", "width": 90, "anchor": "center"},
        ]
        self.nn_tree = self._build_tree("Nearest Neighbor Alignments", nn_columns, parent=alignments_frame)
        self.qubo_tree = self._build_tree("QUBO Alignments", qubo_columns, parent=alignments_frame)

        unaligned_frame = ttk.LabelFrame(
            tree_row,
            text="Unaligned (below thresholds)",
            style="Retro.TLabelframe",
            padding=4,
        )
        unaligned_frame.pack(side="left", fill="y", padx=(8, 0))
        self.unaligned_tree = self._build_unaligned_tree(unaligned_frame)
        self.unaligned_summary = tk.StringVar(value="Run an alignment to see threshold misses.")
        tk.Label(
            unaligned_frame,
            textvariable=self.unaligned_summary,
            wraplength=180,
            foreground="#475569",
            background=RETRO_PANEL_BG,
        ).pack(anchor="w", pady=(4, 0))

        ttk.Label(self._content_frame, text="Pipeline Log").pack(anchor="w", pady=(8, 0))
        self.log_text = tk.Text(
            self._content_frame,
            height=6,
            width=80,
            state="disabled",
            background=RETRO_CANVAS_BG,
            foreground=RETRO_TEXT,
            borderwidth=2,
            relief="sunken",
            font=app_font(10),
        )
        self.log_text.pack(fill="both", expand=True)

        self.energy_var = tk.StringVar(value="QUBO energy: -")
        ttk.Label(
            self._content_frame,
            textvariable=self.energy_var,
            font=app_font(11, "italic"),
            foreground="#f97316",
        ).pack(anchor="w", pady=6)
        if ascii_art:
            art_frame = tk.Frame(self._content_frame, bg=RETRO_BG, padx=6, pady=4, borderwidth=2, relief="sunken")
            art_frame.pack(fill="x", anchor="e")
            tk.Label(
                art_frame,
                text=ascii_art,
                font=app_font(8),
                justify="right",
                bg=RETRO_BG,
                fg="#000080",
            ).pack(anchor="e")

    def _build_tree(
        self,
        title: str,
        columns: Sequence[Dict[str, object]],
        *,
        parent: Optional[tk.Misc] = None,
    ) -> ttk.Treeview:
        container = parent if parent is not None else self
        ttk.Label(container, text=title, font=app_font(11, "bold")).pack(anchor="w", pady=(8, 0))
        column_ids = [cast(str, spec["id"]) for spec in columns]
        tree = ttk.Treeview(container, columns=column_ids, show="headings", height=6)
        for spec in columns:
            col_id = cast(str, spec["id"])
            heading = cast(str, spec.get("label", col_id.title()))
            width = cast(int, spec.get("width", 140))
            anchor = cast(str, spec.get("anchor", "w"))
            tree.heading(col_id, text=heading)
            tree.column(col_id, width=width, anchor=anchor)  # type: ignore[arg-type]
        tree.pack(fill="x", pady=(2, 0))
        return tree

    def _build_unaligned_tree(self, parent: tk.Misc) -> ttk.Treeview:
        columns = ("graph", "entity", "best", "trigger")
        tree = ttk.Treeview(parent, columns=columns, show="headings", height=7)
        headings = ("Graph", "Entity", "Best sim", "Trigger")
        widths = (70, 160, 80, 140)
        for column, heading, width in zip(columns, headings, widths):
            tree.heading(column, text=heading)
            tree.column(column, width=width, anchor="center" if column != "entity" else "w")
        tree.pack(fill="both", expand=True)
        return tree

    def display_result(self, result: PipelineResult) -> None:
        self._populate_tree(self.nn_tree, [row.as_nn_tuple() for row in result.nn_alignments])
        self._populate_tree(self.qubo_tree, [row.as_qubo_tuple() for row in result.qubo_alignments])
        self._populate_tree(self.unaligned_tree, [entry.as_tuple() for entry in result.unaligned_entities])
        if result.unaligned_entities:
            self.unaligned_summary.set(
                f"{len(result.unaligned_entities)} entities filtered by the active thresholds."
            )
        else:
            self.unaligned_summary.set("Every entity met the thresholds that were applied.")
        self.energy_var.set(
            f"QUBO energy: {result.qubo_energy:.4f}" if result.qubo_energy is not None else "QUBO energy: -"
        )
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        for line in result.logs:
            self.log_text.insert(tk.END, f"• {line}\n")
        self.log_text.configure(state="disabled")

    def show_logs(self, logs: List[str]) -> None:
        self._populate_tree(self.nn_tree, [])
        self._populate_tree(self.qubo_tree, [])
        self._populate_tree(self.unaligned_tree, [])
        self.unaligned_summary.set("Run an alignment to see threshold misses.")
        self.energy_var.set("QUBO energy: -")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        for line in logs:
            self.log_text.insert(tk.END, f"• {line}\n")
        self.log_text.configure(state="disabled")

    def _populate_tree(self, tree: ttk.Treeview, rows: Sequence[Tuple[object, ...]]) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", tk.END, values=row)


class NodeMatrixEditor(tk.Toplevel):
    def __init__(self, master: tk.Misc, node_info: Dict[str, object]) -> None:
        super().__init__(master)
        self.title("Edit H_node Matrix")
        self.geometry("600x420")
        self.configure(bg=RETRO_BG)
        self.node_info = node_info
        self.similarity: torch.Tensor = cast(torch.Tensor, node_info["similarity"])
        self.wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        self.arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]
        self._selected_item: Optional[str] = None

        columns = ("wiki", "arxiv", "similarity")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=14)
        for col, title in zip(columns, ("Wiki", "arXiv", "Similarity")):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=180 if col != "similarity" else 120, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        form = ttk.Frame(self, style="Retro.TFrame")
        form.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(form, text="New value:").pack(side="left")
        self.value_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.value_var, width=12).pack(side="left", padx=6)
        retro_button(form, text="Update", command=self._apply_change).pack(side="left", padx=4)
        retro_button(form, text="Close", command=self.destroy).pack(side="right")

        self._populate_rows()

    def _populate_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for i, wiki in enumerate(self.wiki_nodes):
            for j, arxiv in enumerate(self.arxiv_nodes):
                value = float(self.similarity[i, j])
                item_id = self.tree.insert(
                    "",
                    tk.END,
                    values=(wiki.label, arxiv.label, f"{value:.4f}"),
                )
                self.tree.set(item_id, "wiki", wiki.label)
                self.tree.set(item_id, "arxiv", arxiv.label)

    def _on_select(self, _: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            self._selected_item = None
            return
        self._selected_item = selection[0]
        self.value_var.set(self.tree.set(self._selected_item, "similarity"))

    def _apply_change(self) -> None:
        if not self._selected_item:
            messagebox.showwarning("Select an entry", "Choose a pair to edit.")
            return
        try:
            new_value = float(self.value_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Please enter a valid number.")
            return

        row_index = self.tree.index(self._selected_item)
        total_cols = len(self.arxiv_nodes)
        wiki_idx = row_index // total_cols
        arxiv_idx = row_index % total_cols
        self.similarity[wiki_idx, arxiv_idx] = new_value
        self.tree.set(self._selected_item, "similarity", f"{new_value:.4f}")


class StructuralMatrixEditor(tk.Toplevel):
    def __init__(
        self,
        master: tk.Misc,
        node_info: Dict[str, object],
        structural_info: Dict[str, object],
    ) -> None:
        super().__init__(master)
        self.title("Edit H_structure Matrix")
        self.geometry("700x420")
        self.configure(bg=RETRO_BG)
        self.node_info = node_info
        self.structural_info = structural_info
        weights_raw = structural_info.get("weights", {})
        formatted: Dict[Tuple[int, int, int, int], float] = {}
        if isinstance(weights_raw, dict):
            source_items = weights_raw.items()
        else:
            source_items = []
        for key, value in source_items:
            if isinstance(key, (list, tuple)) and len(key) == 4:
                idx_tuple: Tuple[int, int, int, int] = (
                    int(key[0]),
                    int(key[1]),
                    int(key[2]),
                    int(key[3]),
                )
            else:
                continue
            formatted[idx_tuple] = float(value)
        self.weights: Dict[Tuple[int, int, int, int], float] = formatted
        self.structural_info["weights"] = self.weights
        self.wiki_nodes: Sequence[GraphNode] = node_info["wiki_nodes"]  # type: ignore[assignment]
        self.arxiv_nodes: Sequence[GraphNode] = node_info["arxiv_nodes"]  # type: ignore[assignment]
        self._selection: Optional[str] = None
        self._item_to_key: Dict[str, Tuple[int, int, int, int]] = {}

        columns = ("wiki_i", "wiki_j", "arxiv_a", "arxiv_b", "weight")
        self.tree = ttk.Treeview(self, columns=columns, show="headings", height=12)
        headings = ("Wiki i", "Wiki j", "arXiv a", "arXiv b", "Weight")
        widths = (140, 140, 140, 140, 100)
        for col, title, width in zip(columns, headings, widths):
            self.tree.heading(col, text=title)
            self.tree.column(col, width=width, anchor="center")
        self.tree.pack(fill="both", expand=True, padx=8, pady=8)
        self.tree.bind("<<TreeviewSelect>>", self._on_select)

        form = ttk.Frame(self, style="Retro.TFrame")
        form.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(form, text="New weight:").pack(side="left")
        self.weight_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.weight_var, width=12).pack(side="left", padx=6)
        retro_button(form, text="Update", command=self._apply_change).pack(side="left", padx=4)
        retro_button(form, text="Close", command=self.destroy).pack(side="right")

        self._populate_rows()

    def _populate_rows(self) -> None:
        self.tree.delete(*self.tree.get_children())
        for key, value in self.weights.items():
            wiki_i, wiki_j, arxiv_a, arxiv_b = key
            wiki_i_label = self.wiki_nodes[wiki_i].label if wiki_i < len(self.wiki_nodes) else str(wiki_i)
            wiki_j_label = self.wiki_nodes[wiki_j].label if wiki_j < len(self.wiki_nodes) else str(wiki_j)
            arxiv_a_label = self.arxiv_nodes[arxiv_a].label if arxiv_a < len(self.arxiv_nodes) else str(arxiv_a)
            arxiv_b_label = self.arxiv_nodes[arxiv_b].label if arxiv_b < len(self.arxiv_nodes) else str(arxiv_b)
            item_id = self.tree.insert(
                "",
                tk.END,
                values=(wiki_i_label, wiki_j_label, arxiv_a_label, arxiv_b_label, f"{float(value):.4f}"),
            )
            self._item_to_key[item_id] = key

    def _on_select(self, _: tk.Event) -> None:
        selection = self.tree.selection()
        if not selection:
            self._selection = None
            return
        self._selection = selection[0]
        self.weight_var.set(self.tree.set(self._selection, "weight"))

    def _apply_change(self) -> None:
        if not self._selection:
            messagebox.showwarning("Select a row", "Choose a structural pair to edit.")
            return
        try:
            new_value = float(self.weight_var.get())
        except ValueError:
            messagebox.showerror("Invalid value", "Please enter a number.")
            return

        key = self._item_to_key.get(self._selection)
        if key is None:
            messagebox.showerror("Internal error", "Could not determine the selected entry.")
            return
        self.weights[key] = new_value
        self.tree.set(self._selection, "weight", f"{new_value:.4f}")


class DemoApp(ttk.Frame):
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, style="Retro.TFrame", padding=4)
        self.runner = ManualPipelineRunner()
        self.status_var = tk.StringVar(value="Ready")
        self.running = False
        self._art_window: Optional[tk.Toplevel] = None
        self._ascii_art = (
            QUAK_ASCII_MASCOT.rstrip("\n") + "\n" + QUAK_WORDMARK.rstrip("\n")
        )
        self.node_info: Optional[Dict[str, object]] = None
        self.structural_info: Optional[Dict[str, object]] = None
        self.prep_logs: List[str] = []
        self._node_editor: Optional[NodeMatrixEditor] = None
        self._struct_editor: Optional[StructuralMatrixEditor] = None

        self.pack(fill="both", expand=True)
        self.winfo_toplevel().title("Handmade KG Alignment Demo")

        # --- Knowledge Input collapsible panel (top) ---
        self._ki_collapsed = False
        ki_header = ttk.Frame(self, style="Retro.TFrame")
        ki_header.pack(fill="x", padx=8, pady=(8, 0))
        ttk.Label(ki_header, text="Knowledge Input", font=app_font(12, "bold")).pack(side="left")
        self._ki_toggle_text = tk.StringVar(value="▾")

        def _ki_toggle() -> None:
            self._ki_collapsed = not self._ki_collapsed
            self._ki_toggle_text.set("▸" if self._ki_collapsed else "▾")
            if self._ki_collapsed:
                self._ki_content.pack_forget()
            else:
                # Attempt to re-pack the content above the main paned window
                if hasattr(self, "_main_paned"):
                    try:
                        self._ki_content.pack(fill="x", padx=8, pady=(0, 4), before=self._main_paned)
                    except Exception:
                        self._ki_content.pack(fill="x", padx=8, pady=(0, 4))
                    # ensure the paned sash leaves room for the bottom pane
                    try:
                        paned = self._main_paned
                        paned.update_idletasks()
                        total_h = paned.winfo_height() or 0
                        # keep at least this height for the bottom pane
                        min_bottom = 160
                        desired_sash = max(80, int(max(0, total_h) - min_bottom))
                        try:
                            paned.sashpos(0, desired_sash)
                        except Exception:
                            pass
                    except Exception:
                        pass
                else:
                    self._ki_content.pack(fill="x", padx=8, pady=(0, 4))

        retro_button(ki_header, textvariable=self._ki_toggle_text, command=_ki_toggle).pack(side="right")

        self._ki_content = ttk.Frame(self, style="Retro.TFrame")
        self._ki_content.pack(fill="x", padx=8, pady=(0, 4))

        # mode selector
        mode_row = ttk.Frame(self._ki_content, style="Retro.TFrame")
        mode_row.pack(fill="x", pady=(6, 4))
        self._ki_mode = tk.StringVar(value="manual")
        ttk.Radiobutton(mode_row, text="Manual mode", value="manual", variable=self._ki_mode, command=lambda: self._ki_switch_mode()).pack(side="left")
        ttk.Radiobutton(mode_row, text="Crawl mode", value="crawl", variable=self._ki_mode, command=lambda: self._ki_switch_mode()).pack(side="left", padx=(8, 0))

        # manual mode widgets
        self._ki_manual_frame = ttk.Frame(self._ki_content, style="Retro.TFrame")
        ttk.Label(self._ki_manual_frame, text="Wiki text:").pack(anchor="w")
        self._ki_manual_wiki = tk.Text(self._ki_manual_frame, height=4, background=RETRO_CANVAS_BG)
        self._ki_manual_wiki.pack(fill="x", pady=(0, 4))
        ttk.Label(self._ki_manual_frame, text="arXiv text:").pack(anchor="w")
        self._ki_manual_arxiv = tk.Text(self._ki_manual_frame, height=4, background=RETRO_CANVAS_BG)
        self._ki_manual_arxiv.pack(fill="x", pady=(0, 4))

        # crawl mode widgets
        self._ki_crawl_frame = ttk.Frame(self._ki_content, style="Retro.TFrame")
        crawl_row = ttk.Frame(self._ki_crawl_frame, style="Retro.TFrame")
        crawl_row.pack(fill="x")
        ttk.Label(crawl_row, text="Wikipedia URL:").grid(row=0, column=0, sticky="w")
        self._ki_wiki_url = ttk.Entry(crawl_row, width=64)
        self._ki_wiki_url.grid(row=0, column=1, sticky="w", padx=(6, 0))
        ttk.Label(crawl_row, text="arXiv URL:").grid(row=1, column=0, sticky="w")
        self._ki_arxiv_url = ttk.Entry(crawl_row, width=64)
        self._ki_arxiv_url.grid(row=1, column=1, sticky="w", padx=(6, 0))
        fetch_row = ttk.Frame(self._ki_crawl_frame, style="Retro.TFrame")
        fetch_row.pack(fill="x", pady=(6, 0))
        retro_button(fetch_row, text="Fetch", command=self._ki_fetch).pack(side="left")
        ttk.Label(fetch_row, text="(fetched text will appear below for editing)").pack(side="left", padx=(8, 0))

        ttk.Label(self._ki_crawl_frame, text="Fetched / editable Wiki text:").pack(anchor="w", pady=(6, 0))
        self._ki_crawl_wiki_text = tk.Text(self._ki_crawl_frame, height=4, background=RETRO_CANVAS_BG)
        self._ki_crawl_wiki_text.pack(fill="x", pady=(0, 4))
        ttk.Label(self._ki_crawl_frame, text="Fetched / editable arXiv text:").pack(anchor="w", pady=(6, 0))
        self._ki_crawl_arxiv_text = tk.Text(self._ki_crawl_frame, height=4, background=RETRO_CANVAS_BG)
        self._ki_crawl_arxiv_text.pack(fill="x", pady=(0, 4))

        # action buttons
        action_row = ttk.Frame(self._ki_content, style="Retro.TFrame")
        action_row.pack(fill="x", pady=(6, 4))
        self._ki_use_llm = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            action_row,
            text="Use LLM extraction (Groq)",
            variable=self._ki_use_llm,
        ).pack(side="left", padx=(0, 8))
        retro_button(action_row, text="Create Graphs", command=self._ki_create_graphs).pack(side="left")
        retro_button(action_row, text="Save Graphs", command=self._ki_save_graphs).pack(side="left", padx=6)
        retro_button(action_row, text="Display Graphs", command=self._ki_display_graphs).pack(side="left", padx=6)

        # initialize in manual mode
        self._ki_manual_frame.pack(fill="x")

        layout = ttk.PanedWindow(self, orient="vertical")
        # keep a reference so we can re-pack the Knowledge Input above it after collapse
        self._main_paned = layout
        layout.pack(fill="both", expand=True)
        top_section = ttk.Frame(layout, style="Retro.TFrame")
        bottom_section = ttk.Frame(layout, style="Retro.TFrame")
        layout.add(top_section, weight=3)
        layout.add(bottom_section, weight=2)

        panels = ttk.Frame(top_section, style="Retro.TFrame")
        panels.pack(fill="both", padx=8, pady=8)
        self.wiki_panel = GraphPanel(panels, "Wiki Graph", "#38bdf8")
        self.arxiv_panel = GraphPanel(panels, "arXiv Graph", "#fcd34d")
        self.wiki_panel.pack(side="left", expand=True, fill="both")
        self.arxiv_panel.pack(side="left", expand=True, fill="both")

        controls = ttk.Frame(top_section, style="Retro.TFrame")
        controls.pack(fill="x", padx=8, pady=(0, 4))
        self.nn_threshold_var = tk.StringVar(value=f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")
        self.qubo_threshold_var = tk.StringVar(value=f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")
        self.embedding_mode_var = tk.StringVar(value=EmbeddingMode.default().value)
        self.node_weight_var = tk.StringVar(value=f"{QUBO_NODE_WEIGHT:.2f}")
        self.structure_weight_var = tk.StringVar(value=f"{QUBO_STRUCTURE_WEIGHT:.2f}")

        pipeline_row = ttk.Frame(controls, style="Retro.TFrame")
        pipeline_row.pack(fill="x", pady=(0, 4))
        retro_button(
            pipeline_row,
            text="Generate Embeddings",
            command=self._prepare_embeddings,
        ).pack(side="left")
        retro_button(
            pipeline_row,
            text="Run Alignment",
            command=self._run_alignment,
        ).pack(side="left", padx=6)
        ttk.Label(pipeline_row, textvariable=self.status_var).pack(side="left", padx=12)
        retro_button(
            pipeline_row,
            text="Show Q.U.A.K.",
            command=self._show_ascii_art,
        ).pack(side="right")

        threshold_row = ttk.LabelFrame(controls, text="Similarity thresholds (0-1)", style="Retro.TLabelframe")
        threshold_row.pack(fill="x", pady=(0, 4))
        ttk.Label(threshold_row, text="Nearest Neighbor ≥").grid(row=0, column=0, sticky="w")
        ttk.Entry(threshold_row, textvariable=self.nn_threshold_var, width=8).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(threshold_row, text="QUBO ≥").grid(row=0, column=2, sticky="w")
        ttk.Entry(threshold_row, textvariable=self.qubo_threshold_var, width=8).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(
            threshold_row,
            text="Leave blank to disable filtering for either solver.",
            foreground="#475569",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        threshold_row.columnconfigure(4, weight=1)

        weight_row = ttk.LabelFrame(controls, text="QUBO weights", style="Retro.TLabelframe")
        weight_row.pack(fill="x", pady=(0, 4))
        ttk.Label(weight_row, text="Node weight").grid(row=0, column=0, sticky="w")
        ttk.Entry(weight_row, textvariable=self.node_weight_var, width=8).grid(row=0, column=1, padx=(4, 16))
        ttk.Label(weight_row, text="Structure weight").grid(row=0, column=2, sticky="w")
        ttk.Entry(weight_row, textvariable=self.structure_weight_var, width=8).grid(row=0, column=3, padx=(4, 0))
        ttk.Label(
            weight_row,
            text="Weights must be ≥ 0 and influence the QUBO scoring.",
            foreground="#475569",
        ).grid(row=1, column=0, columnspan=4, sticky="w", pady=(4, 0))
        weight_row.columnconfigure(4, weight=1)

        embedding_row = ttk.LabelFrame(controls, text="Embedding mode", style="Retro.TLabelframe")
        embedding_row.pack(fill="x", pady=(0, 4))
        ttk.Label(embedding_row, text="Entity embeddings:").grid(row=0, column=0, sticky="w")
        embedding_choices = [mode.value for mode in EmbeddingMode]
        mode_combo = ttk.Combobox(
            embedding_row,
            textvariable=self.embedding_mode_var,
            values=embedding_choices,
            state="readonly",
            width=32,
        )
        mode_combo.grid(row=0, column=1, sticky="w", padx=(6, 0))
        mode_combo.bind("<<ComboboxSelected>>", self._on_embedding_mode_change)
        ttk.Label(
            embedding_row,
            text="SciBERT-only, hybrid (SciBERT + GNN), or pure GNN-GAEA",
            foreground="#475569",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(4, 0))
        embedding_row.columnconfigure(2, weight=1)

        editor_row = ttk.Frame(controls, style="Retro.TFrame")
        editor_row.pack(fill="x", pady=(0, 4))
        retro_button(
            editor_row,
            text="Edit H_node Matrix",
            command=self._open_node_editor,
        ).pack(side="left")
        retro_button(
            editor_row,
            text="Edit H_structure Matrix",
            command=self._open_structural_editor,
        ).pack(side="left", padx=6)

        experience_row = ttk.Frame(controls, style="Retro.TFrame")
        experience_row.pack(fill="x")
        retro_button(
            experience_row,
            text="Save Experience",
            command=self._save_experience,
        ).pack(side="left")
        retro_button(
            experience_row,
            text="Load Experience",
            command=self._load_experience,
        ).pack(side="left", padx=6)

        self.results = ResultsPanel(bottom_section, ascii_art=self._ascii_art)
        self.results.pack(fill="both", expand=True, padx=8, pady=(0, 8))

    # ------------------------------------------------------------------
    def _current_embedding_mode(self) -> EmbeddingMode:
        value = self.embedding_mode_var.get().strip()
        try:
            return EmbeddingMode.from_value(value)
        except ValueError:
            return EmbeddingMode.default()

    def _on_embedding_mode_change(self, _: Optional[tk.Event] = None) -> None:
        self.node_info = None
        self.structural_info = None
        self.prep_logs = []
        self._close_node_editor()
        self._close_struct_editor()
        if hasattr(self, "results"):
            self.results.show_logs([
                "Embedding mode changed. Run 'Generate Embeddings' to rebuild matrices.",
            ])
        self.status_var.set("Embedding mode changed. Regenerate embeddings before running alignment.")

    # ---------------- Knowledge Input helpers ----------------
    def _ki_switch_mode(self) -> None:
        mode = self._ki_mode.get()
        if mode == "manual":
            self._ki_crawl_frame.pack_forget()
            self._ki_manual_frame.pack(fill="x")
        else:
            self._ki_manual_frame.pack_forget()
            self._ki_crawl_frame.pack(fill="x")

    def _ki_fetch(self) -> None:
        # fetch both URLs (best-effort) and place plain text in editable boxes
        wiki_url = self._ki_wiki_url.get().strip()
        arxiv_url = self._ki_arxiv_url.get().strip()

        def _simple_fetch(url: str) -> str:
            if not url:
                return ""
            try:
                req = urllib.request.Request(url, headers={"User-Agent": "python-urllib/3"})
                with urllib.request.urlopen(req, timeout=15) as resp:
                    raw = resp.read().decode(errors="ignore")
                # strip HTML tags naively
                text = unescape(raw)
                import re

                text = re.sub(r"<script.*?</script>", "", text, flags=re.S | re.I)
                text = re.sub(r"<style.*?</style>", "", text, flags=re.S | re.I)
                text = re.sub(r"<[^>]+>", "\n", text)
                # collapse whitespace
                text = re.sub(r"\n{2,}", "\n\n", text)
                text = text.strip()
                return text
            except Exception as exc:
                messagebox.showerror("Fetch failed", f"Failed to fetch {url}: {exc}")
                return ""

        # run in background thread
        def task() -> None:
            wiki_text = _simple_fetch(wiki_url) if wiki_url else ""
            arxiv_text = _simple_fetch(arxiv_url) if arxiv_url else ""
            self.after(0, lambda: self._ki_crawl_wiki_text.delete("1.0", tk.END) or self._ki_crawl_wiki_text.insert(tk.END, wiki_text))
            self.after(0, lambda: self._ki_crawl_arxiv_text.delete("1.0", tk.END) or self._ki_crawl_arxiv_text.insert(tk.END, arxiv_text))

        threading.Thread(target=task, daemon=True).start()

    def _ki_create_graphs(self) -> None:
        # take text from the active mode and create two GraphModel objects
        if self._ki_mode.get() == "manual":
            wiki_text = self._ki_manual_wiki.get("1.0", tk.END).strip()
            arxiv_text = self._ki_manual_arxiv.get("1.0", tk.END).strip()
        else:
            wiki_text = self._ki_crawl_wiki_text.get("1.0", tk.END).strip()
            arxiv_text = self._ki_crawl_arxiv_text.get("1.0", tk.END).strip()

        if not wiki_text and not arxiv_text:
            messagebox.showinfo("No input", "Provide some text for Wiki and/or arXiv before creating graphs.")
            return

        use_llm = getattr(self, "_ki_use_llm", tk.BooleanVar(value=False)).get()

        def _assign_positions(model: GraphModel) -> None:
            nodes = model.list_nodes()
            for i, node in enumerate(nodes):
                x = random.randint(NODE_RADIUS + 10, CANVAS_WIDTH - NODE_RADIUS - 10)
                y = int((i + 1) * (CANVAS_HEIGHT / max(1, len(nodes))))
                node.position = (x, min(max(y, NODE_RADIUS), CANVAS_HEIGHT - NODE_RADIUS))

        def _build_graph_from_nlp(text: str, name: str, pipeline: NLPPipeline) -> GraphModel:
            model = GraphModel(name)
            if not text.strip():
                return model
            try:
                triples, entities = pipeline.extract_triples(text, return_entities=True)
            except Exception:
                triples = pipeline.extract_triples(text)
                entities = []

            created: Dict[str, str] = {}
            if entities:
                for ent in entities:
                    key = ent.strip()
                    if not key:
                        continue
                    node = model.add_node(key)
                    created[key] = node.id

            for t in triples:
                subj = t.subject.strip()
                obj = t.object.strip()
                if subj and subj not in created:
                    node = model.add_node(subj)
                    created[subj] = node.id
                if obj and obj not in created:
                    node = model.add_node(obj)
                    created[obj] = node.id
                if subj in created and obj in created:
                    rel = t.relation if hasattr(t, "relation") else getattr(t, "relation", "related_to")
                    model.add_edge(created[subj], created[obj], rel)

            _assign_positions(model)
            return model

        def _build_graph_from_llm(entities, triples, name: str) -> GraphModel:
            model = GraphModel(name)
            created: Dict[str, str] = {}
            for ent in (entities or []):
                nm = getattr(ent, "name", str(ent)).strip()
                if not nm:
                    continue
                node = model.add_node(nm)
                created[nm] = node.id

            for t in (triples or []):
                subj = getattr(t, "subject", str(t.get("subject") if isinstance(t, dict) else "")).strip()
                obj = getattr(t, "object", str(t.get("object") if isinstance(t, dict) else "")).strip()
                rel = getattr(t, "relation", str(t.get("relation") if isinstance(t, dict) else "related_to")).strip()
                if subj and subj not in created:
                    node = model.add_node(subj)
                    created[subj] = node.id
                if obj and obj not in created:
                    node = model.add_node(obj)
                    created[obj] = node.id
                if subj in created and obj in created:
                    model.add_edge(created[subj], created[obj], rel or "related_to")

            _assign_positions(model)
            return model

        def task() -> None:
            nlp_pipeline = NLPPipeline()
            wiki_graph = GraphModel("Wiki Graph")
            arxiv_graph = GraphModel("arXiv Graph")

            if use_llm:
                try:
                    llm = LLMBasedPipeline()
                except Exception as exc:
                    # LLM unavailable; warn and fall back to NLP pipeline
                    self.after(0, lambda e=exc: messagebox.showwarning("LLM unavailable", f"LLM initialization failed; ({e})"))
                    use_local_llm = False
                    return;
                else:
                    use_local_llm = True

                if use_local_llm:
                    try:
                        if wiki_text:
                            wiki_ex = llm._process_corpus("wiki", [wiki_text])
                            wiki_entities, wiki_triples = llm._canonicalize_corpus("wiki", wiki_ex)
                            wiki_graph = _build_graph_from_llm(wiki_entities, wiki_triples, "Wiki Graph")
                        if arxiv_text:
                            arxiv_ex = llm._process_corpus("arxiv", [arxiv_text])
                            arxiv_entities, arxiv_triples = llm._canonicalize_corpus("arxiv", arxiv_ex)
                            arxiv_graph = _build_graph_from_llm(arxiv_entities, arxiv_triples, "arXiv Graph")
                    except Exception as exc:
                        # If LLM extraction fails mid-way, warn the user, and halt the graph generation, don't fall back
                        self.after(0, lambda e=exc: messagebox.showerror("LLM extraction failed", f"LLM-based graph extraction failed: {e}"))
                        return

            # if not using LLM or fallback occurred, use NLPPipeline
            if not use_llm or (use_llm and not 'use_local_llm' in locals()) or (use_llm and 'use_local_llm' in locals() and not use_local_llm):
                try:
                    if wiki_text:
                        wiki_graph = _build_graph_from_nlp(wiki_text, "Wiki Graph", nlp_pipeline)
                    if arxiv_text:
                        arxiv_graph = _build_graph_from_nlp(arxiv_text, "arXiv Graph", nlp_pipeline)
                except Exception as exc:
                    self.after(0, lambda: messagebox.showerror("Failed", str(exc)))
                    return

            # store graphs on main thread
            self.after(0, lambda: self._on_graphs_created(wiki_graph, arxiv_graph))

        threading.Thread(target=task, daemon=True).start()

    def _on_graphs_created(self, wiki_graph: GraphModel, arxiv_graph: GraphModel) -> None:
        # set as current models but don't overwrite user's panels until they ask
        self._ki_last_wiki = wiki_graph
        self._ki_last_arxiv = arxiv_graph
        messagebox.showinfo("Graphs created", "Knowledge graphs created. Use 'Display Graphs' to show them in the panels, or 'Save Graphs' to persist them.")

    def _ki_save_graphs(self) -> None:
        if not getattr(self, "_ki_last_wiki", None) and not getattr(self, "_ki_last_arxiv", None):
            messagebox.showinfo("Nothing to save", "Create graphs first before attempting to save them.")
            return
        if getattr(self, "_ki_last_wiki", None):
            path = storage.save_graph(self._ki_last_wiki)
            messagebox.showinfo("Saved", f"Saved Wiki graph to {path}")
        if getattr(self, "_ki_last_arxiv", None):
            path = storage.save_graph(self._ki_last_arxiv)
            messagebox.showinfo("Saved", f"Saved arXiv graph to {path}")

    def _ki_display_graphs(self) -> None:
        if getattr(self, "_ki_last_wiki", None):
            self.wiki_panel.set_model(self._ki_last_wiki)
        if getattr(self, "_ki_last_arxiv", None):
            self.arxiv_panel.set_model(self._ki_last_arxiv)
        messagebox.showinfo("Displayed", "Displayed created graphs in the Wiki and arXiv panels.")

    # ------------------------------------------------------------------
    def _prepare_embeddings(self) -> None:
        if self.running:
            return
        self.running = True
        mode = self._current_embedding_mode()
        self.status_var.set(f"Preparing embeddings ({mode.value})…")

        wiki_copy = copy.deepcopy(self.wiki_panel.model)
        arxiv_copy = copy.deepcopy(self.arxiv_panel.model)

        def task() -> None:
            try:
                node_info, structural_info, logs = self.runner.prepare_inputs(
                    wiki_copy,
                    arxiv_copy,
                    embedding_mode=mode,
                )
            except Exception as exc:
                self.after(0, lambda err=exc: self._handle_error(err))
            else:
                self.after(0, lambda: self._store_prepared_inputs(node_info, structural_info, logs))
            finally:
                self.after(0, self._reset_status)

        threading.Thread(target=task, daemon=True).start()

    def _store_prepared_inputs(
        self,
        node_info: Dict[str, object],
        structural_info: Dict[str, object],
        logs: List[str],
    ) -> None:
        self._close_node_editor()
        self._close_struct_editor()
        self.node_info = node_info
        self.structural_info = structural_info
        self.prep_logs = list(logs)
        display_logs = logs + ["Matrices ready. Edit H_node/H_structure or run alignment."]
        self.results.show_logs(display_logs)

    def _run_alignment(self) -> None:
        if self.running:
            return
        if self.node_info is None or self.structural_info is None:
            messagebox.showinfo(
                "Missing matrices",
                "Run 'Generate Embeddings' first to build the similarity and structural matrices.",
            )
            return
        try:
            nn_threshold = self._parse_threshold_value(self.nn_threshold_var.get(), "Nearest Neighbor")
            qubo_threshold = self._parse_threshold_value(self.qubo_threshold_var.get(), "QUBO")
        except ValueError as exc:
            messagebox.showerror("Invalid threshold", str(exc))
            return
        try:
            node_weight, structure_weight = self._current_qubo_weights()
        except ValueError as exc:
            messagebox.showerror("Invalid QUBO weight", str(exc))
            return
        self.running = True
        self.status_var.set("Running alignment solvers…")

        node_info = self.node_info
        structural_info = self.structural_info
        logs_input = list(self.prep_logs) if self.prep_logs else None

        def task() -> None:
            try:
                result = self.runner.solve_from_inputs(
                    node_info,
                    structural_info,
                    logs_input,
                    node_weight=node_weight,
                    structure_weight=structure_weight,
                    nn_threshold=nn_threshold,
                    qubo_threshold=qubo_threshold,
                )
            except Exception as exc:
                self.after(0, lambda err=exc: self._handle_error(err))
            else:
                self.after(0, lambda: self._display_result(result))
            finally:
                self.after(0, self._reset_status)

        threading.Thread(target=task, daemon=True).start()

    def _handle_error(self, exc: Exception) -> None:
        messagebox.showerror("Pipeline failed", str(exc))

    def _display_result(self, result: PipelineResult) -> None:
        self.prep_logs = list(result.logs)
        self.results.display_result(result)

    def _parse_threshold_value(self, raw_value: str, label: str) -> Optional[float]:
        value = raw_value.strip()
        if not value:
            return None
        try:
            parsed = float(value)
        except ValueError as exc:  # pragma: no cover - GUI validation
            raise ValueError(f"{label} threshold must be a number between 0 and 1.") from exc
        if not 0.0 <= parsed <= 1.0:
            raise ValueError(f"{label} threshold must be between 0 and 1.")
        return parsed

    def _parse_weight_value(self, raw_value: str, label: str) -> float:
        value = raw_value.strip()
        if not value:
            raise ValueError(f"{label} cannot be blank.")
        try:
            parsed = float(value)
        except ValueError as exc:  # pragma: no cover - GUI validation
            raise ValueError(f"{label} must be a number.") from exc
        if parsed < 0.0:
            raise ValueError(f"{label} must be non-negative.")
        return parsed

    def _current_qubo_weights(self) -> Tuple[float, float]:
        node_weight = self._parse_weight_value(self.node_weight_var.get(), "QUBO node weight")
        structure_weight = self._parse_weight_value(
            self.structure_weight_var.get(),
            "QUBO structure weight",
        )
        return node_weight, structure_weight

    def _open_node_editor(self) -> None:
        if self.node_info is None:
            messagebox.showinfo("No similarity matrix", "Generate embeddings before editing H_node.")
            return
        if self._node_editor and self._node_editor.winfo_exists():
            self._node_editor.lift()
            self._node_editor.focus_force()
            return
        editor = NodeMatrixEditor(self, self.node_info)

        def _cleanup() -> None:
            self._node_editor = None
            editor.destroy()

        editor.protocol("WM_DELETE_WINDOW", _cleanup)
        self._node_editor = editor

    def _open_structural_editor(self) -> None:
        if self.node_info is None or self.structural_info is None:
            messagebox.showinfo("No structural weights", "Generate embeddings to build structural weights before editing.")
            return
        if self._struct_editor and self._struct_editor.winfo_exists():
            self._struct_editor.lift()
            self._struct_editor.focus_force()
            return
        editor = StructuralMatrixEditor(self, self.node_info, self.structural_info)

        def _cleanup() -> None:
            self._struct_editor = None
            editor.destroy()

        editor.protocol("WM_DELETE_WINDOW", _cleanup)
        self._struct_editor = editor

    def _close_node_editor(self) -> None:
        if self._node_editor and self._node_editor.winfo_exists():
            self._node_editor.destroy()
        self._node_editor = None

    def _close_struct_editor(self) -> None:
        if self._struct_editor and self._struct_editor.winfo_exists():
            self._struct_editor.destroy()
        self._struct_editor = None

    def _save_experience(self) -> None:
        if self.node_info is None or self.structural_info is None:
            messagebox.showinfo(
                "Nothing to save",
                "Generate embeddings (and optionally edit matrices) before saving an experience.",
            )
            return
        initial = storage.EXPERIENCE_DIR / "experience.json"
        path = filedialog.asksaveasfilename(
            title="Save Experience",
            defaultextension=".json",
            initialdir=storage.EXPERIENCE_DIR,
            initialfile=initial.name,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            node_weight, structure_weight = self._current_qubo_weights()
        except ValueError as exc:
            messagebox.showerror("Invalid QUBO weight", str(exc))
            return
        saved_path = storage.save_experience(
            self.wiki_panel.model,
            self.arxiv_panel.model,
            self.node_info,
            self.structural_info,
            filename=Path(path),
            qubo_weights={
                "node_weight": node_weight,
                "structure_weight": structure_weight,
            },
        )
        messagebox.showinfo("Experience saved", f"Saved matrices and graphs to {saved_path}")

    def _load_experience(self) -> None:
        path = filedialog.askopenfilename(
            title="Load Experience",
            initialdir=storage.EXPERIENCE_DIR,
            filetypes=[("JSON", "*.json")],
        )
        if not path:
            return
        try:
            payload = storage.load_experience(Path(path))
        except Exception as exc:
            messagebox.showerror("Failed to load experience", str(exc))
            return

        wiki_graph: GraphModel = payload["wiki_graph"]
        arxiv_graph: GraphModel = payload["arxiv_graph"]
        wiki_order = payload.get("wiki_order", [])
        arxiv_order = payload.get("arxiv_order", [])
        similarity_payload = payload.get("similarity", [])
        structural_payload = payload.get("structural_info", {})
        qubo_weights_payload = cast(Optional[Dict[str, Any]], payload.get("qubo_weights")) or {}

        self.wiki_panel.set_model(wiki_graph)
        self.arxiv_panel.set_model(arxiv_graph)

        wiki_nodes = self._order_nodes(wiki_graph, wiki_order)
        arxiv_nodes = self._order_nodes(arxiv_graph, arxiv_order)
        similarity_tensor = self._build_similarity_tensor(similarity_payload, len(wiki_nodes), len(arxiv_nodes))
        weights_payload = cast(Sequence[Dict[str, Any]], structural_payload.get("weights", []))
        weights = self._build_structural_weights(weights_payload)
        structural_info = {
            "wiki_edges": structural_payload.get("wiki_edges", []),
            "arxiv_edges": structural_payload.get("arxiv_edges", []),
            "weights": weights,
        }

        def _coerce_weight(value: Any, default: float) -> float:
            try:
                return float(value)
            except (TypeError, ValueError):
                return default

        node_weight_value = _coerce_weight(qubo_weights_payload.get("node_weight"), QUBO_NODE_WEIGHT)
        structure_weight_value = _coerce_weight(
            qubo_weights_payload.get("structure_weight"),
            QUBO_STRUCTURE_WEIGHT,
        )
        self.node_weight_var.set(f"{node_weight_value:.2f}")
        self.structure_weight_var.set(f"{structure_weight_value:.2f}")

        self.node_info = {
            "wiki_nodes": wiki_nodes,
            "arxiv_nodes": arxiv_nodes,
            "similarity": similarity_tensor,
        }
        self.structural_info = structural_info
        self.prep_logs = [f"Experience loaded from {Path(path).name}"]
        self.results.show_logs(self.prep_logs)
        messagebox.showinfo("Experience loaded", "Matrices restored. You can edit them or run alignment immediately.")

    def _order_nodes(self, graph: GraphModel, order: Sequence[str]) -> List[GraphNode]:
        lookup = graph.nodes
        ordered: List[GraphNode] = []
        seen = set()
        for node_id in order:
            node = lookup.get(node_id)
            if node is None:
                continue
            ordered.append(node)
            seen.add(node_id)
        if len(ordered) == len(lookup):
            return ordered
        remaining = [node for node in graph.list_nodes() if node.id not in seen]
        remaining.sort(key=lambda item: item.label.lower())
        return ordered + remaining

    def _build_similarity_tensor(
        self,
        payload: Sequence[Sequence[float]],
        wiki_count: int,
        arxiv_count: int,
    ) -> torch.Tensor:
        similarity = torch.zeros((wiki_count, arxiv_count), dtype=torch.float)
        if not payload:
            return similarity
        try:
            matrix = torch.tensor(payload, dtype=torch.float)
        except Exception:
            return similarity
        rows = min(matrix.size(0), wiki_count)
        cols = min(matrix.size(1), arxiv_count)
        if rows and cols:
            similarity[:rows, :cols] = matrix[:rows, :cols]
        return similarity

    def _build_structural_weights(
        self,
        payload: Sequence[Dict[str, Any]],
    ) -> Dict[Tuple[int, int, int, int], float]:
        weights: Dict[Tuple[int, int, int, int], float] = {}
        for entry in payload:
            try:
                idx = (
                    int(entry["wiki_i"]),
                    int(entry["wiki_j"]),
                    int(entry["arxiv_a"]),
                    int(entry["arxiv_b"]),
                )
                weight = float(entry["weight"])
            except (KeyError, TypeError, ValueError):
                continue
            weights[idx] = weight
        return weights

    def _reset_status(self) -> None:
        self.running = False
        self.status_var.set("Ready")

    def _show_ascii_art(self) -> None:
        if self._art_window and self._art_window.winfo_exists():
            self._art_window.lift()
            self._art_window.focus_force()
            return

        window = tk.Toplevel(self)
        window.title("Q.U.A.K. Mascot")
        window.configure(padx=12, pady=12, bg=RETRO_BG)
        window.resizable(False, False)

        label = ttk.Label(
            window,
            text="Quadratic Unconstrained Alignment of Knowledge (or Quantum Utility Alignment Kit)",
            font=app_font(12, "bold"),
        )
        label.pack(anchor="center", pady=(0, 8))

        text = tk.Text(
            window,
            width=50,
            height=len(self._ascii_art.splitlines()) + 1,
            font=app_font(10),
            background=RETRO_CANVAS_BG,
            foreground=RETRO_TEXT,
            borderwidth=2,
            relief="sunken",
            highlightthickness=0,
        )
        text.insert("1.0", self._ascii_art + "\n")
        text.configure(state="disabled")
        text.pack()

        retro_button(window, text="Close", command=window.destroy).pack(pady=(10, 0))

        def _on_close() -> None:
            self._art_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", _on_close)
        self._art_window = window


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    root.configure(bg=RETRO_BG)
    configure_retro_style(style)
    for font_name in (
        "TkDefaultFont",
        "TkTextFont",
        "TkMenuFont",
        "TkHeadingFont",
        "TkCaptionFont",
        "TkSmallCaptionFont",
        "TkMessageBoxFont",
        "TkTooltipFont",
        "TkFixedFont",
        "TkIconFont",
    ):
        try:
            tkfont.nametofont(font_name).configure(family=APP_FONT_FAMILY)
        except tk.TclError:
            continue
    root.option_add("*Font", app_font(10))
    root.option_add("*Background", RETRO_PANEL_BG)
    app = DemoApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()
