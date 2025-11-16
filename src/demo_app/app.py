from __future__ import annotations

import copy
import random
import threading
import tkinter as tk
from dataclasses import dataclass
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch

from . import storage
from .graph_model import GraphModel, GraphNode
from .manual_pipeline import ManualPipelineRunner, PipelineResult
from src.config import DEFAULT_SIMILARITY_THRESHOLD
from src.quak import QUAK_ASCII_MASCOT, QUAK_WORDMARK

CANVAS_WIDTH = 420
CANVAS_HEIGHT = 320
NODE_RADIUS = 18


@dataclass
class GraphPanelState:
    model: GraphModel
    color: str


class GraphPanel(ttk.Frame):
    def __init__(self, master: tk.Misc, title: str, color: str) -> None:
        super().__init__(master, padding=8)
        self.panel_state = GraphPanelState(model=GraphModel(title), color=color)
        self._drag_node: Optional[str] = None
        self._drag_offset = (0.0, 0.0)
        self._node_order: List[str] = []

        ttk.Label(self, text=title, font=("Helvetica", 14, "bold"), foreground="#000000").pack(
            anchor="w", pady=(0, 6)
        )

        self.canvas = tk.Canvas(
            self,
            width=CANVAS_WIDTH,
            height=CANVAS_HEIGHT,
            bg="#ffffff",
            highlightthickness=0,
        )
        self.canvas.pack(fill="x", pady=(0, 8))
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        button_row = ttk.Frame(self)
        button_row.pack(fill="x", pady=4)
        ttk.Button(button_row, text="Add Node", command=self._prompt_add_node).pack(side="left", padx=2)
        ttk.Button(button_row, text="Edit Node", command=self._prompt_edit_node).pack(side="left", padx=2)
        ttk.Button(button_row, text="Add Edge", command=self._prompt_add_edge).pack(side="left", padx=2)
        ttk.Button(button_row, text="Save", command=self._save_graph).pack(side="left", padx=2)
        ttk.Button(button_row, text="Load", command=self._load_graph).pack(side="left", padx=2)
        ttk.Button(button_row, text="Clear", command=self._clear_graph).pack(side="left", padx=2)

        lists_frame = ttk.Frame(self)
        lists_frame.pack(fill="both", expand=True)
        self.node_list = tk.Listbox(lists_frame, height=6, background="#0f172a", foreground="#f8fafc")
        self.edge_list = tk.Listbox(lists_frame, height=6, background="#0f172a", foreground="#f8fafc")
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

        ttk.Button(dialog, text="Add", command=submit).grid(row=2, column=0, columnspan=2, pady=8)
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

        ttk.Button(dialog, text="Add", command=submit).grid(row=3, column=0, columnspan=2, pady=8)
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
            self.canvas.create_text(mid_x, mid_y, text=edge.relation, fill="#9ca3af", font=("Helvetica", 9))
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
            self.canvas.create_text(x, y, text=node.label[:6], fill="#111827", font=("Helvetica", 10, "bold"))

    def _refresh_lists(self) -> None:
        selection = self.node_list.curselection()
        selected_id: Optional[str] = None
        if selection:
            index = selection[0]
            if 0 <= index < len(self._node_order):
                selected_id = self._node_order[index]

        self.node_list.delete(0, tk.END)
        self._node_order = []
        for idx, node in enumerate(self.panel_state.model.list_nodes()):
            desc = f" - {node.description}" if node.description else ""
            self.node_list.insert(tk.END, f"{node.label}{desc}")
            self._node_order.append(node.id)
            if selected_id and node.id == selected_id:
                self.node_list.selection_set(idx)
        self.edge_list.delete(0, tk.END)
        for edge in self.panel_state.model.list_edges():
            src = self.panel_state.model.nodes.get(edge.source)
            dst = self.panel_state.model.nodes.get(edge.target)
            src_label = src.label if src else edge.source
            dst_label = dst.label if dst else edge.target
            self.edge_list.insert(tk.END, f"{src_label} --{edge.relation}--> {dst_label}")

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

    def _on_node_list_double_click(self, _: tk.Event) -> None:  # type: ignore[override]
        node_id = self._get_selected_node_id()
        if node_id:
            self._open_node_editor(node_id)

    def _open_node_editor(self, node_id: str) -> None:
        node = self.panel_state.model.nodes.get(node_id)
        if not node:
            messagebox.showerror("Missing node", "Could not find the selected node for editing.")
            return

        dialog = tk.Toplevel(self)
        dialog.title("Edit Node")
        dialog.transient(self.winfo_toplevel())
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

        button_row = ttk.Frame(dialog)
        button_row.grid(row=2, column=0, columnspan=2, pady=8)
        ttk.Button(button_row, text="Save", command=save).pack(side="left", padx=4)
        ttk.Button(button_row, text="Cancel", command=dialog.destroy).pack(side="left", padx=4)
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
    def __init__(self, master: tk.Misc) -> None:
        super().__init__(master, padding=8)
        ttk.Label(self, text="Alignment Results", font=("Helvetica", 14, "bold")).pack(anchor="w")

        self.nn_tree = self._build_tree("Nearest Neighbor Alignments")
        self.qubo_tree = self._build_tree("QUBO Alignments")

        ttk.Label(self, text="Pipeline Log").pack(anchor="w", pady=(8, 0))
        self.log_text = tk.Text(
            self,
            height=10,
            width=80,
            state="disabled",
            background="#0f172a",
            foreground="#f8fafc",
        )
        self.log_text.pack(fill="both", expand=True)

        self.energy_var = tk.StringVar(value="QUBO energy: -")
        ttk.Label(
            self,
            textvariable=self.energy_var,
            font=("Helvetica", 11, "italic"),
            foreground="#f97316",
        ).pack(anchor="w", pady=6)

    def _build_tree(self, title: str) -> ttk.Treeview:
        ttk.Label(self, text=title, font=("Helvetica", 11, "bold")).pack(anchor="w", pady=(8, 0))
        columns = ("wiki", "arxiv", "similarity")
        tree = ttk.Treeview(self, columns=columns, show="headings", height=6)
        tree.heading("wiki", text="Wiki Entity")
        tree.heading("arxiv", text="arXiv Entity")
        tree.heading("similarity", text="Similarity")
        for column in columns:
            tree.column(column, width=160)
        tree.pack(fill="x", pady=(2, 0))
        return tree

    def display_result(self, result: PipelineResult) -> None:
        self._populate_tree(self.nn_tree, result.nn_alignments)
        self._populate_tree(self.qubo_tree, result.qubo_alignments)
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
        self.energy_var.set("QUBO energy: -")
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", tk.END)
        for line in logs:
            self.log_text.insert(tk.END, f"• {line}\n")
        self.log_text.configure(state="disabled")

    def _populate_tree(self, tree: ttk.Treeview, rows) -> None:
        tree.delete(*tree.get_children())
        for row in rows:
            tree.insert("", tk.END, values=row.as_tuple())


class NodeMatrixEditor(tk.Toplevel):
    def __init__(self, master: tk.Misc, node_info: Dict[str, object]) -> None:
        super().__init__(master)
        self.title("Edit H_node Matrix")
        self.geometry("600x420")
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

        form = ttk.Frame(self)
        form.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(form, text="New value:").pack(side="left")
        self.value_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.value_var, width=12).pack(side="left", padx=6)
        ttk.Button(form, text="Update", command=self._apply_change).pack(side="left")
        ttk.Button(form, text="Close", command=self.destroy).pack(side="right")

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
        self.title("Edit H_structure Weights")
        self.geometry("700x420")
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

        form = ttk.Frame(self)
        form.pack(fill="x", padx=8, pady=(0, 8))
        ttk.Label(form, text="New weight:").pack(side="left")
        self.weight_var = tk.StringVar()
        ttk.Entry(form, textvariable=self.weight_var, width=12).pack(side="left", padx=6)
        ttk.Button(form, text="Update", command=self._apply_change).pack(side="left")
        ttk.Button(form, text="Close", command=self.destroy).pack(side="right")

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
        super().__init__(master)
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

        panels = ttk.Frame(self)
        panels.pack(fill="both", padx=8, pady=8)
        self.wiki_panel = GraphPanel(panels, "Wiki Graph", "#38bdf8")
        self.arxiv_panel = GraphPanel(panels, "arXiv Graph", "#fcd34d")
        self.wiki_panel.pack(side="left", expand=True, fill="both")
        self.arxiv_panel.pack(side="left", expand=True, fill="both")

        controls = ttk.Frame(self)
        controls.pack(fill="x", padx=8, pady=(0, 4))
        self.nn_threshold_var = tk.StringVar(value=f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")
        self.qubo_threshold_var = tk.StringVar(value=f"{DEFAULT_SIMILARITY_THRESHOLD:.2f}")

        pipeline_row = ttk.Frame(controls)
        pipeline_row.pack(fill="x", pady=(0, 4))
        ttk.Button(
            pipeline_row,
            text="Generate Embeddings",
            command=self._prepare_embeddings,
        ).pack(side="left")
        ttk.Button(
            pipeline_row,
            text="Run Alignment",
            command=self._run_alignment,
        ).pack(side="left", padx=6)
        ttk.Label(pipeline_row, textvariable=self.status_var).pack(side="left", padx=12)
        ttk.Button(
            pipeline_row,
            text="Show Q.U.A.K.",
            command=self._show_ascii_art,
        ).pack(side="right")

        threshold_row = ttk.LabelFrame(controls, text="Similarity thresholds (0-1)")
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

        editor_row = ttk.Frame(controls)
        editor_row.pack(fill="x", pady=(0, 4))
        ttk.Button(
            editor_row,
            text="Edit H_node Matrix",
            command=self._open_node_editor,
        ).pack(side="left")
        ttk.Button(
            editor_row,
            text="Edit H_structure Weights",
            command=self._open_structural_editor,
        ).pack(side="left", padx=6)

        experience_row = ttk.Frame(controls)
        experience_row.pack(fill="x")
        ttk.Button(
            experience_row,
            text="Save Experience",
            command=self._save_experience,
        ).pack(side="left")
        ttk.Button(
            experience_row,
            text="Load Experience",
            command=self._load_experience,
        ).pack(side="left", padx=6)

        self.results = ResultsPanel(self)
        self.results.pack(fill="both", expand=True, padx=8, pady=(0, 8))

        art_container = ttk.Frame(self, padding=(0, 0, 8, 8))
        art_container.pack(fill="x", side="bottom", anchor="e")
        art_label = tk.Label(
            art_container,
            text=self._ascii_art,
            font=("Courier New", 8),
            justify="right",
            bg="#020617",
            fg="#f8fafc",
        )
        art_label.pack(anchor="e")

    # ------------------------------------------------------------------
    def _prepare_embeddings(self) -> None:
        if self.running:
            return
        self.running = True
        self.status_var.set("Preparing embeddings and structural weights…")

        wiki_copy = copy.deepcopy(self.wiki_panel.model)
        arxiv_copy = copy.deepcopy(self.arxiv_panel.model)

        def task() -> None:
            try:
                node_info, structural_info, logs = self.runner.prepare_inputs(wiki_copy, arxiv_copy)
            except Exception as exc:
                self.after(0, lambda: self._handle_error(exc))
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
                    nn_threshold=nn_threshold,
                    qubo_threshold=qubo_threshold,
                )
            except Exception as exc:
                self.after(0, lambda: self._handle_error(exc))
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
        saved_path = storage.save_experience(
            self.wiki_panel.model,
            self.arxiv_panel.model,
            self.node_info,
            self.structural_info,
            filename=Path(path),
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
        window.configure(padx=12, pady=12)
        window.resizable(False, False)

        label = ttk.Label(
            window,
            text="Quantum Unconstrained Alignment of Knowledge (or Quantum Utility Alignment Kit)",
            font=("Helvetica", 12, "bold"),
        )
        label.pack(anchor="center", pady=(0, 8))

        text = tk.Text(
            window,
            width=50,
            height=len(self._ascii_art.splitlines()) + 1,
            font=("Courier New", 10),
            background="#020617",
            foreground="#f8fafc",
            borderwidth=0,
            highlightthickness=0,
        )
        text.insert("1.0", self._ascii_art + "\n")
        text.configure(state="disabled")
        text.pack()

        ttk.Button(window, text="Close", command=window.destroy).pack(pady=(10, 0))

        def _on_close() -> None:
            self._art_window = None
            window.destroy()

        window.protocol("WM_DELETE_WINDOW", _on_close)
        self._art_window = window


def main() -> None:
    root = tk.Tk()
    style = ttk.Style()
    style.theme_use("clam")
    app = DemoApp(root)
    app.mainloop()


if __name__ == "__main__":
    main()
