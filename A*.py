import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import heapq
import networkx as nx
import matplotlib.pyplot as plt

# ===================== Fungsi Heuristik =====================
def heuristic(node, goal):
    return abs(hash(node) - hash(goal)) % 100

# ===================== Estimasi Biaya dan Waktu =====================
def calculate_cost(distance, mode):
    if mode == "Jalan Kaki":
        return 0
    elif mode == "Motor":
        return (distance // 10) * 1000
    elif mode == "Mobil":
        return (distance // 10) * 2000
    return 0

def estimate_time(distance, mode):
    speed = {"Jalan Kaki": 5000, "Motor": 30000, "Mobil": 40000}.get(mode, 1)
    minutes = int((distance / speed) * 60)
    return f"{minutes} menit"

# ===================== Algoritma A* =====================
def a_star(graph, start, goal, heuristic):
    open_set = []
    heapq.heappush(open_set, (0, start))
    g_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    came_from = {}

    while open_set:
        _, current = heapq.heappop(open_set)
        if current == goal:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)
            return path[::-1], g_score[goal]

        for neighbor, weight in graph.get(current, []):
            tentative_g_score = g_score[current] + weight
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score, neighbor))

    return None, float('inf')

# ===================== Visualisasi =====================
def visualize_graph(graph, path=None):
    G = nx.Graph()
    for node in graph:
        for neighbor, weight in graph[node]:
            G.add_edge(node, neighbor, weight=weight)

    pos = nx.spring_layout(G, seed=42)
    plt.figure(figsize=(12, 8))
    edge_labels = nx.get_edge_attributes(G, 'weight')

    if path:
        path_edges = list(zip(path, path[1:]))
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)
        nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='red', width=2)
    else:
        nx.draw(G, pos, with_labels=True, node_color='lightblue', node_size=1000)

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
    plt.title("Visualisasi Graf Kampus UNIB")
    plt.show()

# ===================== Fungsi Simpan =====================
def save_result_to_file(result_text):
    file_path = filedialog.asksaveasfilename(defaultextension=".txt",
                                             filetypes=[("Text Files", "*.txt")])
    if file_path:
        with open(file_path, "w") as f:
            f.write(result_text)
        messagebox.showinfo("Berhasil", f"Hasil disimpan di {file_path}")

# ===================== Fungsi Cari Jalur =====================
def find_path():
    start = start_combobox.get()
    goal = goal_combobox.get()
    mode = transport_combobox.get()

    if not mode:
        messagebox.showwarning("Peringatan", "Silakan pilih moda transportasi.")
        return

    if start == goal:
        result_label.config(text="Titik awal dan tujuan tidak boleh sama.")
        return

    if start not in graph or goal not in graph:
        messagebox.showerror("Error", "Titik tidak ditemukan di graf.")
        return

    path, cost = a_star(graph, start, goal, heuristic)
    if path:
        biaya = calculate_cost(cost, mode)
        waktu = estimate_time(cost, mode)
        result = f"Mode: {mode}\nJalur: {' -> '.join(path)}\nJarak: {cost} meter\nEstimasi Waktu: {waktu}\nEstimasi Biaya: Rp {biaya}"
        result_label.config(text=result)
        save_btn.config(state="normal")
        visualize_btn.config(state="normal")
        save_btn.result_data = result  # simpan data
        visualize_btn.path = path
    else:
        result_label.config(text="Jalur tidak ditemukan.")
        save_btn.config(state="disabled")
        visualize_btn.config(state="disabled")

def reset_selection():
    start_combobox.set("")
    goal_combobox.set("")
    transport_combobox.set("")
    result_label.config(text="")
    save_btn.config(state="disabled")
    visualize_btn.config(state="disabled")

# ===================== Data Graf =====================
graph = {
    "Pintu Gerbang Depan": [("Pasca Hukum", 200)],
    "Pasca Hukum": [("Pintu Gerbang Depan", 200), ("MAKSI (Ged C)", 400), ("Gedung F", 500)],
    "MAKSI (Ged C)": [("Pasca Hukum", 400), ("Ged. B", 300)],
    "Ged. B": [("MAKSI (Ged C)", 300), ("Ged. A", 200)],
    "Ged. A": [("Ged. B", 200), ("Masjid UNIB", 100)],
    "Masjid UNIB": [("Ged. A", 100)],
    "Gedung F": [("Pasca Hukum", 500), ("Lab. Hukum", 300), ("Ged. I", 200), ("Ged. J", 200), ("Dekanat Pertanian", 200)],
    "Lab. Hukum": [("Gedung F", 100)],
    "Ged. I": [("Gedung F", 150), ("Ged. MM", 150)],
    "Ged. MM": [("Ged. I", 200), ("Ged. MPP", 200)],
    "Ged. MPP": [("Ged. MM", 100), ("Ged. UPT B. Inggris", 100)],
    "Ged. J": [("Gedung F", 100), ("Ged. UPT B. Inggris", 100)],
    "Ged. UPT B. Inggris": [("Ged. J", 100), ("REKTORAT", 150)],
    "Dekanat Pertanian": [("Gedung F", 150), ("Ged. T", 150)],
    "Ged. T": [("Dekanat Pertanian", 150), ("Ged. V", 150)],
    "Ged. V": [("Ged. T", 150), ("Ged. Renper", 150), ("REKTORAT", 150), ("UPT Puskom", 150)],
    "Ged. Renper": [("Ged. V", 150), ("Lab. Agro", 150)],
    "Lab. Agro": [("Ged. Renper", 150), ("Ged. Basic Sains", 150)],
    "Ged. Basic Sains": [("Lab. Agro", 150), ("GKB I", 150), ("Dekanat MIPA", 150)],
    "UPT Puskom": [("Ged. V", 150), ("GKB I", 150)],
    "REKTORAT": [("Ged. UPT B. Inggris", 150), ("Ged. V", 150), ("Dekanat FISIP", 150)],
    "Dekanat FISIP": [("REKTORAT", 150), ("Pintu Gerbang", 150), ("GKB II", 150)],
    "Pintu Gerbang": [("Dekanat FISIP", 150), ("Dekanat Teknik", 150)],
    "Dekanat Teknik": [("Pintu Gerbang", 150), ("Gedung Serba Guna (GSG)", 150)],
    "Gedung Serba Guna (GSG)": [("Dekanat Teknik", 150), ("Stadion Olahraga", 150), ("GKB III", 150), ("Dekanat FKIP", 150)],
    "GKB I": [("UPT Puskom", 150), ("GKB II", 150), ("Ged. Basic Sains", 150)],
    "GKB II": [("GKB I", 150), ("Dekanat FKIP", 150), ("Dekanat FISIP", 150)],
    "Dekanat FKIP": [("GKB II", 150), ("Gedung Serba Guna (GSG)", 150)],
    "GKB V": [("PKM", 150), ("PSPD", 150)],
    "Stadion Olahraga": [("GKB III", 150), ("PSPD", 150)],
    "GKB III": [("Gedung Serba Guna (GSG)", 150)],
    "PKM": [("GKB V", 150)],
    "PSPD": [("GKB V", 150), ("Stadion Olahraga", 150)],
    "Dekanat MIPA": [("Ged. Basic Sains", 150)]
}

# ===================== GUI =====================
root = tk.Tk()
root.title("A* Pathfinding - Peta Kampus UNIB")
root.geometry("750x600")
root.configure(bg="#6699CC")

tk.Label(root, text="Titik Awal").pack()
start_combobox = ttk.Combobox(root, values=list(graph.keys()), width=60)
start_combobox.pack()

tk.Label(root, text="Titik Tujuan").pack()
goal_combobox = ttk.Combobox(root, values=list(graph.keys()), width=60)
goal_combobox.pack()

tk.Label(root, text="Moda Transportasi").pack()
transport_combobox = ttk.Combobox(root, values=["Jalan Kaki", "Motor", "Mobil"], width=60)
transport_combobox.pack()

tk.Button(root, text="Cari Jalur A*", command=find_path, bg="green", fg="white").pack(pady=10)
tk.Button(root, text="Reset", command=reset_selection, bg="red", fg="white").pack()

save_btn = tk.Button(root, text="Simpan Hasil ke File", command=lambda: save_result_to_file(save_btn.result_data), state="disabled")
save_btn.pack(pady=5)

visualize_btn = tk.Button(root, text="Tampilkan Visualisasi Graf", command=lambda: visualize_graph(graph, visualize_btn.path), state="disabled")
visualize_btn.pack()

result_label = tk.Label(root, text="", justify="left", wraplength=700, anchor="w")
result_label.pack(pady=20, fill=tk.BOTH, expand=True)

root.mainloop()
