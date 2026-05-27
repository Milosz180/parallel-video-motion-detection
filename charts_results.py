import os
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# tworzenie folderu
output_dir = 'charts_results'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# styl wykresu
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
elif 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')

plt.rcParams.update({
    'font.size': 11, 
    'axes.labelsize': 12, 
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10
})

# dane z wariantów programu
labels = [
    'Sekwencyjny\n(Baseline)', 
    'OpenMP\n(12 wątków)', 
    'MPI\n(2 procesy)', 
    'MPI\n(4 procesy)', 
    'MPI\n(8 procesów)', 
    'CUDA / GPU\n(T-API)'
]

# dane dla ball.mp4
time_ball = [1.9368, 6.3346, 0.9243, 0.8626, 1.1146, 2.4714]
fps_ball = [61.98, 18.95, 130.97, 140.37, 108.81, 48.57]

# dane dla traffic.mp4
time_traffic = [24.8835, 61.7319, 7.8768, 6.8665, 7.4782, 38.8272]
fps_traffic = [45.77, 18.45, 144.73, 166.03, 152.51, 29.34]

# kolory
colors = ['#5c6bc0', '#e57373', '#66bb6a', '#43a047', '#2e7d32', '#ffb74d']

def add_labels(bars, ax, is_time=False):
    for bar in bars:
        height = bar.get_height()
        suffix = ' s' if is_time else ' FPS'
        label_text = f'{height:.4f}{suffix}' if is_time else f'{height:.2f}{suffix}'
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=9)

def create_legend(ax):
    """Tworzy czytelną legendę z pełnymi nazwami bez łamania linii"""
    legend_elements = [Patch(facecolor=colors[i], edgecolor='black', label=labels[i].replace('\n', ' ')) for i in range(len(labels))]
    ax.legend(handles=legend_elements, loc='upper right', frameon=True, facecolor='white', edgecolor='none')

# WYKRESY

# czas dla ball
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, time_ball, color=colors, edgecolor='black', alpha=0.85, width=0.55)
ax.set_title('Średni czas obliczeń potoku dla pliku ball.mp4 (mniej = lepiej)', pad=20, fontweight='bold')
ax.set_ylabel('Czas wykonywania [sekundy]')
ax.set_xlabel('Wariant implementacji algorytmu')
add_labels(bars, ax, is_time=True)
create_legend(ax)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ball_time.png'), dpi=300)
plt.close()

# fps dla ball
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, fps_ball, color=colors, edgecolor='black', alpha=0.85, width=0.55)
ax.set_title('Średnia wydajność przetwarzania dla pliku ball.mp4 (więcej = lepiej)', pad=20, fontweight='bold')
ax.set_ylabel('Przepustowość [FPS]')
ax.set_xlabel('Wariant implementacji algorytmu')
add_labels(bars, ax, is_time=False)
create_legend(ax)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'ball_fps.png'), dpi=300)
plt.close()

# czas dla traffic
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, time_traffic, color=colors, edgecolor='black', alpha=0.85, width=0.55)
ax.set_title('Średni czas obliczeń potoku dla pliku traffic.mp4 (mniej = lepiej)', pad=20, fontweight='bold')
ax.set_ylabel('Czas wykonywania [sekundy]')
ax.set_xlabel('Wariant implementacji algorytmu')
add_labels(bars, ax, is_time=True)
create_legend(ax)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'traffic_time.png'), dpi=300)
plt.close()

# fps dla traffic
fig, ax = plt.subplots(figsize=(10, 6))
bars = ax.bar(labels, fps_traffic, color=colors, edgecolor='black', alpha=0.85, width=0.55)
ax.set_title('Średnia wydajność przetwarzania dla pliku traffic.mp4 (więcej = lepiej)', pad=20, fontweight='bold')
ax.set_ylabel('Przepustowość [FPS]')
ax.set_xlabel('Wariant implementacji algorytmu')
add_labels(bars, ax, is_time=False)
create_legend(ax)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'traffic_fps.png'), dpi=300)
plt.close()

print(f"Sukces! Wszystkie 4 wykresy zostały zapisane w folderze: '{output_dir}'")