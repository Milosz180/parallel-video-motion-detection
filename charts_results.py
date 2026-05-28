import os
import numpy as np
import matplotlib.pyplot as plt

# tworzenie folderu
output_dir = 'charts_results'
os.makedirs(output_dir, exist_ok=True)

# konfiguracja wykresów
if 'seaborn-v0_8-whitegrid' in plt.style.available:
    plt.style.use('seaborn-v0_8-whitegrid')
elif 'seaborn-whitegrid' in plt.style.available:
    plt.style.use('seaborn-whitegrid')
else:
    plt.style.use('default')

plt.rcParams.update({
    'font.size': 10, 
    'axes.labelsize': 11, 
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'figure.titlesize': 14
})

# kolory i nazwy do wykresów
COLORS = {
    'sequential': '#fbc02d',
    
    'openmp_2':   '#ccff90',
    'openmp_4':   '#b2ff59',
    'openmp_6':   '#76ff03',
    'openmp_12':  '#64dd17',
    
    'mpi_2':      '#90caf9',
    'mpi_4':      '#42a5f5',
    'mpi_6':      '#1e88e5',
    'mpi_8':      '#1976d2',
    'mpi_12':     '#1565c0',
    
    'cuda':       '#e53935'
}

VARIANT_NAMES = {
    'sequential': 'Sekwencyjny (Baseline)',
    'openmp_2':   'OpenMP (2 wątki)',
    'openmp_4':   'OpenMP (4 wątki)',
    'openmp_6':   'OpenMP (6 wątków)',
    'openmp_12':  'OpenMP (12 wątków)',
    'mpi_2':      'MPI (2 procesy)',
    'mpi_4':      'MPI (4 procesy)',
    'mpi_6':      'MPI (6 procesów)',
    'mpi_8':      'MPI (8 procesów)',
    'mpi_12':     'MPI (12 procesów)',
    'cuda':       'CUDA (Karta graficzna)'
}

X_AXIS_LABELS = {
    'sequential': 'Sekwencyjny\n(Baseline)',
    'openmp_2':   'OpenMP\n(2 w.)',
    'openmp_4':   'OpenMP\n(4 w.)',
    'openmp_6':   'OpenMP\n(6 w.)',
    'openmp_12':  'OpenMP\n(12 w.)',
    'mpi_2':      'MPI\n(2 pr.)',
    'mpi_4':      'MPI\n(4 pr.)',
    'mpi_6':      'MPI\n(6 pr.)',
    'mpi_8':      'MPI\n(8 pr.)',
    'mpi_12':     'MPI\n(12 pr.)',
    'cuda':       'CUDA\n(GPU)'
}

# baza wyników z testów
VIDEO_DATA = {
    'ball.mp4': {
        'frames': 121,
        'sequential': {'time': 1.7030,  'fps': 71.05,   'speedup': 1.00, 'efficiency': 1.00},
        'openmp_2':   {'time': 9.4402,  'fps': 12.82,   'speedup': 0.18, 'efficiency': 0.09}, 
        'openmp_4':   {'time': 9.4931,  'fps': 12.75,   'speedup': 0.18, 'efficiency': 0.045},
        'openmp_6':   {'time': 9.5410,  'fps': 12.68,   'speedup': 0.18, 'efficiency': 0.03},
        'openmp_12':  {'time': 9.5912,  'fps': 12.62,   'speedup': 0.18, 'efficiency': 0.015},
        'mpi_2':      {'time': 1.0938,  'fps': 110.70,  'speedup': 1.56, 'efficiency': 0.78},
        'mpi_4':      {'time': 0.9754,  'fps': 124.07,  'speedup': 1.75, 'efficiency': 0.44},
        'mpi_6':      {'time': 1.0946,  'fps': 110.90,  'speedup': 1.56, 'efficiency': 0.26},
        'mpi_12':     {'time': 1.4974,  'fps': 83.01,   'speedup': 1.14, 'efficiency': 0.10},
        'cuda':       {'time': 1.9966,  'fps': 60.15,   'speedup': 0.85, 'efficiency': 0.0}
    },
    'traffic.mp4': {
        'frames': 1140,
        'sequential': {'time': 25.0287, 'fps': 45.55,   'speedup': 1.00, 'efficiency': 1.00},
        'openmp_2':   {'time': 33.1024, 'fps': 34.44,   'speedup': 0.76, 'efficiency': 0.38},
        'openmp_4':   {'time': 33.3105, 'fps': 34.22,   'speedup': 0.75, 'efficiency': 0.19},
        'openmp_6':   {'time': 33.5112, 'fps': 34.02,   'speedup': 0.75, 'efficiency': 0.125},
        'openmp_12':  {'time': 34.1124, 'fps': 33.42,   'speedup': 0.73, 'efficiency': 0.06},
        'mpi_2':      {'time': 21.4393, 'fps': 53.44,   'speedup': 1.17, 'efficiency': 0.59},
        'mpi_4':      {'time': 16.5019, 'fps': 69.28,   'speedup': 1.52, 'efficiency': 0.38},
        'mpi_8':      {'time': 18.6327, 'fps': 62.31,   'speedup': 1.34, 'efficiency': 0.17},
        'mpi_12':     {'time': 17.5249, 'fps': 65.25,   'speedup': 1.43, 'efficiency': 0.12},
        'cuda':       {'time': 25.6743, 'fps': 44.40,   'speedup': 0.97, 'efficiency': 0.0}
    },
    'goal.mp4': {
        'frames': 110,
        'sequential': {'time': 0.1235,  'fps': 890.69,  'speedup': 1.00, 'efficiency': 1.00},
        'openmp_2':   {'time': 8.1124,  'fps': 13.56,   'speedup': 0.015, 'efficiency': 0.007},
        'openmp_4':   {'time': 8.1524,  'fps': 13.49,   'speedup': 0.015, 'efficiency': 0.004},
        'openmp_6':   {'time': 8.2015,  'fps': 13.41,   'speedup': 0.015, 'efficiency': 0.002},
        'openmp_12':  {'time': 8.2812,  'fps': 13.28,   'speedup': 0.015, 'efficiency': 0.001},
        'mpi_2':      {'time': 0.1090,  'fps': 1028.24, 'speedup': 1.13, 'efficiency': 0.57},
        'mpi_4':      {'time': 0.1230,  'fps': 899.95,  'speedup': 1.00, 'efficiency': 0.25},
        'mpi_8':      {'time': 0.1588,  'fps': 699.89,  'speedup': 0.78, 'efficiency': 0.10},
        'mpi_12':     {'time': 0.3349,  'fps': 329.75,  'speedup': 0.37, 'efficiency': 0.03},
        'cuda':       {'time': 0.2241,  'fps': 487.89,  'speedup': 0.55, 'efficiency': 0.0}
    }
}

# kolejność renderowania
variants_order = [
    'openmp_2', 'openmp_4', 'openmp_6', 'openmp_12',
    'mpi_2', 'mpi_4', 'mpi_6', 'mpi_12'
]

# najlepsze wyniki konfiguracji
FINAL_VARIANTS = {
    'ball.mp4':    ['sequential', 'openmp_2', 'mpi_4', 'cuda'],
    'traffic.mp4': ['sequential', 'openmp_2', 'mpi_4', 'cuda'],
    'goal.mp4':    ['sequential', 'openmp_2', 'mpi_2', 'cuda']
}

video_files = ['ball.mp4', 'traffic.mp4', 'goal.mp4']

# funkcje graficzne
def add_bar_labels(bars, ax, metric_type):
    """Automatycznie nanosi wartości liczbowe bezpośrednio nad słupki"""
    for bar in bars:
        height = bar.get_height()
        if height == 0.0:
            continue
        if metric_type == 'fps':
            label_text = f'{height:.1f}' if height < 100 else f'{height:.0f}'
        elif metric_type == 'efficiency':
            label_text = f'{height:.2f}'
        elif metric_type == 'speedup':
            label_text = f'{height:.2f}x'
        else:
            label_text = f'{height}'
            
        ax.annotate(label_text,
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 2),  
                    textcoords="offset points",
                    ha='center', va='bottom', fontweight='bold', fontsize=8)


# wykresy zbiorcze dla MPI i OPENMP
print("[MODULE A] Generowanie paneli wielowykresowych OpenMP vs MPI...")

for video_name in video_files:
    video_data = VIDEO_DATA[video_name]
    
    active_variants = []
    for cores in [2, 4, 6, 12]:
        if f'openmp_{cores}' in video_data: 
            active_variants.append(f'openmp_{cores}')
    for cores in [2, 4, 6, 8, 12]:
        if f'mpi_{cores}' in video_data: 
            active_variants.append(f'mpi_{cores}')

    # konfiguracja siatki wykresów
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.88, 0.12])
    
    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[0, 2]) 
    
    # podwykres fps
    fps_values = [video_data[v]['fps'] for v in active_variants]
    bar_colors = [COLORS[v] for v in active_variants]
    x_labels = [X_AXIS_LABELS[v] for v in active_variants]
    
    bars1 = ax1.bar(x_labels, fps_values, color=bar_colors, edgecolor='black', alpha=0.85, width=0.55)
    ax1.set_title('Wydajność przetwarzania potoku', fontweight='bold', pad=10)
    ax1.set_ylabel('Przepustowość [FPS]')
    ax1.set_xlabel('Konfiguracja środowiska')
    
    if video_name == 'goal.mp4':
        ax1.set_yscale('log')
        ax1.set_ylabel('Przepustowość [FPS] (Skala Logarytmiczna)')
        ax1.set_ylim(1, 3500)
    add_bar_labels(bars1, ax1, metric_type='fps')

    # podwykres przyśpieszenie
    ax2.plot([2, 12], [2, 12], color='#9e9e9e', linestyle='--', alpha=0.6) # Linia idealna Amdahla
    
    resources_omp = [2, 4, 6, 12]
    omp_speedup = [video_data[f'openmp_{r}']['speedup'] for r in resources_omp]
    ax2.plot(resources_omp, omp_speedup, marker='o', linewidth=2, color=COLORS['openmp_12'])
    
    resources_mpi = [2, 4, 6, 12] if f'mpi_6' in video_data else [2, 4, 8, 12]
    mpi_speedup = [video_data[f'mpi_{r}']['speedup'] for r in resources_mpi]
    ax2.plot(resources_mpi, mpi_speedup, marker='s', linewidth=2, color=COLORS['mpi_12'])
    
    ax2.set_title('Charakterystyka przyspieszenia algorytmu', fontweight='bold', pad=10)
    ax2.set_xlabel('Liczba wątków / procesów')
    ax2.set_ylabel('Przyspieszenie $S_p$')
    ax2.set_xlim(1.5, 12.5)
    ax2.set_xticks([2, 4, 6, 8, 12])
    ax2.set_ylim(0, max(max(mpi_speedup), max(omp_speedup), 1.5) + 0.3)
    
    for x_p, y_p in zip(resources_omp, omp_speedup):
        ax2.text(x_p, y_p + 0.04, f'{y_p:.2f}x', ha='center', va='bottom', fontsize=8, color='#33691e', fontweight='bold')
    for x_p, y_p in zip(resources_mpi, mpi_speedup):
        ax2.text(x_p, y_p + 0.04, f'{y_p:.2f}x', ha='center', va='bottom', fontsize=8, color='#0d47a1', fontweight='bold')

    # podwykres efektywność
    efficiency_values = [video_data[v]['efficiency'] for v in active_variants]
    efficiency_colors = [COLORS[v] for v in active_variants]
    
    bars3 = ax3.bar(x_labels, efficiency_values, color=efficiency_colors, edgecolor='black', alpha=0.85, width=0.55)
    ax3.set_title('Efektywność wykorzystania zasobów', fontweight='bold', pad=10)
    ax3.set_ylabel('Współczynnik efektywności $E_p$')
    ax3.set_xlabel('Konfiguracja środowiska')
    ax3.set_ylim(0, 1.15)
    add_bar_labels(bars3, ax3, metric_type='efficiency')

    # legenda
    legend_handles = [plt.Rectangle((0,0),1,1, color=COLORS[v], edgecolor='black') for v in active_variants]
    legend_labels = [VARIANT_NAMES[v] for v in active_variants]
    
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    ax_leg.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.4),
                  ncol=len(active_variants), frameon=True, facecolor='white', edgecolor='none')

    fig.suptitle(f'Analiza porównawcza OpenMP vs MPI dla pliku: {video_name}', fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f'MPI_OpenMP_{video_name.replace(".mp4", "")}.png'), dpi=300)
    plt.close()


# zestawienie
print("\n[MODULE B] Generowanie ostatecznych zestawień paradygmatów (Final Showdown)...")

for video_name in video_files:
    video_data = VIDEO_DATA[video_name]
    final_variants = FINAL_VARIANTS[video_name]
    
    time_values  = [video_data[v]['time'] for v in final_variants]
    fps_values   = [video_data[v]['fps'] for v in final_variants]
    speedup_values  = [video_data[v]['speedup'] for v in final_variants]
    
    final_colors = [COLORS[v] for v in final_variants]
    final_labels = [X_AXIS_LABELS[v] for v in final_variants]
    
    fig = plt.figure(figsize=(16, 6))
    gs = fig.add_gridspec(2, 3, height_ratios=[0.88, 0.12])
    
    ax1 = fig.add_subplot(gs[0, 0]) 
    ax2 = fig.add_subplot(gs[0, 1]) 
    ax3 = fig.add_subplot(gs[0, 2]) 
    
    # podwykres 1 czas
    bars1 = ax1.bar(final_labels, time_values, color=final_colors, edgecolor='black', alpha=0.85, width=0.5)
    ax1.set_title('Średni czas wykonania potoku', fontweight='bold', pad=10)
    ax1.set_ylabel('Czas obliczeń [sekundy]')
    ax1.set_xlabel('Strategia optymalizacji')
    
    if video_name == 'goal.mp4':
        ax1.set_yscale('log')
        ax1.set_ylabel('Czas [s] (Skala Logarytmiczna)')
        ax1.set_ylim(0.01, 20)
        
    for bar in bars1:
        h = bar.get_height()
        if h == 0.0: continue
        label_text = f'{h:.2f}s' if h >= 1.0 else f'{h:.4f}s'
        ax1.annotate(label_text, xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 2), textcoords="offset points", ha='center', va='bottom', fontweight='bold', fontsize=8)

    # podwykres 2 fps
    bars2 = ax2.bar(final_labels, fps_values, color=final_colors, edgecolor='black', alpha=0.85, width=0.5)
    ax2.set_title('Wydajność przetwarzania danych', fontweight='bold', pad=10)
    ax2.set_ylabel('Przepustowość [FPS]')
    ax2.set_xlabel('Strategia optymalizacji')
    
    if video_name == 'goal.mp4':
        ax2.set_yscale('log')
        ax2.set_ylabel('Przepustowość [FPS] (Skala Logarytmiczna)')
        ax2.set_ylim(1, 3500)
    add_bar_labels(bars2, ax2, metric_type='fps')

    # podwykres 3 przyśpieszenie
    bars3 = ax3.bar(final_labels, speedup_values, color=final_colors, edgecolor='black', alpha=0.85, width=0.5)
    ax3.set_title('Uzyskane przyspieszenie względne ($S_p$)', fontweight='bold', pad=10)
    ax3.set_ylabel('Współczynnik przyspieszenia')
    ax3.set_xlabel('Strategia optymalizacji')
    ax3.axhline(1.0, color='#757575', linestyle='--', alpha=0.5) # Próg baseline
    ax3.set_ylim(0, max(speedup_values) + 0.3)
    add_bar_labels(bars3, ax3, metric_type='speedup')

    # legenda
    legend_handles = [plt.Rectangle((0,0),1,1, color=COLORS[v], edgecolor='black') for v in final_variants]
    legend_labels = [VARIANT_NAMES[v] for v in final_variants]
    
    ax_leg = fig.add_subplot(gs[1, :])
    ax_leg.axis('off')
    ax_leg.legend(legend_handles, legend_labels, loc='center', bbox_to_anchor=(0.5, 0.4),
                  ncol=len(final_variants), frameon=True, facecolor='white', edgecolor='none')

    fig.suptitle(f'Zbiorcze zestawienie ostateczne dla pliku wideo: {video_name}', fontweight='bold', fontsize=14, y=0.98)
    
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.savefig(os.path.join(output_dir, f'final_results_{video_name.replace(".mp4", "")}.png'), dpi=300)
    plt.close()

print("\n[KONIEC] Wszystkie wykresy (łącznie 6 szerokich paneli) zostały poprawnie wygenerowane w rozdzielczości 300 DPI!")
print(f"Wyniki graficzne znajdziesz w folderze: '{os.path.abspath(output_dir)}/'")