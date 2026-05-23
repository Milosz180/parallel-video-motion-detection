import cv2
import numpy as np
import time
import os
from mpi4py import MPI

def main():

    # inicjalizacja środowiska MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    input_dir = "videos_input"
    output_dir = os.path.join("videos_output", "mpi")

    # zmienna na nazwę pliku wideo
    video_name = None

    # tylko proces główy pobiera nazwę od użytkownika (aby każdy z procesów nie próbował przyjmować nazwy)
    if rank == 0:
        print("--- POTOK DETEKCJI RUCHU: WARIANT ROZPROSZONY MPI ---")
        video_name = input("Podaj nazwę pliku wideo (np. video.mp4): ").strip()

    # broadcast nazwy pliku do wszystkich procesów
    video_name = comm.bcast(video_name, root=0)

    # ścieżka do pliku
    video_path = os.path.join(input_dir, video_name)

    # otwarcie pliku z walidacją istnienia pliku na każdym procesie
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        if rank == 0:
            print(f"\nBłąd: Nie można otworzyć pliku wideo '{video_path}'.")
        return

    # pobieranie parametrów technicznych strumienia wideo
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps_video = cap.get(cv2.CAP_PROP_FPS) or 30
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # parametry techniczne algorytmu zależne od rozdzielczości
    max_side = max(width, height)
    if max_side >= 1920:
        threshold_val = 20
        blur_kernel_size = (7, 7)
        morph_kernel_size = (7, 7)
        min_contour_area = 600
    elif max_side >= 1280:
        threshold_val = 15
        blur_kernel_size = (5, 5)
        morph_kernel_size = (5, 5)
        min_contour_area = 150
    else:
        threshold_val = 10
        blur_kernel_size = (3, 3)
        morph_kernel_size = (5, 5)
        min_contour_area = 50

    # podział pracy pomiędzy procesy MPI
    # obliczanie ile klatek przypada na każdy proces
    frames_per_process = total_frames // size
    start_frame = rank * frames_per_process
    # ostatni proces przejmuje ewentualną resztę z dzielenia
    end_frame = total_frames if rank == size - 1 else (rank + 1) * frames_per_process
    num_frames_to_process = end_frame - start_frame

    # każdy proces przewija wideo do swojej klatki startowej
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # pobranie poprzedzającej klatki od startowej do detekcji ruchu Frame-to-Frame
    if start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame - 1)
        ret, prev_frame = cap.read()
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.GaussianBlur(gray_prev, blur_kernel_size, 0)
    else:
        # jeśli początek filmu, inicjalizujemy pierwszą klatką
        ret, prev_frame = cap.read()
        gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        gray_prev = cv2.GaussianBlur(gray_prev, blur_kernel_size, 0)

    if rank == 0:
        print("--- POTOK DETEKCJI RUCHU: WARIANT ROZPROSZONY MPI ---")
        print(f"[START] Analiza pliku MPI: {video_path}")
        print(f"[INFO] Liczba procesów MPI: {size}")
        print(f"[INFO] Rozdzielczość: {width}x{height}, Łącznie klatek: {total_frames}")
        os.makedirs(output_dir, exist_ok=True)
        start_time = time.time()

    # lokalna tablica na przetworzone klatki wynikowe danego procesu
    local_frames = []

    # główna pętla przetwarzania
    for _ in range(num_frames_to_process):
        ret, frame = cap.read()
        if not ret:
            break
        
        # preprocessing bieżącej klatki
        gray_curr = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_curr = cv2.GaussianBlur(gray_curr, blur_kernel_size, 0)

        # różnicowanie klatek od modelu tła
        diff = cv2.absdiff(gray_curr, gray_prev)
        
        # progowanie
        _, mask = cv2.threshold(diff, threshold_val, 255, cv2.THRESH_BINARY)

        # asymetryczne czyszczenie maski w pętli
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        clean_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)
        
        # wykrywanie konturów obiektów w ruchu i rysowanie ramek
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = frame.copy()

        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            # wyznaczenie boxa dla poruszającego się obiektu
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # aktualizacja klatki - bieżąca będzie poprzednią
        gray_prev = gray_curr.copy()
        local_frames.append(result_frame)

    cap.release()

    # zbieranie wyników
    # wszystkie procesy wysyłają swoje gotowe pakiety klatek do procesu Rank 0
    all_gathered_chunks = comm.gather(local_frames, root=0)

    # zapis wyniku
    if rank == 0:
        name_without_ext, _ = os.path.splitext(video_name)
        output_path = os.path.join(output_dir, f"{name_without_ext}_mpi.mp4")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

        print("Scalanie pakietów danych i zapis na dysk...")
        total_processed_frames = 0
        
        # przechodzimy po kolei przez wyniki od każdego procesu i zapisujemy do pliku
        for chunk in all_gathered_chunks:
            for final_frame in chunk:
                writer.write(final_frame)
                total_processed_frames += 1

        # zapis do folderu
        writer.release()
        
        # koniec pomiaru czasu i czyszczenie zasobów systemowych
        end_time = time.time()
        total_time = end_time - start_time
        fps_achieved = total_processed_frames / total_time

        # podsumowanie metody
        print("\n--- PODSUMOWANIE WARIANTU ROZPROSZONEGO MPI ---")
        print(f"Czas obliczeń i scalania: {total_time:.4f} sekund")
        print(f"Łącznie przetworzone klatki: {total_processed_frames}")
        print(f"Wydajność przetwarzania:     {fps_achieved:.2f} FPS")

if __name__ == "__main__":
    main()