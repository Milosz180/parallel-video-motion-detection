import cv2
import numpy as np
import time
import os
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def openmp_pixel_pipeline(gray_curr, gray_prev, threshold_val, blur_k):
    
    # równoległe jądro OpenMP (Numba) zastępujące sekwencyjny potok pikselowy.
    # wykonuje uśredniający filtr wygładzający (odpowiednik blur), absdiff oraz threshold.

    height, width = gray_curr.shape
    
    # alokacja macierzy wyjściowych
    blurred_curr = np.zeros_like(gray_curr)
    mask = np.zeros_like(gray_curr)
    
    half_k = blur_k // 2

    # równoległa pętla OpenMP - rozbicie wierszy obrazu na wątki CPU
    for i in prange(height):
        for j in range(width):
            
            # równoległy filtr wygładzający
            sum_val = 0
            count = 0
            for ki in range(-half_k, half_k + 1):
                for kj in range(-half_k, half_k + 1):
                    ni = i + ki
                    nj = j + kj
                    if ni >= 0 and ni < height and nj >= 0 and nj < width:
                        sum_val += gray_curr[ni, nj]
                        count += 1
            blurred_curr[i, j] = sum_val // count

            # równoległe różnicowanie klatek i progowanie 
            pixel_diff = abs(int(blurred_curr[i, j]) - int(gray_prev[i, j]))
            
            if pixel_diff > threshold_val:
                mask[i, j] = 255
            else:
                mask[i, j] = 0
                
    return mask, blurred_curr

def main():
    # foldery na filmy
    input_dir = "videos_input"
    output_dir = os.path.join("videos_output", "openmp")

    # tworzenie folderów jeśli nie istnieją
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # pobieranie nazwy pliku
    print("--- POTOK DETEKCJI RUCHU: WARIANT ROWNOLEGLY OpenMP ---")
    video_name = input("Podaj nazwe pliku wideo (np. video.mp4): ").strip()

    # ścieżka dla pliku
    video_path = os.path.join(input_dir, video_name)

    # odseparowanie nazwy pliku dla końcowego wyniku
    name_without_ext, _ = os.path.splitext(video_name)
    output_name = f"{name_without_ext}_openmp.mp4"
    output_path = os.path.join(output_dir, output_name)

    # otwarcie pliku z walidacją istnienia pliku
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nBlad: Nie mozna otworzyc pliku wideo '{video_path}'.")
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
        blur_k = 7
        morph_kernel_size = (7, 7)
        min_contour_area = 600
    elif max_side >= 1280:
        threshold_val = 15
        blur_k = 5
        morph_kernel_size = (5, 5)
        min_contour_area = 150
    else:
        threshold_val = 10
        blur_k = 3
        morph_kernel_size = (5, 5)
        min_contour_area = 50

    # rezerwacja pamieci
    processed_frames_list = []

    # inicjalizacja modelu tła za pomocą pierwszej klatki
    ret, first_frame = cap.read()
    if not ret:
        print("Blad: Nie mozna odczytac klatki inicjalizujacej.")
        cap.release()
        return

    # klatka bazowa - pierwszy rozruch sekwencyjny
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.blur(gray_prev, (blur_k, blur_k))

    print(f"\n[START] Analiza pliku OpenMP: {video_path}")
    print(f"[INFO] Rozdzielczosc: {width}x{height}, Klatki: {total_frames}")
    print(f"[INFO] Wynik zostanie zapisany w: {output_path}")
    
    # pomiar czasu
    start_time = time.time()
    processed_frames = 0

    # przetwarzanie wideo klatka po klatce
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # preprocessing bieżącej klatki
        gray_curr_raw = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # wywołanie równoległego potoku OpenMP
        mask, gray_curr_blurred = openmp_pixel_pipeline(gray_curr_raw, gray_prev, threshold_val, blur_k)

        # czyszczenie maski za pomocą klasycznej morfologii (operacje na CPU)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, morph_kernel_size)
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) 
        clean_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel)

        # wykrywanie konturów obiektów w ruchu i rysowanie ramek
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = frame.copy()
        
        for contour in contours:
            if cv2.contourArea(contour) < min_contour_area:
                continue
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # aktualizacja klatki - bieżąca będzie poprzednią
        gray_prev = gray_curr_blurred.copy()

        processed_frames_list.append(result_frame)
        processed_frames += 1

    # koniec pomiaru czasu i czyszczenie zasobów systemowych
    end_time = time.time()
    total_time = end_time - start_time
    fps_achieved = processed_frames / total_time

    cap.release()

    # zapis do pliku poza pomiarem czasu
    print(f"\n[INFO] Obliczenia zakonczone. Trwa zapisywanie pliku na dysk...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    for saved_frame in processed_frames_list:
        writer.write(saved_frame)
        
    writer.release()
    print(f"[INFO] Plik zapisany pomyslnie w: {output_path}")

    print("\n--- PODSUMOWANIE WARIANTU ROWNOLEGLEGO OpenMP ---")
    print(f"Czas obliczen:            {total_time:.4f} sekund")
    print(f"Przetworzone klatki w petli: {processed_frames}")
    print(f"Wydajnosc przetwarzania:    {fps_achieved:.2f} FPS")

if __name__ == "__main__":
    main()