import cv2
import numpy as np
import time
import os

def main():
    # foldery na filmy
    input_dir = "videos_input"
    output_dir = os.path.join("videos_output", "sequential")

    # tworzenie folderów jeśli nie istnieją
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # pobieranie nazwy pliku
    print("--- POTOK DETEKCJI RUCHU: WARIANT SEKWENCYJNY ---")
    video_name = input("Podaj nazwe pliku wideo (np. video.avi lub video.mp4): ").strip()

    # ścieżka dla pliku
    video_path = os.path.join(input_dir, video_name)

    # odseparowanie nazwy pliku dla końcowego wyniku
    name_without_ext, _ = os.path.splitext(video_name)
    output_name = f"{name_without_ext}_seq.mp4"
    output_path = os.path.join(output_dir, output_name)

    # otwarcie pliku z walidacją istnienia pliku
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"\nBlad: Nie można otworzyc pliku wideo '{video_path}'.")
        print(f"Upewnij sie, ze film znajduje się w folderze: {os.path.abspath(input_dir)}")
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

    # rezerwacja pamięci
    processed_frames_list = []

    # inicjalizacja modelu tła za pomocą pierwszej klatki
    ret, first_frame = cap.read()
    if not ret:
        print("Blad: Nie mozna odczytać klatki inicjalizujacej.")
        cap.release()
        return

    # klatka bazowa
    gray_prev = cv2.cvtColor(first_frame, cv2.COLOR_BGR2GRAY)
    gray_prev = cv2.GaussianBlur(gray_prev, blur_kernel_size, 0)

    print(f"\n[START] Analiza pliku: {video_path}")
    print(f"[INFO] Rozdzielczosc: {width}x{height}, Klatki: {total_frames}")
    print(f"[INFO] Wynik zostanie zapisany w: {output_path}\nPrzetwarzanie...")
    
    # pomiar czasu
    start_time = time.time()
    processed_frames = 0

    # przetwarzanie wideo klatka po klatce
    while True:
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
        opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # usuwanie szumu
        clean_mask = cv2.morphologyEx(opened, cv2.MORPH_CLOSE, kernel) # scalanie przerwanych konturów

        # wykrywanie konturów obiektów w ruchu i rysowanie ramek
        contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        result_frame = frame.copy()
        
        for contour in contours:
            if cv2.contourArea(contour)< min_contour_area:
                continue
            # wyznaczenie boxa dla poruszającego się obiektu
            (x, y, w, h) = cv2.boundingRect(contour)
            cv2.rectangle(result_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

        # aktualizacja klatki - bieżąca będzie poprzednią
        gray_prev = gray_curr.copy()

        processed_frames_list.append(result_frame)
        processed_frames += 1

    # koniec pomiaru czasu i czyszczenie zasobów systemowych
    end_time = time.time()
    total_time = end_time - start_time
    fps_achieved = processed_frames / total_time

    cap.release()
    
    print(f"\n[INFO] Obliczenia zakonczone. Trwa zapisywanie pliku wynikowego na dysk...")
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    writer = cv2.VideoWriter(output_path, fourcc, fps_video, (width, height))

    for saved_frame in processed_frames_list:
        writer.write(saved_frame)
        
    writer.release()
    print(f"[INFO] Plik zapisany pomyslnie w: {output_path}")

    # podsumowanie metody
    print("\n--- PODSUMOWANIE WARIANTU SEKWENCYJNEGO ---")
    print(f"Czas obliczen:            {total_time:.4f} sekund")
    print(f"Przetworzone klatki w petli: {processed_frames}")
    print(f"Wydajnosc przetwarzania:    {fps_achieved:.2f} FPS")

if __name__ == "__main__":
    main()