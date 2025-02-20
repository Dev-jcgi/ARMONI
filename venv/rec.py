import csv
import mindwave
import time
import datetime
import os
import string
import threading
import queue
import numpy as np

class MindwaveRecorder:
    def __init__(self, port='COM3', duration=1, sampling_rate=1024):
        self.port = port
        self.duration = duration
        self.sampling_rate = sampling_rate
        self.sampling_interval = 1 / sampling_rate
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.headset = None
        self.sample_counter = 0
        self.start_time = None

    def connect_headset(self):
        try:
            print('Conectando a Mindwave...')
            self.headset = mindwave.Headset(self.port)
            print('Conectado, esperando estabilización')
            time.sleep(10)
        except Exception as e:
            print(f"Error de conexión: {e}")
            return False
        return True

    def get_valid_filename(self, filename):
        valid_chars = "-_.() %s%s" % (string.ascii_letters, string.digits)
        filename = ''.join(c for c in filename if c in valid_chars)
        return filename.replace(' ', '_')

    def data_capture_thread(self):
        self.start_time = time.time()
        self.sample_counter = 0

        while not self.stop_event.is_set() and time.time() - self.start_time < self.duration:
            try:
                current_time = time.time()
                counter_ms = (current_time - self.start_time) * 1000.0

                data_point = {
                    'timestamp': datetime.datetime.now(datetime.UTC).isoformat(),
                    'sample_index': self.sample_counter,
                    'counter': counter_ms,
                    'raw': self.headset.raw_value,
                    'attention': self.headset.attention,
                    'meditation': self.headset.meditation,
                    'waves': self.headset.waves
                }
                self.data_queue.put(data_point)
                
                self.sample_counter += 1
                
                time.sleep(self.sampling_interval)
            
            except Exception as e:
                print(f"Error en captura de datos: {e}")
                break

    def write_data_thread(self, filename):
        with open(filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            
            headers = [
                'SampleIndex',
                'Timestamp', 
                'Counter',
                'Raw', 
                'Attention', 
                'Meditation', 
                'Delta', 
                'Theta', 
                'Low-Alpha', 
                'High-Alpha', 
                'Low-Beta', 
                'High-Beta', 
                'Low-Gamma', 
                'Mid-Gamma'
            ]
            writer.writerow(headers)

            while not self.stop_event.is_set():
                try:
                    data_point = self.data_queue.get(timeout=1)
                    
                    values = [
                        data_point['sample_index'],
                        data_point['timestamp'],
                        data_point['counter'],
                        data_point['raw'],
                        data_point['attention'],
                        data_point['meditation']
                    ]
                    
                    values.extend(list(data_point['waves'].values()))
                    
                    writer.writerow(values)
                    
                    print(f"Índice: {data_point['sample_index']}, "
                          f"Counter (ms): {data_point['counter']}, "
                          f"Raw: {data_point['raw']}, "
                          f"Atención: {data_point['attention']}, "
                          f"Meditación: {data_point['meditation']}")
                
                except queue.Empty:
                    if self.stop_event.is_set():
                        break
                except Exception as e:
                    print(f"Error al escribir datos: {e}")
                    break

    def record_session(self, session_name):
        try:
            if not self.connect_headset():
                return

            ts = datetime.datetime.now(datetime.UTC).strftime('%Y-%m-%dT%H-%M-%S')
            filename = f'{self.get_valid_filename(session_name)}_{ts}.csv'
            
            documents_path = os.path.join(os.path.expanduser('~'), 'Documents', 'EEG_Recordings')
            os.makedirs(documents_path, exist_ok=True)
            full_path = os.path.join(documents_path, filename)

            print(f'Grabando datos en: {full_path}')
            print(f'Frecuencia de muestreo: {self.sampling_rate} Hz')
            print(f'Duración: {self.duration} segundo(s)')

            capture_thread = threading.Thread(target=self.data_capture_thread)
            write_thread = threading.Thread(target=self.write_data_thread, args=(full_path,))

            capture_thread.start()
            write_thread.start()

            capture_thread.join(timeout=self.duration)
            
            self.stop_event.set()
            
            capture_thread.join()
            write_thread.join()

            print(f"Grabación completada. Archivo guardado en: {full_path}")
            print(f"Muestras capturadas: {self.sample_counter}")

        except KeyboardInterrupt:
            print("\nGrabación interrumpida por el usuario.")
            self.stop_event.set()
        except Exception as e:
            print(f"Error inesperado: {e}")
            self.stop_event.set()

def plot_eeg_data(filename):
    import pandas as pd
    import matplotlib.pyplot as plt

    df = pd.read_csv(filename)

    plt.figure(figsize=(15, 6))
    plt.plot(df['Counter'], df['Raw'])
    plt.title('Señal Raw EEG')
    plt.xlabel('Tiempo (milisegundos)')
    plt.ylabel('Valor Raw')
    plt.show()

    plt.figure(figsize=(15, 6))
    plt.plot(df['SampleIndex'], df['Raw'])
    plt.title('Señal Raw EEG')
    plt.xlabel('Índice de Muestra (512 Hz)')
    plt.ylabel('Valor Raw')
    plt.show()

def main():
    print('Ingrese el nombre de la sesión de grabación (por ejemplo, nombre de la persona)')
    session_name = input('Nombre de sesión: ')
    
    # Solicitar duración al usuario
    while True:
        try:
            duration = float(input('Ingrese la duración de la grabación en segundos: '))
            if duration <= 0:
                print("La duración debe ser un número positivo.")
                continue
            break
        except ValueError:
            print("Por favor, ingrese un número válido.")
    
    recorder = MindwaveRecorder(duration=duration)
    recorder.record_session(session_name)

    # Opcional: Graficar datos inmediatamente después de la grabación
    # Buscar el último archivo generado
    documents_path = os.path.join(os.path.expanduser('~'), 'Documents', 'EEG_Recordings')
    list_of_files = os.listdir(documents_path)
    latest_file = max([os.path.join(documents_path, f) for f in list_of_files], key=os.path.getctime)
    
    plot_eeg_data(latest_file)

if __name__ == "__main__":
    main()