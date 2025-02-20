import mindwave
import time
import collections
import matplotlib.pyplot as plt
import numpy as np
import threading
import queue
import streamlit as st
import pandas as pd
from datetime import datetime
import os
import getpass
from scipy import signal
from sklearn.preprocessing import StandardScaler
import serial

class EEGDataCollector:
    def __init__(self, port='COM3', max_points=1000):
        self.port = port
        self.max_points = max_points
        self.display_points = 200  # Número de puntos a mostrar
        self.headset = None
        self.data_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.connection_status = False
        self.calibration_data = {}
        self.is_calibrated = False
        
        # Contenedores para la visualización
        self.plot_container = None
        self.fig = None
        self.axs = None
        
        # Inicialización de datos con deque para eficiencia
        self.data = {
            'sample_index': collections.deque(maxlen=max_points),
            'timestamp': collections.deque(maxlen=max_points),
            'attention': collections.deque(maxlen=max_points),
            'meditation': collections.deque(maxlen=max_points),
            'raw': collections.deque(maxlen=max_points),
            'delta': collections.deque(maxlen=max_points),
            'theta': collections.deque(maxlen=max_points),
            'low_alpha': collections.deque(maxlen=max_points),
            'high_alpha': collections.deque(maxlen=max_points),
            'low_beta': collections.deque(maxlen=max_points),
            'high_beta': collections.deque(maxlen=max_points),
            'low_gamma': collections.deque(maxlen=max_points),
            'mid_gamma': collections.deque(maxlen=max_points)
        }

        # Añadir análisis en tiempo real
        self.analysis_data = {
            'attention_mean': collections.deque(maxlen=max_points),
            'meditation_mean': collections.deque(maxlen=max_points),
            'focus_score': collections.deque(maxlen=max_points),
            'stress_level': collections.deque(maxlen=max_points),
            'signal_quality': collections.deque(maxlen=max_points)
        }

        # Inicializar filtros
        self.setup_filters()
        self.scaler = StandardScaler()

    def setup_filters(self):
        """Configuración de filtros para el procesamiento de señales"""
        self.fs = 512  # Frecuencia de muestreo
        self.nyquist = self.fs / 2
        self.b, self.a = signal.butter(2, [0.5/self.nyquist, 30/self.nyquist], btype='band')

    def initialize_plot(self):
        """Inicializar el contenedor de la gráfica"""
        # Crear contenedor para la gráfica
        self.plot_container = st.empty()
        
        # Crear figura inicial
        self.fig, self.axs = plt.subplots(4, 1, figsize=(12, 10))
        
        # Configurar límites y etiquetas
        for ax in self.axs:
            ax.grid(True)
            ax.set_xlim(0, self.display_points)
        
        # Configurar rangos específicos para cada tipo de señal
        self.axs[0].set_ylim(0, 100)  # Atención y Meditación
        self.axs[1].set_ylim(-2000, 2000)  # Raw
        self.axs[2].set_ylim(0, 1000000)  # Ondas cerebrales
        self.axs[3].set_ylim(0, 100)  # Métricas de análisis
        
        # Configurar títulos
        self.axs[0].set_title('Atención y Meditación')
        self.axs[1].set_title('Señal Raw Filtrada')
        self.axs[2].set_title('Ondas Cerebrales')
        self.axs[3].set_title('Análisis en Tiempo Real')

    def update_plot(self):
        """Actualizar la gráfica en tiempo real"""
        # Limpiar los ejes
        for ax in self.axs:
            ax.clear()
            ax.grid(True)
            ax.set_xlim(0, self.display_points)
        
        # Obtener índices para el eje x
        x = np.arange(min(len(self.data['raw']), self.display_points))
        data_slice = slice(-self.display_points, None)
        
        # Actualizar cada subplot
        # Atención y Meditación
        self.axs[0].plot(x, list(self.data['attention'])[data_slice], label='Atención')
        self.axs[0].plot(x, list(self.data['meditation'])[data_slice], label='Meditación')
        self.axs[0].set_ylim(0, 100)
        self.axs[0].set_title('Atención y Meditación')
        self.axs[0].legend()
        
        # Señal Raw
        self.axs[1].plot(x, list(self.data['raw'])[data_slice], label='Raw Filtrada')
        self.axs[1].set_title('Señal Raw Filtrada')
        self.axs[1].legend()
        
        # Ondas Cerebrales
        self.axs[2].plot(x, list(self.data['delta'])[data_slice], label='Delta')
        self.axs[2].plot(x, list(self.data['theta'])[data_slice], label='Theta')
        self.axs[2].plot(x, list(self.data['low_alpha'])[data_slice], label='Low Alpha')
        self.axs[2].set_title('Ondas Cerebrales')
        self.axs[2].legend()
        
        # Métricas de análisis
        if len(self.analysis_data['focus_score']) > 0:
            x_analysis = np.arange(min(len(self.analysis_data['focus_score']), self.display_points))
            analysis_slice = slice(-self.display_points, None)
            self.axs[3].plot(x_analysis, list(self.analysis_data['focus_score'])[analysis_slice], label='Focus Score')
            self.axs[3].plot(x_analysis, list(self.analysis_data['stress_level'])[analysis_slice], label='Stress Level')
            self.axs[3].plot(x_analysis, list(self.analysis_data['signal_quality'])[analysis_slice], label='Signal Quality')
            self.axs[3].set_title('Análisis en Tiempo Real')
            self.axs[3].legend()
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Actualizar el contenedor con la nueva figura
        self.plot_container.pyplot(self.fig)
        
    def connect_headset(self):
        """Conexión mejorada con manejo de reconexión"""
        max_attempts = 3
        attempt = 0
        
        while attempt < max_attempts and not self.connection_status:
            try:
                print(f'Intento de conexión {attempt + 1} de {max_attempts}...')
                
                # Verificar si el puerto existe
                if not os.path.exists(self.port):
                    print(f"Puerto {self.port} no encontrado")
                    return False
                    
                self.headset = mindwave.Headset(self.port)
                time.sleep(2)
                
                # Verificar conexión
                if self.verify_connection():
                    self.connection_status = True
                    print('Conectado exitosamente')
                    return True
                
            except serial.SerialException as e:
                print(f"Error de puerto serial: {e}")
            except Exception as e:
                print(f"Error de conexión: {e}")
            
            attempt += 1
            time.sleep(2)
        
        return False

    def verify_connection(self):
        """Verificar estado de conexión del dispositivo"""
        try:
            test_value = self.headset.attention
            return True if test_value is not None else False
        except:
            return False

    def auto_reconnect(self):
        """Intenta reconectar automáticamente si se pierde la conexión"""
        if not self.verify_connection():
            print("Conexión perdida. Intentando reconectar...")
            self.connection_status = False
            return self.connect_headset()
        return True

    def data_capture_thread(self, sampling_rate):
        """Thread para captura de datos del dispositivo"""
        sampling_interval = 1 / sampling_rate
        
        while not self.stop_event.is_set():
            try:
                if not self.verify_connection():
                    continue
                    
                # Captura de datos
                data_point = {
                    'sample_index': len(self.data['attention']) + 1,
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'attention': self.headset.attention,
                    'meditation': self.headset.meditation,
                    'raw': self.headset.raw_value,
                    'waves': self.headset.waves
                }
                
                # Añadir a la cola
                self.data_queue.put(data_point)
                
                # Control de intervalo de muestreo
                time.sleep(sampling_interval)
            
            except Exception as e:
                print(f"Error en captura de datos: {e}")
                time.sleep(1)  # Esperar antes de reintentar
                continue

    def calibrate(self, duration=10):
        """Calibración automática del dispositivo"""
        print("Iniciando calibración...")
        st.write("Por favor, manténgase relajado durante la calibración...")
        
        calibration_data = {
            'attention': [],
            'meditation': [],
            'raw': []
        }
        
        start_time = time.time()
        while time.time() - start_time < duration:
            if self.verify_connection():
                calibration_data['attention'].append(self.headset.attention)
                calibration_data['meditation'].append(self.headset.meditation)
                calibration_data['raw'].append(self.headset.raw_value)
                time.sleep(0.1)
        
        # Calcular valores base
        self.calibration_data = {
            'attention_baseline': np.mean(calibration_data['attention']),
            'meditation_baseline': np.mean(calibration_data['meditation']),
            'raw_std': np.std(calibration_data['raw'])
        }
        
        self.is_calibrated = True
        print("Calibración completada")
        st.write("Calibración completada exitosamente")

    def process_data_thread(self):
        """Procesamiento de datos mejorado"""
        buffer_size = 50  # Tamaño mínimo del buffer para filtrado
        raw_buffer = []   # Buffer para datos raw
        
        while not self.stop_event.is_set():
            try:
                if not self.auto_reconnect():
                    continue
                
                data_point = self.data_queue.get(timeout=1)
                
                # Acumular datos raw para filtrado
                raw_buffer.append(data_point['raw'])
                
                # Aplicar filtro cuando hay suficientes datos
                if len(raw_buffer) >= buffer_size:
                    try:
                        # Aplicar filtro al buffer completo
                        filtered_data = signal.filtfilt(self.b, self.a, raw_buffer)
                        # Tomar solo el último valor filtrado
                        data_point['raw'] = filtered_data[-1]
                        # Mantener el buffer en un tamaño manejable
                        raw_buffer = raw_buffer[-buffer_size:]
                    except Exception as filter_error:
                        print(f"Error en filtrado: {filter_error}")
                        # Si hay error en el filtrado, usar el valor raw sin filtrar
                        data_point['raw'] = raw_buffer[-1]
                else:
                    # Si no hay suficientes datos, usar el valor sin filtrar
                    data_point['raw'] = data_point['raw']
                
                # Análisis en tiempo real
                analysis_results = self.analyze_real_time(data_point)
                
                # Añadir datos procesados
                self.update_data_structures(data_point, analysis_results)
                
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error en procesamiento: {e}")
                continue
            
    def analyze_real_time(self, data_point):
        """Análisis en tiempo real de los datos"""
        attention = data_point.get('attention', 0)
        meditation = data_point.get('meditation', 0)
        waves = data_point.get('waves', {})
        
        # Calcular score de foco
        beta = waves.get('low_beta', 0) + waves.get('high_beta', 0)
        theta = waves.get('theta', 0)
        focus_score = beta / theta if theta > 0 else 0
        
        # Calcular nivel de estrés
        alpha = waves.get('low_alpha', 0) + waves.get('high_alpha', 0)
        stress_level = beta / alpha if alpha > 0 else 0
        
        # Calidad de señal
        raw_value = abs(data_point.get('raw', 0))
        signal_quality = 100 * (1 - (raw_value / 32768))
        
        # Actualizar análisis
        self.analysis_data['attention_mean'].append(np.mean(list(self.data['attention'])))
        self.analysis_data['meditation_mean'].append(np.mean(list(self.data['meditation'])))
        self.analysis_data['focus_score'].append(focus_score)
        self.analysis_data['stress_level'].append(stress_level)
        self.analysis_data['signal_quality'].append(signal_quality)
        
        return {
            'focus_score': focus_score,
            'stress_level': stress_level,
            'signal_quality': signal_quality
        }

    def update_data_structures(self, data_point, analysis_results):
        """Actualización de estructuras de datos"""
        # Datos básicos
        self.data['sample_index'].append(data_point['sample_index'])
        self.data['timestamp'].append(data_point['timestamp'])
        self.data['attention'].append(data_point['attention'])
        self.data['meditation'].append(data_point['meditation'])
        self.data['raw'].append(data_point['raw'])
        
        # Ondas cerebrales
        try:
            waves = data_point.get('waves', {})
            self.data['delta'].append(waves.get('delta', 0))
            self.data['theta'].append(waves.get('theta', 0))
            self.data['low_alpha'].append(waves.get('low-alpha', 0))
            self.data['high_alpha'].append(waves.get('high-alpha', 0))
            self.data['low_beta'].append(waves.get('low-beta', 0))
            self.data['high_beta'].append(waves.get('high-beta', 0))
            self.data['low_gamma'].append(waves.get('low-gamma', 0))
            self.data['mid_gamma'].append(waves.get('mid-gamma', 0))
        except Exception as e:
            print(f"Error al procesar ondas cerebrales: {e}")

    def save_to_csv(self):
        """Guardar datos en CSV"""
        documents_path = os.path.join(os.path.expanduser("~"), "Documents")
        eegdata_folder = os.path.join(documents_path, "eegdata")
        
        if not os.path.exists(eegdata_folder):
            os.makedirs(eegdata_folder)
        
        user_name = getpass.getuser()
        timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
        filename = f"eegdata_{timestamp}_{user_name}.csv"
        file_path = os.path.join(eegdata_folder, filename)
        
        data_dict = {key: list(value) for key, value in self.data.items()}
        df = pd.DataFrame(data_dict)
        df.to_csv(file_path, index=False)
        
        return file_path

    def cleanup(self):
        """Limpieza de recursos mejorada"""
        self.stop_event.set()
        
        try:
            if self.headset:
                if hasattr(self.headset, 'dongle') and self.headset.dongle:
                    self.headset.dongle.close()
                self.headset = None
        except Exception as e:
            print(f"Error al cerrar la conexión: {e}")
        
        self.connection_status = False

    def real_time_eeg_capture(self, duration=60, sampling_rate=512):
        """Captura de EEG mejorada con visualización en tiempo real"""
        if not self.connect_headset():
            st.error("No se pudo conectar al dispositivo")
            return

        if not self.is_calibrated:
            self.calibrate()

        try:
            # Inicializar la gráfica
            self.initialize_plot()
            
            # Crear métricas containers
            metrics_container = st.empty()
            
            # Iniciar hilos
            capture_thread = threading.Thread(target=self.data_capture_thread, args=(sampling_rate,))
            process_thread = threading.Thread(target=self.process_data_thread)
            
            capture_thread.start()
            process_thread.start()

            start_time = time.time()
            
            while time.time() - start_time < duration:
                if not self.connection_status:
                    st.warning("Reconectando...")
                    if not self.auto_reconnect():
                        st.error("No se pudo reconectar")
                        break
                
                # Actualizar gráfica
                self.update_plot()
                
                # Actualizar métricas
                if len(self.analysis_data['focus_score']) > 0:
                    col1, col2, col3 = metrics_container.columns(3)
                    with col1:
                        st.metric("Focus Score", f"{list(self.analysis_data['focus_score'])[-1]:.2f}")
                    with col2:
                        st.metric("Stress Level", f"{list(self.analysis_data['stress_level'])[-1]:.2f}")
                    with col3:
                        st.metric("Signal Quality", f"{list(self.analysis_data['signal_quality'])[-1]:.2f}%")
                
                time.sleep(0.1)  # Ajustar según necesidad
                
                if not capture_thread.is_alive() or not process_thread.is_alive():
                    st.error("Error en los hilos de captura")
                    break

        except Exception as e:
            st.error(f"Error durante la captura: {e}")
        
        finally:
            self.cleanup()
            self.save_and_offer_download()

    def save_and_offer_download(self):
        """Guardar datos y ofrecer descarga"""
        try:
            csv_filename = self.save_to_csv()
            st.success(f"Datos guardados en: {csv_filename}")
            
            with open(csv_filename, 'rb') as file:
                st.download_button(
                    label="Descargar datos CSV",
                    data=file,
                    file_name=os.path.basename(csv_filename),
                    mime='text/csv'
                )
        except Exception as e:
            st.error(f"Error al guardar datos: {e}")
            
def main():
    st.title("Sistema Avanzado de Captura EEG")
    
    st.sidebar.header("Configuración")
    port = st.sidebar.text_input("Puerto COM", "COM3")
    duration = st.sidebar.slider("Duración (segundos)", 10, 300, 60)
    sampling_rate = st.sidebar.slider("Frecuencia de muestreo (Hz)", 1, 512, 512)
    
    st.sidebar.header("Opciones Avanzadas")
    enable_filtering = st.sidebar.checkbox("Habilitar filtrado", True)
    show_analysis = st.sidebar.checkbox("Mostrar análisis en tiempo real", True)
    
    if st.button("Iniciar Captura EEG"):
        collector = EEGDataCollector(port=port)
        collector.real_time_eeg_capture(duration=duration, sampling_rate=sampling_rate)

if __name__ == "__main__":
    main()