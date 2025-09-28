# MATLAB Engine API für Python & MatlabWrapper Nutzung

Dieses Dokument beschreibt die Installation der MATLAB Engine API für Python, sowie die Nutzung des `MatlabWrapper` zum Starten von MATLAB, Ausführen von Simulink-Simulationen und Plotten der Ergebnisse.

---

## 📋 Voraussetzungen
- Installiertes **MATLAB** (R2014b oder neuer)
- Installiertes **Python** (3.7 – 3.11 empfohlen)
- MATLAB und Python müssen **gleiche Architektur** haben (z. B. beide 64-Bit)
- Optional: `matplotlib` und `numpy` für die Visualisierung:
  ```bash
  pip install matplotlib numpy
  ```

---

## 🔧 Installation der MATLAB Engine API

1. **Terminal oder Anaconda Prompt öffnen**  
   Stelle sicher, dass du das richtige Python verwendest (z. B. deine Conda-Umgebung aktivieren).

2. **Zum MATLAB-Ordner wechseln**  
   Navigiere zum Unterordner der Engine-API:

   ```bash
   cd "MATLABROOT/extern/engines/python"
   ```

3. **Installation starten**
   ```bash
   python setup.py install
   ```

   **Hinweis:**  
   - Windows: `MATLABROOT` z. B. `C:\Program Files\MATLAB\R2023b`  
   - Linux: `MATLABROOT` z. B. `/usr/local/MATLAB/R2023b`

4. **Installation testen**
   ```bash
   python -c "import matlab.engine; print('MATLAB Engine erfolgreich installiert!')"
   ```

---

## 🚀 Erste Schritte mit MATLAB Engine

### MATLAB starten und beenden
```python
import matlab.engine

# MATLAB starten
eng = matlab.engine.start_matlab()

# Beispielberechnung
result = eng.sqrt(16.0)
print("Ergebnis:", result)

# MATLAB beenden
eng.quit()
```

---

## 🛠 Nutzung des `MatlabWrapper`

Die Klasse `MatlabWrapper` befindet sich in der Datei:  
[MatlabWrapper.py](MatlabWrapper.py)

### Klasse importieren
```python
from MatlabWrapper import MatlabWrapper

### Simulink-Simulation ausführen
```python
with MatlabWrapper() as mat:
    mat.run_simulation(
        simulink_model='model_name_or_path.slx',
        variable_name_out='y',
        start_time=0,
        stop_time=10,
        solver='ode45',
        max_step=0.01,
        G='tf([2],[1 3 2])',
        K=5
    )

    # Zugriff auf Zeitvektor und Signale
    t = mat.t
    values = mat.values
```

### Simulationsergebnisse plotten
```python
mat.plot_simulation(
    figure_name="step_response",
    title="Step Response",
    path="C:/Users/Me/Plots",
    save=True,
    show=True
)
```

---

## 🔄 Datentypen zwischen Python und MATLAB

- **Python → MATLAB**:
  ```python
  import matlab
  data = matlab.double([1, 2, 3, 4, 5])
  ```
- **MATLAB → Python**:
  ```python
  py_list = list(data)
  ```

---

## ⚡ Tipps

- MATLAB schneller ohne GUI starten:
  ```python
  eng = matlab.engine.start_matlab("-nojvm -nodisplay")
  ```
- Überprüfe die Python-Umgebung, damit die Engine in der richtigen Umgebung installiert wird.
- `MatlabWrapper` unterstützt beliebige MATLAB Workspace-Variablen über `kwargs`.
- Die Engine kann auch in Jupyter Notebooks genutzt werden.
