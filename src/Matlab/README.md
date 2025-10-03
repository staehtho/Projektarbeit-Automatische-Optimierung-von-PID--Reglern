 MATLAB Engine API for Python & MatlabInterface Usage
 ----------------------------------------------------
 This script demonstrates installation and usage of the MATLAB Engine API for Python,
 as well as the MatlabInterface class to start MATLAB, run Simulink simulations,
 and plot results.
 ----------------------------
 📋 Requirements
 ----------------------------
 - MATLAB installed (R2014b or newer)
 - Python installed (3.7 – 3.11 recommended)
 - Both MATLAB and Python must have the same architecture (e.g., 64-bit)
 - Optional: matplotlib and numpy for plotting
   pip install matplotlib numpy
 ----------------------------
 🔧 Installing MATLAB Engine API
 ----------------------------
 1. Open terminal / Anaconda Prompt
 2. Navigate to MATLAB engine Python folder:
    ``` bash
    cd "MATLABROOT/extern/engines/python"
    ```

 3. Install:
    python setup.py install
 4. Test:
    ``` bash
    python -c "import matlab.engine; print('MATLAB Engine successfully installed!')"
    ```
 ----------------------------
 🚀 MATLAB Engine Basic Example
 ----------------------------
``` python
import matlab.engine

 Start MATLAB
eng = matlab.engine.start_matlab()

 Example calculation
result = eng.sqrt(16.0)
print("Result:", result)

 Quit MATLAB
eng.quit()
```
 ----------------------------
 🛠 MatlabInterface Usage
 ----------------------------
 Assuming MatlabInterface.py is in the project directory
``` python
from MatlabInterface import MatlabInterface

 Run a Simulink simulation
with MatlabInterface() as mat:
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

    # Access time vector and signal values
    t = mat.t
    values = mat.values

 Plot simulation results
mat.plot_simulation(
    figure_name="step_response",
    title="Step Response",
    path="C:/Users/Me/Plots",
    save=True,
    show=True
)
```
 ----------------------------
 🔄 Data type conversion Python <-> MATLAB
 ----------------------------
 Python to MATLAB
``` python
import matlab
data = matlab.double([1, 2, 3, 4, 5])

 MATLAB to Python
py_list = list(data)
```
 ----------------------------
 ⚡ Tips
 ----------------------------
 Start MATLAB faster without GUI
``` python
eng = matlab.engine.start_matlab("-nojvm -nodisplay")
```
 Ensure correct Python environment is used for installation
 MatlabInterface supports arbitrary workspace variables via kwargs
 MATLAB Engine can be used in Jupyter Notebooks as well
