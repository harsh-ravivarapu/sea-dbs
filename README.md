# SEA-DBS Setup Guide

This repository provides a setup for the **SEA-DBS** environment. Follow the instructions below to create a virtual environment, install dependencies, and run the training script.

## Step 1: Create a Virtual Environment

To begin, create a virtual environment called `sea-dbs`:

```bash
python -m venv sea-dbs
```

Activate the virtual environment:

- **Windows**:
    ```bash
    .\sea-dbs\Scripts\activate
    ```

- **Linux/Mac**:
    ```bash
    source sea-dbs/bin/activate
    ```

## Step 2: Install Dependencies

Once your virtual environment is activated, install the required dependencies using the wheel file. Run the following command:

```bash
pip install --no-index --find-links=wheelhouse -r requirements.txt
```

## Step 3: Install MATLAB Engine for Python

To enable the integration of MATLAB with Python, the MATLAB engine for Python must be installed.

### Install MATLAB Engine for Python:

Follow the installation steps as described in the [MATLAB documentation](https://www.mathworks.com/help/matlab/matlab_external/install-the-matlab-engine-for-python.html).

### Install Compatible Version:

Depending on your MATLAB version, you must install the compatible MATLAB Engine version. To check which version of MATLAB is installed on your system, refer to the [release history on PyPI](https://pypi.org/project/matlabengine/).

Once you've determined the correct version, install it using the following command:

```bash
python -m pip install matlabengine==23.2.1
```

Replace `23.2.1` with the appropriate version number compatible with your MATLAB installation.

## Step 4: Verify MATLAB Engine Integration

To verify that the MATLAB engine is correctly integrated with Python, perform the following steps:

1. Open a Python shell:

    ```bash
    python
    ```

2. Run the following Python code to check if the MATLAB engine is working:

    ```python
    import matlab.engine
    eng = matlab.engine.start_matlab()
    result = eng.sqrt(matlab.double([4]))
    print(result)
    ```

The expected output should be:

```python
[2.0]
```

If you see this output, the MATLAB engine has been successfully integrated with Python!

## Step 5: Running the Training Script

The **`rl_new.py`** script is the main training script for the project. In this script, the hyperparameters in the `main` function can be altered to adjust the training process and save the model.

To run the training script, use the following command:

```bash
python rl_new.py
```

This will initiate the training process based on the current hyperparameters and save the trained model.

