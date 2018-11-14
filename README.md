# SVM-Isosurface3d
Implementation of nonlinear SVM (with SMO algorithm), as well as isosurface visualization (using mayavi)

- Basics:

  - Quadratic programming involved:

    \max_\alpha W(\alpha)= \max_\alpha \left[ \sum_{i=1}^{n}\alpha_i - \frac{1}{2} \sum_{i,j=1}^{n} y_i y_j \alpha_i \alpha_j K(x_i, x_j) \right]

    s.t. ~~~~ 0 \le \alpha_i \le C ~~~~~ i=1,\cdots,n

    \sum_{i=1}^{n} \alpha_i y_i =0


  - Kernel trick used: *RBF Kernel* $K(x,z) = \exp(-\frac{\left\|x-z\right\|^2}{2\sigma^2})$;

  - Stopping condition of SMO iteration: *KKT conditions* are satisfied.

- Snapshot / sketches:
  - `SVM.py`: 
    1. read data from `data` directory; 
    2. solve the corresponding *quadratic programming* using *SMO algorithm*;
    3. finally obtain and save the result array ($\alpha$, $b$) to `tmp` directory.
  - `SVM_isosurface.py`:
    1. load array ($\alpha$, $b$) from `tmp` directory;
    2. plot scatter data points;
    3. plot three 3d isosurfaces representing one separating surface and two margin surfaces using [mayavi](https://docs.enthought.com/mayavi/mayavi/) package.

- Dependencies:
  - `SVM.py`: `numpy`, `scipy`, `collections`, `math`, `os`, `argparse`
  - `SVM_isosurface.py`:  `numpy`, `scipy`, `math`, `os`, `mayavi`

- Optional Arguments:
  - `SVM.py`: 
    - `-s`, `--sigma`: parameter in RBF Kernel (float, >0)
    - `-c`, `--C`: penalty term (float, >0)

- Notes:
  - As far as I know, `mayavi` is now only available through Python2. **Please run `SVM_isosurface.py` with Python2.7** instead of Python3; However, `SVM.py` is okay for both;
  - This repository currently only serves to deal with 3d input data. Further changes will be made in order to render it suitable for 2d data classification and isoline plotting in the future.
  - `SVM_isosurface.py` cannot be run until `SVM.py` has been run at least once;
  - Configurations reside in`config.py`, you may change some of them at your pleasure;
  - Data file should be placed in `data` directory and formatted according to the example data file: every line stands for one data point, with the first three fields representing its co-ordinates and the last field its label (either `0` or `1`); Once your data file is placed, don't forget to modify `data_file_name` in `config.py`.
  - The result graph for the example data file is also included in `data` directory.
  
  
