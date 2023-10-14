# 2d forward modeling in acoustic case

The code of this section locates at `examples/forward_modeling2d`. This example shows how to run forward modeling with your own model and geometry.

-   First we need to change the directory into it.

    ```shell
    cd examples/forward_modeling
    ```

-   Generate a two layer model and the corresponding sources-receivers pairs by typing:

    ```shell
    python generate_model_geometry.py
    ```

    Two new folders **geometry** and **velocity_model** will be created. The figure **model_geometry.png** illustrates the generated layer model and the locations of source-receiver pairs.

    ![Geometry](figures/forward_modeling2d/model_geometry.png)

-   Running the shell script `forward.sh`, a file called `shot_gather.npy` will be created.

    ```shell
    sh forward.sh
    ```

-   Show the shot gathers.

    ```shell
    python show_shotgather.py
    ```

    The plotted results will be save in **shot_gather.png** (shown as follows).
    
    ![ShotGather](figures/forward_modeling2d/shot_gather.png)


# 2d forward modeling in elastic case

The code of this section locates at `examples/elastic_modeling2d`. This example shows how to run forward modeling using elastic wave equation.

-   Step1: Generate a two layer model and the corresponding sources-receivers pairs by typing:

    ```shell
    python generate_model_geometry.py
    ```

    Two new folders **geometry** and **velocity_model** will be created. The figure **model_geometry.png** illustrates the generated layer model and the locations of source-receiver pairs.

    ![Geometry](figures/elastic_modeling2d/model_geometry.png)

-   Step2: Running the shell script `forward.sh`, a file called `shot_gather.npy` will be created.

    ```shell
    sh forward.sh
    ```
-   Step3: Show the shot gathers.

    ```shell
    python show_shotgather.py
    ```

    The plotted results will be save in **shot_gather.png** (shown as follows).
    
    ![ShotGather](figures/elastic_modeling2d/shot_gather.png)


# 3d forward modeling in acoustic case

The code of this section locates at `examples/forward_modeling3d`. This example shows how to run forward modeling with your own model and geometry.

- **Generate geometry and velocity model**

    The script `generate_model_geometry.py` generates a 3D velocity model with two layers. A ricker source at the center of the model suface is used for modeling. Moreover, we have created a three-dimensional observational system, and a schematic diagram of this observational system will be generated after running this script.

    ```shell
    python generate_model_geometry.py
    ```

    ![Geometry](figures/forward_modeling3d/model_geometry.png)

- **Run forward modeling**

    Perform simulation by running script `forward.sh`.

    ```shell
    sh forward.sh
    ```

- **Show results**

    The modeled data has 1 shot with 2000 time samples, 128 traces and a single component (displacement in scalar wave equation). The first 64 and last 64 traces are recorded along different line directions. Run the script will show the recorded data.

    ```shell
    python show_shotgather.py
    ```
    ![Geometry](figures/forward_modeling3d/shot_gather.png)


If you wanna generate your own 3D geometry and 3D velocity model, please refer to the section [data format](data_format.md).