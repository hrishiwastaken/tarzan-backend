# 🚀 Path Analyzer 9000 🚀

Ever wanted to turn a squiggly line in a picture into a real-world, drivable path for your robot? Tired of manually plotting waypoints? Then welcome to the Path Analyzer 9000!

This tool lets you take any image with a line on it, and with a few clicks, it magically extracts that line, smooths it out, and calculates the perfect speed for every twist and turn. It then spits out a simple CSV file that your robot's controller can follow.

## ✨ Features

*   **Load Any Image:** Grab a photo of a track, a drawing on paper, or chalk on the floor.
*   **Interactive Color Tuner:** Easily isolate the path from the background with simple sliders. No need to be a Photoshop wizard!
*   **Intelligent Path Finding:** Automatically finds the single, clean centerline of your path, whether it's thick or thin.
*   **Click-to-Select Endpoints:** You're the boss. Just click where the path starts and ends.
*   **Buttery-Smooth Curves:** Converts jagged pixel lines into a beautiful, mathematically perfect spline curve.
*   **Physics-Based Speed Profiling:** Calculates the ideal speed and curvature for every point on the path based on real-world physics (like friction and G-forces). This isn't just a path, it's a *race line*!
*   **Export to CSV:** Generates a ready-to-use `.csv` file with coordinates, speed, and curvature, perfect for feeding into your vehicle's control system.
*   **Helpful Diagnostic Tools:** Calibrate your image scale (pixels to meters) and interactively inspect the path's slope and angle.

## 🤔 The Secret Sauce: How It Works

The magic happens in a few key steps:

1.  **Color Sorcery (Image Processing):** First, we convert the image to the HSV color space, which is great for isolating colors. Using the `` `Color Tuner` ``, you create a mask that highlights just your path. The app is smart enough to find the biggest colored shape and fill it in, so it doesn't matter if your line is thick or thin.

2.  **Finding the Centerline (Skeletonization):** The solid shape of your path is then put on a diet! A skeletonization algorithm whittles it down to its bare bones—a perfect, one-pixel-wide centerline. This ensures you get a single, clean path of nodes.

3.  **Connecting the Dots (Pathfinding):** With a cloud of skeleton points, you click on your desired start and end. The app uses **Dijkstra's algorithm** (like a mini-GPS) to find the absolute shortest path between your two clicks, flawlessly navigating any branches or forks in the skeleton. No more weird glitches!

4.  **Making it Smooth (Spline Interpolation):** The jagged list of pixel coordinates is transformed into a smooth, continuous curve using B-splines. This gives us a path that a real vehicle can actually follow without getting jerky.

5.  **Adding the Physics (Path Properties):** This is where the real brains are. The tool analyzes the smoothed path's geometry at every point to calculate its **curvature** (how tight a turn is). Using parameters like the coefficient of friction (`` `mu` ``), it then calculates the maximum safe speed for every single point on the path. This means your robot will automatically know to slow down for sharp corners and speed up on the straights!

## 🔧 Setup & Installation

Getting started is easy. You'll need Python 3 installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/hrishiwastaken/path-analyzer-backend.git
    cd path-analyzer-backend
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    # For Mac/Linux
    python3 -m venv venv
    source venv/bin/activate

    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the dependencies:**
    Create a file named `requirements.txt` in the project folder and add the following lines:
    ```
    opencv-python
    numpy
    scipy
    matplotlib
    Pillow
    ```
    Then, run this command in your terminal:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the app!**
    ```bash
    python path_analyzer.py
    ```

## 🗺️ A Step-by-Step Adventure: How to Use It

Follow the workflow buttons from top to bottom!

1.  **Step 1: Load Image**
    *   Click the `` `1. Load Image` `` button and select an image file (`.jpg`, `.png`, etc.).

2.  **Step 2: Tune Color Range**
    *   Click `` `2. Tune Color Range...` ``. A new window will pop up.
    *   Play with the `` `H` ``, `` `S` ``, and `` `V` `` sliders until the path you want to track is **bright white** and everything else is **black**.
    *   Click `` `Apply` `` when you're happy.

3.  **Step 3: Calibrate Scale**
    *   If you want your final CSV to be in real-world meters, click `` `Calibrate Pixel/Meter Scale...` ``.
    *   Click on two points in your image (e.g., the start and end of a ruler or a known-length object).
    *   A dialog box will ask for the real-world distance between those two points in meters. Enter it. Voilà, the app now knows how big things are!

4.  **Step 4: Analyze & Select Endpoints**
    *   Click `` `3. Analyze & Select Endpoints` ``. The app will process the image and overlay a grid of cyan nodes on the path's centerline.
    *   The status bar will ask you to **click on the START node**. Hover over a node (it will turn yellow) and click.
    *   The status bar will then ask you to **click on the END node**. Hover and click again.
    *   The app will process the selection and draw the final smooth path in red.

5.  **Fine-Tune the Path (Optional)**
    *   Use the **"Path Endpoint Control"** slider to shorten or lengthen the path from the end point.
    *   Use the **"Number of Nodes to Display"** slider to change how many cyan dots you see (this is just for visualization).

6.  **Step 5: Export Path to CSV**
    *   Use the **"Export Path Resolution"** slider to decide how many points you want in your final CSV file (100 is a good start).
    *   Click the big green `` `4. Export Path to CSV` `` button.
    *   Choose a file name and location, and you're done!

## 📄 The Golden Output: Your CSV File

The exported CSV file is simple and ready for your robot's controller. It contains four columns:

| Column      | Description                                     | Unit    |
|-------------|-------------------------------------------------|---------|
| `` `x_m` ``       | The X-coordinate of the path point.             | meters  |
| `` `y_m` ``       | The Y-coordinate of the path point.             | meters  |
| `` `speed_mps` `` | The calculated maximum safe speed at that point.| m/s     |
| `` `curvature` `` | The tightness of the curve at that point.       | 1/meter |

Your robot can now read this file line by line, moving to the next `(x, y)` coordinate while aiming for the target `speed_mps`. Although it will need a seperate .cpp program to use this raw data and translate it to something dynamic to the car.
The .cpp program for a basic car will be added here soon. For now though, you're on your own ;)

## ⚙️ Pro Tips & Tuning Knobs

Want to supercharge your path generation? You can edit the physics parameters directly in the code!

In the `export_path_data` function, look for the `planning_params` dictionary:
```python
planning_params = {
    'g': 9.81,          # Gravity (m/s^2)
    'mu': 0.8,          # Coefficient of friction (e.g., 0.8 for grippy tires on asphalt)
    'safety_factor': 0.7, # A number less than 1 to stay well below the physical limit
    'v_max': 2.5,       # The absolute top speed of your vehicle (m/s)
    # ... other advanced params
}
```
*   **`` `mu` `` (Friction):** This is the most important one! Higher `` `mu` `` means more grip, allowing for higher speeds in corners. Lower it for slippery surfaces.
*   **`` `v_max` `` (Max Velocity):** Set this to your robot's top speed. The calculated speeds will never exceed this value.

---
