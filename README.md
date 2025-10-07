# 🚀 Path Analyzer 9000 🚀

Ever wanted to turn a squiggly line in a picture into a real-world, drivable path for your robot? Tired of manually plotting waypoints? Then welcome to the Path Analyzer 9000!

This tool is your one-stop-shop for autonomous path creation. It lets you take any image with a line on it, and with a few clicks, it magically extracts that line, smooths it out, and breaks it down into intelligent segments of straights and turns. It calculates the perfect speed for every twist and curve, then spits out a simple command file that your robot's controller can follow.

## ✨ Features

*   **Load Any Image:** Grab a photo of a track, a drawing on paper, or chalk on the floor.
*   **Interactive Color Tuner:** Easily isolate the path from the background with simple sliders. No need to be a Photoshop wizard!
*   **Intelligent Path Finding:** Automatically finds the single, clean centerline of your path, whether it's thick or thin.
*   **Click-to-Select Endpoints:** You're the boss. Just click where the path starts and ends.
*   **Buttery-Smooth Curves:** Converts jagged pixel lines into a beautiful, mathematically perfect spline curve.
*   **Physics-Based Speed Profiling:** Calculates the ideal speed and curvature for every point on the path based on real-world physics (like friction and G-forces). This isn't just a path, it's a *race line*!
*   **Interactive Segment Tuner:** A powerful pop-up window where you can visually tune how the path is broken into straights and turns in real-time. What you see is what you get!
*   **Smart Segment Export:** Generates a ready-to-use `PATH.TXT` file with a list of high-level commands (`DISTANCE,SPEED,CURVATURE`), perfect for a simple robot controller.

## 🤔 The Secret Sauce: How It Works

The magic happens in a few key steps:

1.  **Color Sorcery (Image Processing):** First, we convert the image to the HSV color space, which is great for isolating colors. Using the `Color Tuner`, you create a mask that highlights just your path.

2.  **Finding the Centerline (Skeletonization):** The solid shape of your path is then put on a diet! A skeletonization algorithm whittles it down to its bare bones—a perfect, one-pixel-wide centerline.

3.  **Connecting the Dots (Pathfinding):** With a cloud of skeleton points, you click on your desired start and end. The app uses **Dijkstra's algorithm** (like a mini-GPS) to find the absolute shortest path between your two clicks.

4.  **Making it Smooth (Spline Interpolation):** The jagged list of pixel coordinates is transformed into a smooth, continuous curve using B-splines. This gives us a path that a real vehicle can actually follow without getting jerky.

5.  **Adding the Physics (Path Properties):** This is where the real brains are. The tool analyzes the path's geometry to calculate its **curvature** (how tight a turn is). Using physics parameters, it then calculates the maximum safe speed for every single point.

6.  **Intelligent Segmentation (The Fun Part!):** Instead of just dumping a thousand tiny waypoints, the tool analyzes the curvature data. It intelligently groups the path into logical segments: **Straights** and **Arcs (Turns)**. You can tune this process in real-time in the **Segment Tuner** to get the perfect balance for your robot.

## 🔧 Setup & Installation

Getting started is easy. You'll need Python 3 installed.

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/your-repo-name.git
    cd your-repo-name
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
    *   Click the `1. Load Image` button and select an image file.

2.  **Step 2: Tune Color Range**
    *   Click `2. Tune Color Range...`.
    *   Play with the `H`, `S`, and `V` sliders until the path is **bright white** and everything else is **black**.
    *   Click `Apply`.

3.  **Step 3: Calibrate Scale**
    *   This is important! Click `Calibrate Pixel/Meter Scale...`.
    *   Click on two points in your image with a known real-world distance (e.g., the ends of a 1-meter ruler).
    *   Enter the distance in meters when prompted. The app now knows how big things are.

4.  **Step 4: Analyze & Select Endpoints**
    *   Click `3. Analyze & Select Endpoints`. The app will overlay cyan nodes on the path's centerline.
    *   Click your desired **START node**, then click your **END node**. The final smooth path will be drawn in red.

5.  **Step 5: Tune and Export!**
    *   Click the big green `4. Tune and Export Path...` button.
    *   The **Interactive Segment Tuner** window will appear. You'll see your path colored in alternating shades of cyan (straights) and magenta (turns).
    *   **Play with the sliders:**
        *   **`Straightness Threshold`:** A higher value makes the app more "generous" about what it considers a straight line. Lower it to detect even very gentle curves as turns.
        *   **`Minimum Segment Length`:** A higher value merges tiny, noisy segments into larger ones, cleaning up the path.
    *   Watch the path colors change in real-time!
    *   Once you're happy with the visual segmentation, click the **`Generate PATH.TXT`** button at the bottom of the tuner.
    *   Choose a file name and location, and you're done!

## 📄 The Golden Output: Your `PATH.TXT` File

The exported file is a simple list of commands, ready for your Arduino or other microcontroller. Each line is a segment with three values:

| Column    | Description                                                                 | Unit      |
|-----------|-----------------------------------------------------------------------------|-----------|
| `DISTANCE`  | The length of this segment.                                                 | meters    |
| `SPEED`     | The calculated average safe speed for this segment.                         | m/s       |
| `CURVATURE` | The average tightness of the curve for this segment (0.0 for straights).    | 1/meter   |

Your robot's code can now read this file line by line. For each line, it will drive forward while applying a turn proportional to the `CURVATURE` value, stopping only when its wheel encoders report that it has traveled the specified `DISTANCE`.

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
