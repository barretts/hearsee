# HearSee - An Image Sonification Tool 🎧

HearSee is a real-time sensory substitution tool written in Python that translates visual images into an intuitive and emotionally resonant auditory experience. Using a webcam or a static image, the application analyzes color, form, and facial expressions, converting them into a multi-layered soundscape.

The project is designed to be a flexible platform for exploring different methods of sonification, with the ultimate goal of creating a system that can help a user "hear" the key features of an image, including the mood of a landscape or the smile on a person's face.

## Core Features

* **Real-Time Sonification:** Translates images from a webcam or file into sound with minimal latency.
* **Mouse and Face Control:** Navigate the image using either standard mouse input or by using our own face as a pointer via the webcam.
* **Multiple Sound Models:** Switch between different modes of sound generation on the fly to explore various ways of interpreting visual data:
    1.  **Naturalistic Model:** An intuitive mapping where color hue, saturation, and lightness control the sound's pitch, timbre, and volume.
    2.  **Symbolic/Musical Model:** Translates colors into musical chords based on the circle of fifths, creating a more traditionally harmonic soundscape.
    3.  **Direct Emotional Model:** Uses psychoacoustic principles to map color properties directly to sound characteristics that evoke specific emotions like joy, tension, or calm.
* **Auditory Gestures:** The system can detect specific geometric shapes and overlay distinct sound events. Currently, it can detect smiles and represent them with a bright, upward-rising arpeggio.
* **Contextual Audio Modes:**
    * **Global Mix:** Toggle a mode to hear a quiet background tone representing the image's average color, providing ambient context.
    * **Spatial Blending:** Adjust the "focus" of your auditory lens, from a sharp point to a wide, atmospheric field that blends the sounds of nearby colors.
* **Custom Image Loading:** Load and explore any JPG or PNG image from your computer.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone <your-repo-url>
    cd <your-repo-directory>
    ```
2.  **Install dependencies:** This project requires Python 3 and several libraries. You can install them using pip:
    ```bash
    pip install pygame numpy opencv-python mediapipe
    ```

## How to Use

1.  **Run the script:**
    ```bash
    python hearsee.py
    ```
2.  The application will start. By default, it will attempt to use your webcam for face tracking. If no webcam is found, it will switch to mouse control.
3.  Use your face or mouse to move the cursor around the image and listen to the sound change.
4.  Use the keyboard controls below to change settings and models.

### Controls

| Key           | Action                                                    |
| :------------ | :-------------------------------------------------------- |
| `Mouse`       | Controls the cursor (if no webcam is active).             |
| `Face`        | Controls the cursor (if webcam is active).                |
| `1`, `2`, `3` | Switch between Naturalistic, Symbolic, and Emotional models. |
| `L`           | Load a new image from your computer.                      |
| `G`           | Toggle the "Global Mix" background sound.                 |
| `TAB`         | Toggle spatial audio blending on or off.                  |
| `UP`/`DOWN`   | Adjust blending intensity.                                |
| `LEFT`/`RIGHT`| Adjust blending radius (focus).                           |
| `SPACE`       | Re-center the face tracking cursor.                       |
| `\`` (Backtick) | Show or hide the detailed info panel.                     |
| `ESC`         | Quit the application.                                     |


## Future Development

<<<<<<< HEAD
```

A window will appear displaying the default image, and the program will attempt to activate your webcam for face tracking.

## 🎮 Controls

* **L Key:** **L**oad a new image from your computer.
* **G Key:** Toggle **G**lobal Mix mode on or off.
* **TAB Key:** Toggle the spatial sound **B**lending on or off.
* **SPACE Key:** (Face Tracking Mode) Re-calibrate the center point to your current head position.
* **UP/DOWN Arrows:** Increase/Decrease the **intensity** of the sound blending.
* **LEFT/RIGHT Arrows:** Decrease/Increase the **radius** of the sound blending.
* **ESC Key:** **Esc**ape and quit the application.

## 🔮 Future Goals

The ambition of HearSee is to create a tool capable of conveying complex visual information, including emotion and form. Future development is focused on:

* **Hearing a Smile:** Implementing computer vision techniques (like edge and feature detection) to translate not just color, but the *shapes* and *forms* of an image into distinct auditory cues, such as a rising pitch for the curve of a smile.
* **Emotional Sound Palettes:** Researching and implementing more advanced sound models based on psychoacoustics to better convey emotions like joy, tension, or calm.
* **Inclusive Design:** Developing an adaptive framework that can analyze the primary color palette of an image (e.g., different skin tones in a portrait) and adjust its sonification model to best highlight the subtle features within that specific context, ensuring the tool is effective for all users.
