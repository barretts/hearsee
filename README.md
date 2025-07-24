<<<<<<< HEAD
# hearsee
playing with the idea of visualization through sound
=======
# HearSee - An Image Sonification Tool 🎧

HearSee is a real-time sensory substitution tool that translates visual images into an auditory soundscape. Using a webcam for face tracking or a mouse for control, users can "scan" an image and hear a rich, multi-layered representation of its colors. The goal is to create an intuitive and emotionally resonant experience, moving beyond simple technical translation to something that can be genuinely felt and understood.

## ✨ Features

* **HSL-Based Sonification:** Colors are not just mapped to simple tones. They are translated based on the HSL (Hue, Saturation, Lightness) model:
  * **Hue** determines the fundamental **pitch** of the sound.
  * **Lightness** controls the **volume** (amplitude).
  * **Saturation** adjusts the **timbre** (richness or complexity) of the sound.
* **Dynamic Blending:** An inverse-square falloff blending system allows the user to perceive the "texture" of colors surrounding their focus point. Pixels closer to the cursor have a much stronger influence, creating a natural sense of proximity.
* **Global Mix Mode:** Users can toggle a "Global Mix" to hear a quiet, ambient representation of the entire image's average color, layered underneath the detailed sound of their focus point. This provides both detailed exploration and overall context.
* **Intuitive Audio Cues:** The sound model includes special tuning for a more human feel, such as a calmer, lower-frequency sound for blues and a "glare" effect for intensely bright and saturated colors.
* **Flexible Input:**
  * **Face Tracking:** Uses your computer's webcam and MediaPipe to track your head movement, allowing you to naturally look around an image.
  * **Mouse Control:** If no webcam is available, the system seamlessly falls back to mouse input.
* **Load Your Own Images:** Load any JPG or PNG image from your computer to explore its unique soundscape.

## 🚀 Getting Started

### Prerequisites

You will need Python 3.8 or newer. You can check your version with `python --version`.

### Installation

1. **Clone or download the project.**
2. **Navigate to the project directory** in your terminal.
3. **Install the required libraries:**
   ```
   pip install pygame numpy opencv-python mediapipe

   ```

### Running the Application

To start the application, simply run the Python script from your terminal:

```
python hearsee.py

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
>>>>>>> c49346b (add readme.md)
