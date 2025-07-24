# **HearSee Project: Development Plan**

This document outlines the strategic development plan for the HearSee project, translating the foundational research from the "Auditory Portraits" framework into a series of actionable engineering phases. The goal is to evolve HearSee from a color sonification tool into a sophisticated, emotionally resonant, and inclusive sensory substitution device.

## **Phase 1: Implementing Advanced Auditory Palettes 🎨**

This phase focuses on moving beyond the current HSL model and implementing the three distinct, perceptually-grounded sound models proposed in Part I of the research.

### **1.1. Core Audio Engine Refactor**

* **Action:** Abstract the current color\_to\_sound\_properties function into a modular "Sound Model" architecture.  
* **Goal:** Create a system where different mapping strategies can be selected and activated by the user without altering the core application logic.

### **1.2. Implement the Three Sound Models**

* **Action:** Create three new sound model modules based on the research:  
  1. **Naturalistic Model:** Map lightness to pitch, saturation to timbral purity (sine vs. complex wave), and hue to the timbral base.  
  2. **Symbolic/Musical Model:** Map hue to the circle of fifths (major/minor keys), saturation to harmonic richness (adding instruments/notes), and lightness to the octave.  
  3. **Direct Emotional Model:** Map hue to harmonic dissonance, saturation to tempo/attack, and lightness to loudness/spectral centroid, using the psychoacoustic table as a guide.  
* **Goal:** Provide the user with a diverse set of auditory experiences to choose from, each offering a different way to interpret the visual scene.

### **1.3. User Interface for Model Selection**

* **Action:** Assign number keys (e.g., 1, 2, 3\) to allow the user to switch between the Naturalistic, Symbolic, and Emotional models in real-time. Update the on-screen info panel to display the currently active model.  
* **Goal:** Make model selection seamless and intuitive, encouraging user experimentation.

## **Phase 2: Sonifying Form and "Hearing a Smile" 😊**

This phase implements the computer vision pipeline detailed in Part II of the research to translate geometric shapes and facial features into a language of "auditory gestures."

### **2.1. Advanced Feature Extraction**

* **Action:** Enhance the existing MediaPipe integration to not only get a single point but to extract the full set of 478 facial landmarks.  
* **Action:** Develop functions to calculate key geometric features from these landmarks in real-time:  
  * Curvature of the upper and lower lip lines.  
  * Mouth width-to-height ratio.  
  * Presence of "crow's feet" by analyzing the compression of landmarks around the outer eyes.  
* **Goal:** Create a live data stream of quantitative facial expression metrics.

### **2.2. Building the Auditory Gesture Lexicon**

* **Action:** Create a new audio synthesis module dedicated to generating the short, distinct sound events ("earcons" or "auditory gestures") from the research lexicon.  
  * **Sharp Curve:** A rapid, upward pitch sweep (glissando).  
  * **Sharp Corner:** A percussive, staccato "pluck."  
  * **Texture:** A granular or noisy sound layer.  
* **Goal:** Develop the sonic building blocks needed to represent form.

### **2.3. Synthesis: The Smile Signature**

* **Action:** Create a trigger system that detects the combined visual signature of a smile (mouth corners moving up, lip curvature increasing).  
* **Action:** When triggered, the system will play the composed auditory event: two staccato "plucks" for the mouth corners, followed by an upward glissando representing the smile's curve.  
* **Goal:** Achieve the project's central objective of creating a distinct, recognizable auditory event for a smile.

## **Phase 3: Building the Inclusive and Adaptive Framework 🌍**

This final phase implements the crucial two-stage adaptive model from Part III to ensure the tool is robust, fair, and effective across all users and lighting conditions.

### **3.1. Integrate Perceptually Uniform Color Space**

* **Action:** Add the CIELAB color space to the image processing pipeline. All color analysis will be performed in CIELAB, leveraging its perceptual uniformity.  
* **Goal:** Align the technical foundation with the principles of human perception.

### **3.2. Stage 1: Implement Global Palette Adaptation**

* **Action:** On loading a new image or re-calibrating, the system will perform the "Stage 1" analysis:  
  1. Detect the primary face.  
  2. Perform skin segmentation within the face bounding box to identify the dominant cluster of skin pixels in CIELAB space.  
  3. Calculate the mean L\*, a\*, b\* of this cluster to establish the **auditory baseline**.  
* **Goal:** Create a dynamic, personalized calibration that adapts to the specific subject and lighting.

### **3.3. Stage 2: Sonify Relative to Baseline**

* **Action:** Refactor the sonification engine to operate in "Stage 2" mode. For any pixel, it will now sonify its **relative difference** from the calculated auditory baseline, not its absolute color value.  
* **Action:** Layer the auditory gestures from Phase 2 on top of this adaptive textural sound.  
* **Goal:** Create a sonification that focuses on the universal language of expression (changes in shadow, highlight, and shape) rather than superficial color, making the tool fundamentally inclusive and robust.