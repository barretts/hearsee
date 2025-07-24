

# **Auditory Portraits: A Framework for Emotionally Resonant and Inclusive Sonification of the Human Face**

## **Executive Summary**

This report provides a comprehensive technical framework for the development of a real-time sensory substitution application that translates visual images of human faces into a rich, intuitive, and emotionally resonant auditory experience for visually impaired users. Moving beyond simplistic color-to-sound mappings, this document outlines a sophisticated, multi-layered approach grounded in principles of perceptual science, psychoacoustics, computer vision, and inclusive AI design. The proposed system architecture is designed to be not only perceptually coherent and emotionally communicative but also fundamentally adaptive and fair, ensuring robust performance across diverse user populations and environmental conditions.

The framework is presented in three parts. **Part I: The Auditory Palette** establishes the foundational principles of sound design, drawing a parallel between perceptually uniform colormaps in data visualization and the necessity of using perceptually linear scales for pitch and loudness in sonification. It proposes three distinct color-to-sound mapping models—"Naturalistic," "Symbolic/Musical," and "Direct Emotional"—and provides a detailed psychoacoustic codex for conveying core emotions through specific sonic parameters.

**Part II: Sonifying Form** addresses the challenge of translating shape, texture, and expression into sound. It details a computer vision pipeline for extracting salient facial features, such as the curve of a smile, and introduces a lexicon of "auditory gestures" that map these visual primitives to intuitive sonic events. This section culminates in a synthesized approach for enabling a user to "hear a smile" as a complex, recognizable auditory signature.

**Part III: An Inclusive and Adaptive Framework** confronts the critical issue of algorithmic bias. It deconstructs the limitations of standard approaches and proposes a novel two-stage adaptive sonification model. This model first performs a global analysis of an image to establish a personalized, context-aware auditory baseline calibrated to the specific subject's skin tone and the ambient lighting. It then sonifies features relative to this baseline, ensuring the system focuses on the universal language of facial expression rather than superficial, variable color values. This adaptive strategy represents a crucial step toward creating a truly equitable and universally effective assistive technology.

---

## **Part I: The Auditory Palette: From Perceptual Uniformity to Emotional Resonance**

### **Introduction to Part I**

The creation of a successful sensory substitution device hinges on the design of its core sensory language. The auditory experience must be more than a raw data stream; it must be a coherent, non-fatiguing, and interpretable sound world capable of conveying nuanced information. This section establishes the foundational principles for such an experience. A direct parallel exists between the goals of modern scientific data visualization and data sonification: both seek to move away from arbitrary, potentially misleading representations toward mappings grounded in the science of human perception.1 The primary objective is to build an auditory palette that is as perceptually honest and intuitively scalable as the scientific colormaps that inspired this inquiry.

### **1.1 The Foundation: Perceptual Uniformity in Vision and Its Auditory Analogue**

#### **Analysis of Scientific Colormaps (Viridis, Magma, Inferno)**

The scientific colormaps Viridis, Magma, and Inferno were developed by Stéfan van der Walt and Nathaniel Smith to solve a critical problem in data visualization.4 Widely used legacy colormaps like 'jet' and 'rainbow' introduce visual distortions, creating false boundaries and obscuring real features in data.6 The core design principle of the Viridis family of colormaps is

**perceptual uniformity**, which dictates that equal steps in data should correspond to equal perceived steps in color.6

This uniformity is achieved primarily through the careful control of **lightness**. Research has shown that the human brain perceives changes in lightness far more effectively than changes in hue when interpreting ordered data.6 Therefore, these colormaps are designed to have a monotonically and smoothly increasing lightness value (

L∗) when plotted in a perceptually uniform color space such as CIELAB or CAM02-UCS.6 This smooth gradient ensures that the colormap translates effectively to grayscale for printing and is more accessible to individuals with common forms of color vision deficiency.5 The specific bluish-to-reddish-to-yellowish hue progression was chosen because it avoids relying heavily on red-green contrast, making it robust for viewers with deuteranopia and protanopia.9 The use of a formal perceptual model like CAM02-UCS in their design process allowed the creators to make these colormaps analytically, rather than just aesthetically, uniform.9 This rigorous design prevents the perceptual "kinks" and "dead zones" that plague older colormaps, where large changes in data can produce little perceived color change, or vice versa.5

#### **Translating Visual Uniformity to Auditory Congruence**

To create a sonification that honors the perceptual integrity of Viridis, a direct mapping of a physical color property to a physical sound property is insufficient and incorrect. For instance, mapping the lightness value (L∗) linearly to frequency in Hertz (Hz) would reintroduce the very problem Viridis was designed to solve. Human perception of pitch is logarithmic, not linear; the perceived difference between 100 Hz and 200 Hz (an octave) is vastly greater than the perceived difference between 4000 Hz and 4100 Hz. Such a mapping would produce jarring, non-intuitive auditory leaps.

The correct approach is to translate from one perceptual domain to another, ensuring that a perceived change in color maps to a similarly perceived change in sound. This requires the use of psychoacoustic scales that are themselves perceptually uniform.

* **For Pitch:** Instead of linear frequency (Hz), the lightness value from the colormap should be mapped to a psychoacoustic pitch scale, such as the **Mel scale**. The Mel scale is designed so that equal distances along the scale correspond to equal perceived differences in pitch by human listeners. This ensures that a smooth, linear increase in the source data results in a smooth, perceptually linear ascent in pitch.  
* **For Loudness:** A direct mapping to sound pressure level (decibels, dB) is similarly flawed due to the phenomenon described by **equal-loudness contours** (also known as Fletcher-Munson curves).14 Human hearing sensitivity varies dramatically with frequency; a sound at 1000 Hz is perceived as much louder than a sound of the same decibel level at 100 Hz. To achieve perceptually uniform loudness, the mapping should target  
  **Phons**, the unit of perceived loudness. The application's audio engine must dynamically adjust the amplitude (dB) of a tone based on its frequency to maintain a constant perceived loudness level.

By implementing these psychoacoustically-informed mappings, the smooth, non-jarring, and information-rich transitions of the Viridis colormap can be faithfully reproduced in the auditory domain. This creates a sonic gradient that is coherent, intuitive, and non-fatiguing for the user, forming the essential foundation upon which more complex emotional and structural information can be built.

### **1.2 Proposing Three Color-to-Sound Mapping Models**

To move beyond a single mapping strategy and provide the user with a flexible and expressive tool, this report proposes three distinct models for translating color into sound. These models range from the concrete and physical to the abstract and emotional, each leveraging different aspects of cross-modal perception. They are based on the common approach of decomposing color into Hue, Saturation, and Lightness/Value (HSL/HSV), which have been frequently associated with the auditory parameters of pitch, loudness, and timbre in sonification research.15

#### **Model 1: The "Naturalistic" Model**

* **Principle:** This model aims for a highly intuitive, almost physical-feeling output by mapping color properties to sonic parameters that mimic cause-and-effect relationships found in the natural world. It leverages innate or deeply learned cross-modal associations, such as the link between object size and pitch.16  
* **Mapping Strategy:**  
  * **Lightness → Pitch & Spectral Brightness:** Higher lightness values are mapped to a higher fundamental pitch and a brighter timbre (characterized by more energy in higher-frequency harmonics). This mirrors the real-world phenomenon where smaller, lighter objects tend to produce higher-pitched, thinner sounds, while larger, heavier, and often darker objects produce lower, fuller sounds.  
  * **Saturation → Timbral Complexity & Purity:** Saturation, the measure of a color's vividness, is mapped to the purity of the sound's timbre. Low saturation (approaching grey) is represented by a pure sine wave or a sound with very few harmonics, like a simple flute. High saturation (a pure, vivid color) is mapped to a harmonically rich and complex timbre, such as a sawtooth waveform or a bowed string played with significant pressure. This creates a direct analogy between visual purity and auditory (timbral) purity.  
  * **Hue → Timbral Base & Filtering:** Hue determines the fundamental character of the sound's timbre, analogous to the base material of a physical object. The 360-degree color wheel can be mapped to a "circle of timbres." For example, warm colors like reds and oranges could be associated with brassy or slightly distorted waveforms, evoking heat and energy. Cool colors like blues and greens could be mapped to smoother, hollower sounds, such as woodwinds or filtered square waves, evoking air and water.

#### **Model 2: The "Symbolic/Musical" Model**

* **Principle:** This model leverages the highly structured and culturally understood conventions of Western music theory to produce an output that can be both aesthetically pleasing and rich in relational information. Its effectiveness relies on the listener's (often subconscious) familiarity with musical language.16 A pilot study involving both children and adults found a preference for a mapping of Hue-Octave, Saturation-Mode, and Brightness-Root Tone, demonstrating the intuitive appeal of a musically structured approach.15  
* **Mapping Strategy:**  
  * **Hue → Musical Key/Root Note:** The color wheel is mapped directly onto the circle of fifths, a foundational concept in tonal music. A critical refinement is to link color temperature to musical mode: warm colors (red, orange, yellow) are mapped to major keys, which are psychoacoustically associated with positive or happy emotions, while cool colors (blue, green, violet) are mapped to minor keys, associated with sad or melancholic emotions.17  
  * **Saturation → Harmonic Richness/Instrumentation:** Saturation controls the complexity of the musical texture. Very low saturation maps to a simple sound, such as a single instrument playing a basic interval (e.g., root and fifth). As saturation increases, more notes are added to form richer chords (e.g., adding the third, then the seventh), and more virtual instruments are layered into the sound, creating a fuller orchestral texture.  
  * **Lightness → Pitch Register/Octave:** In line with common and intuitive sonification practice, lightness is mapped to pitch register.15 Low lightness values correspond to low octaves (bass notes), and high lightness values correspond to high octaves (treble notes).

#### **Model 3: The "Direct Emotional" Model**

* **Principle:** This model is the most abstract and ambitious, seeking to bypass conventional musical structures entirely. It maps color properties directly to psychoacoustic parameters that have been scientifically shown to elicit specific emotional responses in listeners.14 The goal is not necessarily to create "music" but to generate a soundscape that directly communicates the affective quality of the visual scene.  
* **Mapping Strategy:**  
  * **Hue (as Emotional Valence) → Harmonic Dissonance & Pitch Contour:** Hue is interpreted as a measure of emotional valence. Warm, typically positive hues like yellows and oranges are mapped to consonant harmonic intervals (e.g., major thirds, perfect fifths) and rising pitch contours. Cool, often tense or negative hues like dark blues and purples are mapped to dissonant intervals (e.g., minor seconds, tritones) and descending or erratic pitch contours.  
  * **Saturation (as Emotional Arousal) → Tempo & Attack/Decay:** Saturation is interpreted as a measure of emotional arousal. Low saturation (calm, low arousal) maps to slow tempos, soft sound onsets (slow attack), and long, fading decays. High saturation (excitement, high arousal) maps to fast tempos, sharp and abrupt sound onsets (fast attack), and short, clipped decays.20  
  * **Lightness (as Energy) → Loudness & Spectral Centroid:** Lightness is interpreted as a measure of energy. Low lightness (low energy) corresponds to quiet sounds with a low spectral centroid (a "dull" or "dark" timbre). High lightness (high energy) corresponds to loud sounds with a high spectral centroid (a "bright" or "sharp" timbre).21

### **1.3 A Psychoacoustic Framework for Emotional Sonification**

Psychoacoustics, the study of the perception of sound, provides the scientific underpinning for creating an emotionally resonant auditory experience.17 The emotional impact of sound is not arbitrary but is systematically linked to specific, controllable acoustic parameters.18 By manipulating these parameters, a developer can move from simply representing data to actively shaping the listener's emotional interpretation.

The core acoustic parameters that correlate with emotional expression include:

* **Pitch:** High pitch is broadly associated with happiness, excitement, and tension, while low pitch is linked to sadness, gravity, and seriousness.18  
* **Loudness and Dynamics:** Changes in loudness are powerful emotional cues. A gradual increase in volume (crescendo) builds anticipation and tension, whereas a decrease (diminuendo) can suggest sadness, calmness, or surprise. Loudness is one of the strongest predictors of perceived arousal.14  
* **Tempo and Rhythm:** The speed and pattern of sounds are fundamental to their emotional character. Fast, regular tempos are perceived as exciting, happy, or agitated, while slow, steady tempos are perceived as calm, stately, or sad.18  
* **Timbre and Spectral Content:** The "color" of a sound is a key emotional signifier. Bright timbres, which have more energy in higher frequencies (a high spectral centroid), are associated with joy and excitement. Dull timbres, with energy concentrated in lower frequencies, are linked to sadness or tenderness. Furthermore, timbral complexity matters: simple, pure tones (like a flute) are often perceived as pleasant and peaceful, while complex, harmonically rich, or noisy timbres (like a distorted electric guitar) can convey power, anger, or fear.20  
* **Harmony and Dissonance:** In musical contexts, the relationship between notes is critical. Consonant harmonies, such as major chords, are consistently perceived as pleasant, stable, and happy. Dissonant harmonies, such as minor or diminished chords, evoke feelings of sadness, tension, or unease.14  
* **Envelope (Attack/Decay):** The onset and offset characteristics of a sound shape its perceived character. Sharp, abrupt onsets (fast attack) are associated with surprise, anger, or excitement. In contrast, gentle, gradual onsets (slow attack) are perceived as calm, tender, or sad.20

To translate this research into an actionable engineering blueprint, the following table synthesizes these findings, providing a direct look-up for mapping target emotions to specific sonic parameter settings.

| Emotion | Pitch | Harmony | Tempo/Rhythm | Loudness/Dynamics | Timbre (Spectral) | Attack/Decay |
| :---- | :---- | :---- | :---- | :---- | :---- | :---- |
| **Joy** | High register, upward contours | Consonant, Major key | Fast, regular, lively | Moderately loud to loud, stable | Bright (high spectral centroid), rich in harmonics | Fast/sharp attack, moderate decay |
| **Tension** | Rising pitch, erratic/unstable | Dissonant, minor/diminished chords | Accelerating (accelerando), irregular | Crescendo (gradual increase) | Complex, noisy, sharp | Sharp attack, very short decay |
| **Calm** | Low to mid register, stable/flat contour | Consonant, open intervals (fifths) | Slow, steady, simple | Quiet (pianissimo), stable | Simple (few harmonics, sine-like), smooth | Slow/gentle attack, long decay/release |
| **Sadness** | Low register, downward contours | Dissonant, Minor key | Very slow, sparse | Quiet, slow decrease (diminuendo) | Dull (low spectral centroid), filtered | Slow/gentle attack, long decay |

---

## **Part II: Sonifying Form: Translating Visual Features into Auditory Gestures**

### **Introduction to Part II**

The ultimate ambition of this project—to allow a user to "hear a smile"—necessitates a leap from pixel-level sonification to a more holistic system that can perceive, interpret, and translate geometric shapes, textures, and dynamic facial features. Color and lightness provide the foundational texture of the auditory scene, but form and expression give it meaning and narrative. This section details the computer vision techniques required to deconstruct a facial expression into its constituent parts and provides a lexicon for translating those visual primitives into a language of distinct, intuitive "auditory gestures."

### **2.1 A Computer Vision Toolkit for Facial Feature Extraction**

To sonify a smile, the system must first robustly detect and quantify it. This involves identifying not only the mouth but also its specific curvature, the presence of teeth, the contraction of the orbicularis oculi muscles that create "crow's feet" (the hallmark of a genuine Duchenne smile), and the dynamic movement of facial features over time. This requires a sophisticated computer vision pipeline.

#### **Traditional (Feature-Based) Techniques**

These methods rely on handcrafted features and classical image processing algorithms.

* **Edge and Contour Analysis:** Algorithms like the Canny edge detector are effective at identifying the high-contrast boundaries of the lips and eyes.27 The geometric properties of these detected contours, such as their curvature and length, can be analyzed to classify an expression. While computationally efficient, these methods are often sensitive to variations in lighting, shadows, and image noise.28  
* **Geometric Feature Point Tracking:** This approach identifies a sparse set of key facial landmarks (e.g., corners of the mouth, eyes, tip of the nose) and tracks their spatial relationships. A smile is reliably characterized by the upward and outward displacement of the mouth corners relative to other stable features.27 Optical flow algorithms, such as the Lucas-Kanade method, can be used to compute the motion vectors of these feature points between consecutive video frames, providing a dynamic representation of the expression.27  
* **Appearance-Based Methods:** Instead of discrete points or lines, these methods analyze the textural patterns of entire facial regions. Techniques like Histogram of Oriented Gradients (HOG) capture the distribution of local intensity gradients, while Local Binary Patterns (LBP) describe local texture. The resulting feature vectors can be fed into a classifier like a Support Vector Machine (SVM) to recognize the overall visual pattern of a smile.28

#### **Modern (Deep Learning-Based) Techniques**

These methods leverage deep neural networks to learn relevant features automatically from vast amounts of data, offering superior robustness and accuracy.

* **Convolutional Neural Networks (CNNs):** CNNs are the current state-of-the-art for most image recognition tasks. A CNN can be trained on large, labeled datasets of faces (such as CK+, FER2013, or AffectNet) to learn a hierarchical set of features—from simple edges in early layers to complex object parts in deeper layers—that are discriminative for smiling versus non-smiling expressions.28  
* **Facial Landmark Detection:** Modern deep learning models can locate a dense set of facial landmarks (e.g., 68 points or more) with high precision in real-time. The rich geometric information derived from these landmarks—such as distances, angles, and curvatures—provides a powerful and quantitative descriptor for any facial expression.  
* **Hybrid and Advanced Architectures:** For analyzing expressions in video, hybrid models are particularly effective. A CNN can be used to extract spatial features from each frame, and a Recurrent Neural Network (RNN), such as a Long Short-Term Memory (LSTM) network, can then model the temporal evolution of these features.28 This allows the system to understand the dynamics of a smile as it forms, holds, and fades. Furthermore, Graph Neural Networks (GNNs) can explicitly model the face as a deformable graph of landmarks, capturing the complex, non-rigid structural changes that occur during an expression.28

#### **Recommended Strategy**

For a real-time, robust, and developer-friendly implementation, a hybrid approach is optimal:

1. **Leverage a Pre-trained Model:** Utilize a high-performance, pre-trained facial landmark detector. Libraries like Google's MediaPipe or Dlib offer lightweight yet powerful models that can provide a real-time stream of facial keypoints with minimal setup.  
2. **Derive Geometric Features:** From the raw keypoint coordinates, calculate a set of meaningful geometric features. Key metrics for a smile include: the curvature of the upper and lower lip lines, the mouth width-to-height ratio, the vertical distance between the mouth corners and eye corners, and the aperture of the eyes.  
3. **Analyze Dynamics:** The true nature of an expression lies in its dynamics. The system should calculate the first derivative (velocity) of these geometric features over time. This captures the speed and direction of muscle movements, allowing the system to distinguish between a static grin and a genuinely unfolding smile.

### **2.2 Mapping Visual Primitives to Auditory Cues**

Once the visual features are extracted, they must be translated into a sonic language. Instead of a continuous, monolithic sound, individual features should trigger discrete or evolving sound events—"auditory gestures." This approach, which draws from the concepts of **model-based sonification**, **earcons** (abstract, symbolic sounds), and **auditory icons** (sounds that mimic their real-world source), creates a compositional and learnable system.31 The goal is to build a lexicon where each sound intuitively reflects the character of the visual feature it represents.

The following table provides a foundational lexicon for this visual-to-auditory mapping. This structured dictionary is essential for building a comprehensible and consistent system, translating the output of the computer vision pipeline directly into specifications for the audio synthesis engine.

| Visual Feature | CV Detection Method | Auditory Representation (Earcon/Gesture) | Rationale |
| :---- | :---- | :---- | :---- |
| **Sharp, Curved Line (e.g., upper lip of a smile)** | Landmark analysis (curvature), Edge Detection | A rapid, upward pitch sweep (glissando) with a bright, clear timbre. | The upward trajectory of the sound directly mirrors the upward curve of the smile. The speed of the sweep and the brightness of the timbre convey sharpness and positive affect. |
| **Soft, Diffuse Edge (e.g., shadow under the chin)** | Texture analysis (low gradient magnitude) | A low-pass filtered, low-volume sound with a slow attack and long release. A soft "whoosh" or "pad" sound. | The auditory "blurriness" created by the low-pass filter and slow envelope provides a direct cross-modal mapping to the visual softness of the shadow. |
| **Smooth, Uniform Texture (e.g., forehead, cheek)** | Texture analysis (low variance, LBP) | A stable, consonant, low-complexity drone (e.g., a pure sine tone or a simple, sustained chord). | The auditory stability and timbral purity reflect the visual uniformity and lack of fine detail, serving as a neutral background texture. |
| **Rough, Complex Texture (e.g., beard stubble, wrinkled skin)** | Texture analysis (high variance, Gabor filters) | A granular, noisy, or rapidly modulating sound (e.g., filtered white noise, granular synthesis, or a tremolo effect). | The auditory "roughness" and rapid fluctuation provide an intuitive sonic analogue to the complex visual texture. |
| **Sharp Corner (e.g., corner of the mouth)** | Landmark detection, Harris corner detection | A short, percussive, high-frequency "click" or "pluck" sound (staccato). | The sound's sharp temporal profile (very fast attack, quick decay) and bright spectral content mimic the sharp geometric point. |
| **Direction of Motion (Optical Flow)** | Optical Flow analysis on feature points | Rhythmic panning or Doppler shift. Upward motion causes a rising pitch; leftward motion causes the sound to pan to the left channel. | This directly maps the visual kinematics of the face into the spatial and pitch domains of sound, creating a dynamic and intuitive sense of movement. |
| **"Hearing a Smile" \- A Synthesis:** | Combination of the above features. | A composed sequence of auditory gestures: two distinct staccato "plucks" for the mouth corners, followed immediately by a synchronized upward glissando for the lip line, all layered over a subtle brightening of the "cheek" texture drone. | This approach combines multiple, simpler cues into a single, complex, and recognizable auditory event. The speed and pitch range of the glissando can encode the intensity of the smile, while the simultaneous brightening of the cheek texture can represent the raising of the zygomatic major muscle. This synthesis moves from describing parts to describing the whole, achieving the project's central goal. |

---

## **Part III: An Inclusive and Adaptive Framework for Real-World Use**

### **Introduction to Part III**

For any assistive technology to be successful, it must be reliable and effective for all its intended users. This final section addresses the most critical and challenging requirement for a real-world sonification system: it must be inclusive by design. This involves confronting the well-documented problem of algorithmic bias in computer vision and proposing a concrete, two-stage technical strategy. The goal is to create a system that is robust, fair, and adaptive to both the person in the image and the environment in which they are viewed.

### **3.1 The Imperative of Inclusivity: Moving Beyond Biased Models**

The field of computer vision has a history of developing models that exhibit significant performance disparities across demographic groups. Seminal studies like "Gender Shades" have highlighted how commercial facial analysis systems perform substantially worse for individuals with darker skin tones, and particularly for women with darker skin tones.34 This bias often originates from unrepresentative training datasets and the use of measurement scales that are not fit for purpose.

A common tool used in machine learning fairness research is the **Fitzpatrick Skin-Type Scale**. However, this scale was originally developed in dermatology to classify the sun-reactivity of Caucasian skin types and was never intended to be a comprehensive measure of global skin tone diversity. Research has shown it is perceived as less inclusive and representative, especially by people from historically marginalized communities with darker skin tones.35

A more inclusive alternative is the 10-point **Monk Skin Tone (MST) Scale**, developed through a partnership between Google and sociologist Dr. Ellis Monk.34 The MST scale was specifically designed for a broader range of applications, including computer vision, and aims to capture a more continuous and representative spectrum of human skin tones. While directly classifying a person into an MST category presents its own challenges related to lighting and annotator subjectivity, the

*principle* behind the MST scale—recognizing and representing a wide, continuous spectrum—provides the correct conceptual model for building a fair system.

Sonification presents a unique opportunity to mitigate some of these visual biases. A study on the sonification of skin lesions for cancer detection demonstrated that the auditory modality could be used to highlight malignant features in a way that was independent of the patient's base skin color.37 This suggests that by translating visual information into a different sensory domain, it is possible to abstract away from surface-level features that are often the source of algorithmic bias, and instead focus on the underlying structural or relational information.

### **3.2 Technical Strategy 1: Relative Local Contrast Sonification**

A foundational principle for building an inclusive system is to focus on **relative local changes** rather than absolute color or lightness values. A smile is a universal human expression defined by a specific pattern of muscular contraction and the resulting changes in shape, shadow, and highlight on the face. These relational changes are consistent across all skin tones. A shadow on a dark-skinned face has a different absolute color value than a shadow on a light-skinned face, but in both cases, its defining characteristic is its *relative* darkness compared to the surrounding skin.

To implement this, the sonification engine must operate on a normalized space. Instead of mapping absolute RGB or HSL values, the system should first calculate a local average for a region of interest. The value to be sonified for any given point within that region would then be a normalized difference, such as sonification\_value=(Lpoint∗​−Llocal\_average∗​)/Llocal\_range∗​. This simple but powerful transformation makes the core sonification inherently robust to global variations in both skin tone and illumination, as it focuses on conveying contrast, which is the basis of feature perception.

### **3.3 Technical Strategy 2: A Two-Stage Adaptive Sonification Model**

This report's central technical proposal for achieving a truly inclusive, robust, and adaptive system is a two-stage model. This architecture is directly inspired by advanced methodologies in computational photography, specifically adaptive white balance correction using skin color modeling in the perceptually uniform CIELAB color space.38

The core challenge is that the visual appearance of a face is a product of two independent variables: the intrinsic properties of the skin (its reflectance) and the extrinsic properties of the environment (the color and intensity of the ambient light). A system based on fixed, absolute mappings will inevitably fail when either of these variables changes. The proposed two-stage model solves this by first analyzing the image to build a personalized, context-aware model of the face's appearance *in that specific moment*, and then sonifying all features relative to that dynamic baseline. This approach simultaneously accounts for the diversity of human skin tones and the variability of real-world lighting conditions.

#### **Stage 1: Global Analysis and Palette Adaptation (Calibration)**

This initial stage runs on each new frame (or periodically) to calibrate the auditory palette.

1. **Face Detection:** A robust face detector, such as those available in OpenCV or MediaPipe, is used to identify the bounding box of the primary face in the visual field.39  
2. **Skin Segmentation:** Within the facial bounding box, the system performs skin pixel segmentation. The adaptive threshold ellipsoid method described in recent research is an ideal candidate for this task.38 This process involves converting the pixels in the facial region to the CIELAB color space and then iteratively calculating the center and spread of the dominant color cluster. This yields a statistical model—an ellipsoid in 3D color space—that represents the specific "base palette" of that individual's skin under the current, uncorrected lighting conditions.  
3. **Palette Calibration:** This calculated skin palette becomes the anchor for the entire sonification map. The center of the color ellipsoid (a specific mean L∗, a∗, b∗ value) is designated as the **auditory baseline**, mapping to the neutral, middle-ground sound of the chosen sonification model. The statistical spread of colors within the ellipsoid (e.g., the standard deviation along the lightness, red-green, and yellow-blue axes) defines the dynamic range of the sonification. For example, the mean lightness Lmean∗​ will map to a central pitch, while the observed range from Lmin∗​ to Lmax∗​ will be mapped across the full available pitch range of the audio engine.

#### **Stage 2: Relative Feature Sonification (Execution)**

Once the system is calibrated, it can proceed with the detailed sonification.

1. **Normalization:** For any pixel or feature to be sonified, its absolute color value is not used. Instead, its *vector difference* from the calibrated baseline is calculated. A highlight on the cheek is sonified based on how much its L∗a∗b∗ value deviates from the mean skin L∗a∗b∗. A shadow is sonified based on its deviation in the opposite direction. This ensures that the sonification represents features as they are perceived: as modulations of a base skin tone.  
2. **Mapping:** These normalized, relative values are then fed as input into one of the sonification models described in Part I (e.g., Naturalistic, Symbolic, or Emotional).  
3. **Feature Integration:** In parallel, the computer vision pipeline for detecting form and expression (from Part II) identifies geometric features. The auditory gestures it triggers (from Table 2\) are then layered on top of the adaptive textural sonification. The glissando of a smile will thus be a distinct, dynamic event that stands out clearly against the stable, calibrated drone representing the person's unique skin tone.

This adaptive framework makes the system fundamentally more robust and equitable. It does not need to be exhaustively pre-trained on every conceivable skin tone under every possible lighting condition because it learns the specific palette "on the fly" from the person it is observing. It intelligently separates the stable variable (the person's face) from the dynamic variable (their expression), focusing its communicative power on sonifying the expressions and features that are common to all of humanity, rather than the superficial color that varies. This is the key to creating an assistive technology that is truly universal.

#### **Works cited**

1. Data Sonification for Beginners | Music Library Emerging Technologies and Services, accessed July 24, 2025, [https://mlaetsc.hcommons.org/2023/01/18/data-sonification-for-beginners/](https://mlaetsc.hcommons.org/2023/01/18/data-sonification-for-beginners/)  
2. From Data to Melody: Data Sonification and Its Role in Open Science | NASA Earthdata, accessed July 24, 2025, [https://www.earthdata.nasa.gov/news/blog/from-data-melody-data-sonification-its-role-open-science](https://www.earthdata.nasa.gov/news/blog/from-data-melody-data-sonification-its-role-open-science)  
3. Stumbling Upon Data Sonification When I Fused My Passion for Music with Coding | D-Lab, accessed July 24, 2025, [https://dlab.berkeley.edu/news/stumbling-upon-data-sonification-when-i-fused-my-passion-music-coding](https://dlab.berkeley.edu/news/stumbling-upon-data-sonification-when-i-fused-my-passion-music-coding)  
4. Colorblind-Friendly Color Maps for R • viridis, accessed July 24, 2025, [https://sjmgarnier.github.io/viridis/](https://sjmgarnier.github.io/viridis/)  
5. Introduction to the viridis color maps, accessed July 24, 2025, [https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html](https://cran.r-project.org/web/packages/viridis/vignettes/intro-to-viridis.html)  
6. Choosing Colormaps in Matplotlib — Matplotlib 3.10.3 documentation, accessed July 24, 2025, [https://matplotlib.org/stable/users/explain/colors/colormaps.html](https://matplotlib.org/stable/users/explain/colors/colormaps.html)  
7. Color in a Perceptual Uniform Way | Nightingale, accessed July 24, 2025, [https://nightingaledvs.com/color-in-a-perceptual-uniform-way/](https://nightingaledvs.com/color-in-a-perceptual-uniform-way/)  
8. CET Perceptually Uniform Colour Maps, accessed July 24, 2025, [https://colorcet.com/](https://colorcet.com/)  
9. matplotlib colormaps \- GitHub Pages, accessed July 24, 2025, [https://bids.github.io/colormap/](https://bids.github.io/colormap/)  
10. ViridisColor | Wolfram Function Repository, accessed July 24, 2025, [https://resources.wolframcloud.com/FunctionRepository/resources/c337bfe2-97b2-4140-ba35-a2a0afb72b8e/](https://resources.wolframcloud.com/FunctionRepository/resources/c337bfe2-97b2-4140-ba35-a2a0afb72b8e/)  
11. Perceptually uniform colormaps \- File Exchange \- MATLAB Central \- MathWorks, accessed July 24, 2025, [https://www.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps](https://www.mathworks.com/matlabcentral/fileexchange/51986-perceptually-uniform-colormaps)  
12. DeepSqueak/Functions/Colormaps/magma.m at master \- GitHub, accessed July 24, 2025, [https://github.com/DrCoffey/DeepSqueak/blob/master/Functions/Colormaps/magma.m](https://github.com/DrCoffey/DeepSqueak/blob/master/Functions/Colormaps/magma.m)  
13. Collection of perceptually accurate colormaps — colorcet v3.1.0, accessed July 24, 2025, [https://colorcet.holoviz.org/](https://colorcet.holoviz.org/)  
14. Psychoacoustics 101: How To Manipulate Emotions With Sound \- Unison Audio, accessed July 24, 2025, [https://unison.audio/psychoacoustics/](https://unison.audio/psychoacoustics/)  
15. (PDF) Investigating Colour-Sound Mapping in Children and Adults: A Pilot Study, accessed July 24, 2025, [https://www.researchgate.net/publication/374735000\_Investigating\_Colour-Sound\_Mapping\_in\_Children\_and\_Adults\_A\_Pilot\_Study](https://www.researchgate.net/publication/374735000_Investigating_Colour-Sound_Mapping_in_Children_and_Adults_A_Pilot_Study)  
16. Hear The Rainbow: Interactive & Emotive Sonification of 3D Color Space, accessed July 24, 2025, [https://munsell.com/color-blog/hear-the-rainbow-color-sound/](https://munsell.com/color-blog/hear-the-rainbow-color-sound/)  
17. The Bridge Between Psychoacoustics and Music Emotion \- MasteringBOX, accessed July 24, 2025, [https://www.masteringbox.com/learn/psychoacoustics-for-music-production](https://www.masteringbox.com/learn/psychoacoustics-for-music-production)  
18. Psychoacoustics: Decoding Sound's Impact on Mind and Health \- Schallertech, accessed July 24, 2025, [https://schallertech.com/en/psychoacoustics-the-effect-of-sound-on-human-perception/](https://schallertech.com/en/psychoacoustics-the-effect-of-sound-on-human-perception/)  
19. Understanding Psychoacoustics: Definition and Applications \- Ansys, accessed July 24, 2025, [https://www.ansys.com/blog/understanding-psychoacoustics](https://www.ansys.com/blog/understanding-psychoacoustics)  
20. Timbre \- Emotional Expression and Emotional Effects \- How Music REALLY Works, accessed July 24, 2025, [https://www.howmusicreallyworks.com/chapter-three-tones-overtones/timbre-emotional-expression.html](https://www.howmusicreallyworks.com/chapter-three-tones-overtones/timbre-emotional-expression.html)  
21. Perception and Modeling of Affective Qualities of Musical ... \- Frontiers, accessed July 24, 2025, [https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00153/full](https://www.frontiersin.org/journals/psychology/articles/10.3389/fpsyg.2017.00153/full)  
22. Understanding Psychoacoustics in Sound \- Number Analytics, accessed July 24, 2025, [https://www.numberanalytics.com/blog/understanding-psychoacoustics-in-sound](https://www.numberanalytics.com/blog/understanding-psychoacoustics-in-sound)  
23. Unlocking Psychoacoustics in Music \- Number Analytics, accessed July 24, 2025, [https://www.numberanalytics.com/blog/psychoacoustics-in-music](https://www.numberanalytics.com/blog/psychoacoustics-in-music)  
24. Designing Emotional and Intuitive Sounds for Tech: Insights From Psychoacoustics, accessed July 24, 2025, [https://www.researchgate.net/publication/388438697\_Designing\_Emotional\_and\_Intuitive\_Sounds\_for\_Tech\_Insights\_From\_Psychoacoustics](https://www.researchgate.net/publication/388438697_Designing_Emotional_and_Intuitive_Sounds_for_Tech_Insights_From_Psychoacoustics)  
25. Effects of Sound Interventions on the Mental Stress Response in Adults: Scoping Review, accessed July 24, 2025, [https://pmc.ncbi.nlm.nih.gov/articles/PMC11976171/](https://pmc.ncbi.nlm.nih.gov/articles/PMC11976171/)  
26. The importance of timbre in generating an emotional reaction \- Instrumentality, accessed July 24, 2025, [https://instrumentality.me/magic/how-tasty-is-your-timbre/](https://instrumentality.me/magic/how-tasty-is-your-timbre/)  
27. Detecting human facial expression by common computer vision techniques, accessed July 24, 2025, [https://www.interactivearchitecture.org/detecting-human-facial-expression-by-common-computer-vision-techniques.html](https://www.interactivearchitecture.org/detecting-human-facial-expression-by-common-computer-vision-techniques.html)  
28. (PDF) Computer Vision of Smile Detection Based on Machine and ..., accessed July 24, 2025, [https://www.researchgate.net/publication/389922188\_Computer\_Vision\_of\_Smile\_Detection\_Based\_on\_Machine\_and\_Deep\_Learning\_Approach](https://www.researchgate.net/publication/389922188_Computer_Vision_of_Smile_Detection_Based_on_Machine_and_Deep_Learning_Approach)  
29. Sonification of Face | Patil | Automation and Autonomous System, accessed July 24, 2025, [https://www.ciitresearch.org/dl/index.php/aa/article/view/AA072012028/0](https://www.ciitresearch.org/dl/index.php/aa/article/view/AA072012028/0)  
30. Computer Vision of Smile Detection Based on Machine and Deep Learning Approach \- ASPG, accessed July 24, 2025, [https://www.americaspg.com/article/pdf/3731](https://www.americaspg.com/article/pdf/3731)  
31. A Proposal for the Interactive Sonification of the Human Face \- Luca Andrea Ludovico, accessed July 24, 2025, [https://ludovico.lim.di.unimi.it/download/papers/CHIRA2018\_presti.pdf](https://ludovico.lim.di.unimi.it/download/papers/CHIRA2018_presti.pdf)  
32. A Proposal for the Interactive Sonification of the Human Face \- SciTePress, accessed July 24, 2025, [https://www.scitepress.org/PublishedPapers/2018/69570/pdf/index.html](https://www.scitepress.org/PublishedPapers/2018/69570/pdf/index.html)  
33. Sonification of Facial Actions for Musical Expression, accessed July 24, 2025, [https://www.nime.org/proceedings/2005/nime2005\_127.pdf](https://www.nime.org/proceedings/2005/nime2005_127.pdf)  
34. Consensus and subjectivity of skin tone annotation for ML fairness \- Google Research, accessed July 24, 2025, [https://research.google/blog/consensus-and-subjectivity-of-skin-tone-annotation-for-ml-fairness/](https://research.google/blog/consensus-and-subjectivity-of-skin-tone-annotation-for-ml-fairness/)  
35. Which Skin Tone Measures are the Most Inclusive? An Investigation of Skin Tone Measures for Artificial Intelligence. \- Harvard University, accessed July 24, 2025, [https://scholar.harvard.edu/sites/scholar.harvard.edu/files/monk/files/heldreth\_et\_al.\_which\_skin\_tone\_measures\_are\_the\_most\_inclusive\_vpreprint.pdf](https://scholar.harvard.edu/sites/scholar.harvard.edu/files/monk/files/heldreth_et_al._which_skin_tone_measures_are_the_most_inclusive_vpreprint.pdf)  
36. Building more skin tone inclusive computer vision models \- YouTube, accessed July 24, 2025, [https://www.youtube.com/watch?v=vuv\_r3iGM14](https://www.youtube.com/watch?v=vuv_r3iGM14)  
37. Skin Cancer Detection in Diverse Skin Tones by Machine Learning ..., accessed July 24, 2025, [https://karger.com/ocl/article/doi/10.1159/000541573/913627/Skin-Cancer-Detection-in-Diverse-Skin-Tones-by](https://karger.com/ocl/article/doi/10.1159/000541573/913627/Skin-Cancer-Detection-in-Diverse-Skin-Tones-by)  
38. Improve Image White Balance by Facial Skin Color \- Society for ..., accessed July 24, 2025, [https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/31/1/9](https://library.imaging.org/admin/apis/public/api/ist/website/downloadArticle/cic/31/1/9)  
39. Face Detection with Python Using OpenCV Tutorial | DataCamp, accessed July 24, 2025, [https://www.datacamp.com/tutorial/face-detection-python-opencv](https://www.datacamp.com/tutorial/face-detection-python-opencv)