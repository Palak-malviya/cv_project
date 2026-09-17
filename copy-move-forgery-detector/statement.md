# Problem Statement: Copy-Move Image Forgery

## The Problem
With the proliferation of digital media, image manipulation has become incredibly accessible. A common manipulation technique is the **copy-move forgery**, where a section of an image is copied and pasted into another area of the *same* image. This is often done to hide an object or duplicate elements (e.g., in journalism, legal evidence, or social media). 

Detecting these forgeries manually can be extremely difficult, especially if the pasted region has been slightly rotated, scaled, or blended.

## Scope
This project focuses **exclusively on single-image copy-move forgeries**. It does not address:
- Splicing (inserting elements from different images).
- Retouching (blurring, airbrushing).
- AI-generated images (Deepfakes/GANs).

By narrowing the scope, we can rely entirely on classical Computer Vision (CV) techniques without the need for massive training datasets or expensive GPU compute required by deep learning approaches.

## Target Audience
- **Digital Forensics Students & Researchers**: To understand the fundamentals of local feature matching and geometric verification.
- **Content Moderators**: As a fast, lightweight first-pass filter to flag suspicious images.
- **Journalists**: To quickly verify the authenticity of submitted visual evidence.

## High-Level Features
- **Deterministic Pipeline**: Relies on mathematical models (SIFT, RANSAC) ensuring explainable results.
- **No Training Data Required**: Works out-of-the-box on any image.
- **Explainable Metrics**: Outputs a structured report with confidence scores and intermediate visual steps.
