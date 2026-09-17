import gradio as gr
import cv2
import numpy as np
import preprocessing
from detection.keypoint_matcher import extract_sift_features, self_match
from detection.geometric_filter import ransac_filter, cluster_matches
from detection.region_localizer import localize_regions, compute_confidence_score
import visualizer

def detect_forgery(image):
    if image is None:
        return None, "Please upload an image."
        
    try:
        # Image is already a numpy array (RGB from Gradio)
        # Convert to BGR for our pipeline since OpenCV expects BGR internally
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Preprocessing
        resized = preprocessing.resize_if_large(image_bgr)
        gray = preprocessing.to_grayscale(resized)
        
        # Keypoint Matching
        keypoints, descriptors = extract_sift_features(gray)
        raw_matches = self_match(keypoints, descriptors)
        
        # Geometric Filtering
        inlier_matches, matrix = ransac_filter(raw_matches, threshold=5.0)
        
        clusters = []
        if len(inlier_matches) >= 4:
            clusters = cluster_matches(inlier_matches, eps=40.0)
            
        # Region Localization
        regions = localize_regions(clusters)
        
        # Compute metrics
        image_area = resized.shape[0] * resized.shape[1]
        total_region_area = sum(r['area'] for r in regions)
        confidence = compute_confidence_score(len(inlier_matches), total_region_area, image_area)
        
        # Visualization
        final_img = visualizer.draw_forged_regions(resized, regions)
        
        # Convert back to RGB for Gradio
        final_img_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
        
        report = f"Raw Matches: {len(raw_matches)}\n" \
                 f"Inlier Matches: {len(inlier_matches)}\n" \
                 f"Suspected Regions: {len(regions)}\n" \
                 f"Confidence Score: {confidence:.2f} (0=Authentic, 1=Forged)"
                 
        return final_img_rgb, report
        
    except Exception as e:
        return None, f"Error processing image: {str(e)}"

# Create Gradio interface
iface = gr.Interface(
    fn=detect_forgery,
    inputs=gr.Image(type="numpy", label="Upload Image"),
    outputs=[
        gr.Image(type="numpy", label="Detection Result"),
        gr.Textbox(label="Metrics Report")
    ],
    title="Copy-Move Forgery Detection",
    description="Upload an image to detect if any regions have been copied and pasted within it (copy-move forgery).",
    allow_flagging="never"
)

if __name__ == "__main__":
    iface.launch()
