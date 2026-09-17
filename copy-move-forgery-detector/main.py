import argparse
import sys
import os
import time
from utils import setup_logger, InvalidImageError, NoKeypointsFoundError
import preprocessing
from detection.keypoint_matcher import extract_sift_features, self_match
from detection.geometric_filter import ransac_filter, cluster_matches
from detection.region_localizer import localize_regions, compute_confidence_score
import visualizer

def main():
    parser = argparse.ArgumentParser(description="Copy-Move Image Forgery Detection")
    parser.add_argument("--input", required=True, help="Path to input image")
    parser.add_argument("--output", default="output/", help="Output directory")
    parser.add_argument("--min-matches", type=int, default=4, help="Minimum matches for RANSAC")
    parser.add_argument("--ransac-threshold", type=float, default=5.0, help="RANSAC reprojection threshold")
    parser.add_argument("--dbscan-eps", type=float, default=40.0, help="DBSCAN epsilon for clustering")
    parser.add_argument("--verbose", action="store_true", help="Enable verbose logging")
    
    args = parser.parse_args()
    
    # Setup logger
    logger = setup_logger('main')
    if args.verbose:
        logger.setLevel("DEBUG")
        
    start_time = time.time()
    
    try:
        # 1. Preprocessing
        logger.info(f"Processing image: {args.input}")
        original_image = preprocessing.load_image(args.input)
        image = preprocessing.resize_if_large(original_image)
        gray_image = preprocessing.to_grayscale(image)
        
        # 2. Keypoint Matching
        keypoints, descriptors = extract_sift_features(gray_image)
        raw_matches = self_match(keypoints, descriptors)
        
        # 3. Geometric Filtering
        inlier_matches, transform_matrix = ransac_filter(raw_matches, threshold=args.ransac_threshold)
        
        clusters = []
        if len(inlier_matches) >= args.min_matches:
            clusters = cluster_matches(inlier_matches, eps=args.dbscan_eps)
            
        # 4. Region Localization
        regions = localize_regions(clusters)
        
        # Compute metrics
        image_area = image.shape[0] * image.shape[1]
        total_region_area = sum(r['area'] for r in regions)
        confidence = compute_confidence_score(len(inlier_matches), total_region_area, image_area)
        
        # 5. Visualization & Reporting
        keypoints_img = visualizer.draw_keypoints(image, keypoints)
        matches_img = visualizer.draw_matches(image, inlier_matches)
        final_img = visualizer.draw_forged_regions(image, regions)
        
        out_base = os.path.basename(args.input).rsplit('.', 1)[0]
        report_path = os.path.join(args.output, f"{out_base}_report.json")
        fig_path = os.path.join(args.output, f"{out_base}_summary.png")
        
        results = {
            'input_file': args.input,
            'keypoints_extracted': len(keypoints),
            'raw_matches': len(raw_matches),
            'inlier_matches': len(inlier_matches),
            'regions_detected': len(regions),
            'confidence_score': confidence,
            'processing_time_seconds': time.time() - start_time
        }
        
        visualizer.save_report(results, report_path)
        visualizer.create_summary_figure(image, keypoints_img, matches_img, final_img, fig_path)
        
        # Console Summary
        print("\n" + "="*40)
        print("FORGERY DETECTION SUMMARY")
        print("="*40)
        print(f"Input Image       : {args.input}")
        print(f"Raw Matches       : {len(raw_matches)}")
        print(f"Inlier Matches    : {len(inlier_matches)}")
        print(f"Suspected Regions : {len(regions)}")
        print(f"Confidence Score  : {confidence:.2f} (0=Authentic, 1=Forged)")
        print(f"Total Time        : {results['processing_time_seconds']:.2f} s")
        print("="*40 + "\n")
        
    except (InvalidImageError, NoKeypointsFoundError) as e:
        logger.error(str(e))
        print(f"\nError: {e}\n")
        sys.exit(1)
    except Exception as e:
        logger.exception("An unexpected error occurred.")
        print(f"\nUnexpected Error: {e}\n")
        sys.exit(1)

if __name__ == "__main__":
    main()
