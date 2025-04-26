import os
import argparse
import logging
from pathlib import Path
from typing import List, Tuple

from detect_and_annotate import PoseProcessor
from visualize_skeleton_from_detailed_csv import visualize as visualize_skeleton

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('pipeline.log')
    ]
)
logger = logging.getLogger(__name__)

def get_project_root() -> Path:
    # ⬇⬇⬇ Переход на уровень выше src/
    return Path(__file__).parent.parent

def validate_dataset_structure(dataset_path: Path) -> bool:
    required_folders = {'standing', 'sitting', 'lying'}
    if not dataset_path.exists():
        logger.error(f"Dataset directory not found: {dataset_path}")
        return False

    present_folders = {f.name for f in dataset_path.iterdir() if f.is_dir()}
    missing = required_folders - present_folders
    if missing:
        logger.error(f"Missing required folders: {missing}")
        return False

    return True

def process_dataset(dataset_root: Path = None) -> List[Tuple[str, str]]:
    project_root = get_project_root()
    dataset_root = dataset_root or project_root / 'data' / 'dataset'

    if not validate_dataset_structure(dataset_root):
        raise ValueError("Invalid dataset structure")

    output_dir = project_root / 'data' / 'annotations'
    output_dir.mkdir(parents=True, exist_ok=True)

    processor = PoseProcessor(yolo_model_path=str(project_root / 'yolo11n.pt'))
    csv_paths = []

    for class_name in ['standing', 'sitting', 'lying']:
        class_dir = dataset_root / class_name
        output_csv = output_dir / f'{class_name}_keypoints.csv'

        logger.info(f"Processing {class_name} images...")
        try:
            processor.process_class_directory(str(class_dir), str(output_csv))
            csv_paths.append((class_name, str(output_csv)))
            logger.info(f"Saved to {output_csv}")
        except Exception as e:
            logger.error(f"Failed to process {class_name}: {str(e)}")

    if not csv_paths:
        raise RuntimeError("No CSV files were created")

    return csv_paths

def visualize_annotations(csv_paths: List[Tuple[str, str]], clean_canvas: bool = False, live: bool = False) -> None:
    project_root = get_project_root()
    vis_output_dir = project_root / 'data' / 'visualizations'

    if live:
        logger.info(f"Starting live visualization...")
        visualize_skeleton(csv_path=None, output_dir=None, live=True)
        return

    for class_name, csv_path in csv_paths:
        output_dir = vis_output_dir / class_name
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Visualizing {class_name} skeletons... (clean={clean_canvas})")
        try:
            visualize_skeleton(
                csv_path=csv_path,
                output_dir=str(output_dir),
                clean_canvas=clean_canvas,
                live=False
            )
            logger.info(f"Visualizations saved to {output_dir}")
        except Exception as e:
            logger.error(f"Failed to visualize {class_name}: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Pose Detection and Visualization Pipeline')
    parser.add_argument('--dataset', type=str, default=None, help='Path to dataset')
    parser.add_argument('--train', action='store_true', help='Train model after processing')
    parser.add_argument('--visualize', action='store_true', help='Visualize skeletons')
    parser.add_argument('--clean_visuals', action='store_true', help='Draw on white background')
    parser.add_argument('--live', action='store_true', help='Use webcam for live visualization')
    parser.add_argument('--log_level', default='INFO', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'], help='Logging level')
    args = parser.parse_args()

    logger.setLevel(args.log_level)

    try:
        logger.info("Starting pipeline...")
        dataset_path = Path(args.dataset) if args.dataset else None
        csv_files = process_dataset(dataset_path)

        if args.visualize or args.live:
            visualize_annotations(csv_files, clean_canvas=args.clean_visuals, live=args.live)

        if args.train:
            from cnn_model import train_pose_classifier
            model_path = get_project_root() / 'models' / 'pose_classifier.pth'
            model_path.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Starting model training...")
            train_pose_classifier(
                csv_paths=[path for _, path in csv_files],
                model_save_path=str(model_path)
            )
            logger.info(f"Model saved to {model_path}")

        logger.info("Pipeline completed successfully")
    except Exception as e:
        logger.error(f"Pipeline failed: {str(e)}")
        raise

if __name__ == "__main__":
    main()
