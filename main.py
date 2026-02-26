"""
Heart Disease Prediction - Main Pipeline

Run the complete pipeline or individual steps:
    python main.py --train        # Run training pipeline
    python main.py --predict      # Run interactive prediction
    python main.py --batch        # Run batch prediction

Based on the heart_disease_prediction.ipynb implementation.
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from backend.training.predict import interactive_prediction, predict_batch
from backend.training.preprocessing import preprocess_data
from backend.training.train import train_models
from config import MODEL_FILE, SCALER_FILE


def print_banner():
    """Print project banner."""
    print("\n" + "=" * 60)
    print("  HEART DISEASE PREDICTION PIPELINE")
    print("  CatBoost Model for Medical Diagnosis")
    print("=" * 60)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Heart Disease Prediction Pipeline',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py --train                Train model from scratch
  python main.py --predict              Run interactive prediction
  python main.py --batch input.csv      Run batch prediction
  python main.py --batch input.csv output.csv
        """
    )
    
    parser.add_argument('--train', action='store_true',
                        help='Train model from scratch (preprocessing + training)')
    parser.add_argument('--predict', action='store_true', 
                        help='Run interactive prediction')
    parser.add_argument('--batch', nargs='+',
                        help='Run batch prediction: --batch input.csv [output.csv]')
    
    args = parser.parse_args()
    
    # If no arguments provided, show help
    if not any(vars(args).values()):
        print_banner()
        print("\nNo arguments provided. Running interactive prediction...")
        print("Use --help for available options.\n")
        interactive_prediction()
        return
    
    print_banner()
    
    if args.train:
        print("\n🔄 Starting training pipeline...")
        print("\nStep 1/2: Preprocessing data...")
        X_train, X_test, y_train, y_test = preprocess_data()
        
        print("\nStep 2/2: Training model with best parameters...")
        best_model, best_scaler, best_score = train_models()
        
        print(f"\n✅ Training complete! ROC-AUC: {best_score:.4f}")
        print(f"Model saved to: models/{MODEL_FILE}")
        print(f"Scaler saved to: models/{SCALER_FILE}")
    
    if args.predict:
        interactive_prediction()
    
    if args.batch:
        if len(args.batch) < 1:
            print("Error: --batch requires at least input CSV file")
            sys.exit(1)
        
        input_file = args.batch[0]
        output_file = args.batch[1] if len(args.batch) > 1 else "predictions.csv"
        
        print(f"\nRunning batch prediction...")
        print(f"Input: {input_file}")
        print(f"Output: {output_file}\n")
        
        predict_batch(input_file, output_file)


if __name__ == "__main__":
    main()
