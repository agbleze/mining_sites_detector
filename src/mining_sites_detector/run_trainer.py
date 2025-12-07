from argparse import ArgumentParser
import os
from mining_sites_detector.model_trainer import trigger_training_process, get_model
from mining_sites_detector.data_preprocessor import get_data, get_tiff_img
from mining_sites_detector.export_utils import export_onnx_model
import json
import torch


def parse_args():
    parser = ArgumentParser(description="Run Model Trainer")
    parser.add_argument('--train_img_dir', type=str, required=True, help='Path to training images directory')
    parser.add_argument('--val_img_dir', type=str, required=True, help='Path to validation images directory')
    parser.add_argument('--train_target_file', type=str, required=True, help='Path to training target file')
    parser.add_argument('--val_target_file', type=str, required=True, help='Path to validation target file')
    parser.add_argument('--num_epochs', type=int, default=10, help='Number of training epochs')
    parser.add_argument("--save_train_results_as", type=str, default="train_results.json", help="File path to save training results json")
    parser.add_argument('--model_store_dir', type=str, default="model_store", help='Directory to store trained models')
    parser.add_argument('--model_name', type=str, default="mining_site_detector_model", help='Base name for the saved model files')
    parser.add_argument('--normalize_bands', action='store_true', help='Whether to normalize image bands')
    parser.add_argument('--checkpoint_interval', type=int, default=1, help='Interval (in epochs) to save model checkpoints')
    parser.add_argument('--export_to_onnx', action='store_true', help='Whether to export the trained model to ONNX format after training')
    return parser.parse_args()


def main():
    args = parse_args()
    train_img_dir = args.train_img_dir
    val_img_dir = args.val_img_dir
    train_target_file = args.train_target_file
    val_target_file = args.val_target_file
    num_epochs = args.num_epochs
    save_train_results_as = args.save_train_results_as
    model_store_dir = args.model_store_dir
    model_name = args.model_name
    checkpoint_interval = args.checkpoint_interval
    export_to_onnx = args.export_to_onnx
    
    train_dl, val_dl = get_data(train_img_dir=train_img_dir, 
                            val_image_dir=val_img_dir,
                            train_target_file_path=train_target_file,
                            val_target_file_path=val_target_file,
                            target_file_has_header=True, 
                            loader=get_tiff_img,
                            return_all_bands=True, batch_size=10
                            )

    model, loss_fn, optimizer = get_model()

    result = trigger_training_process(train_dataload=train_dl, val_dataload=val_dl,
                                    model=model, loss_fn=loss_fn,
                                    optimizer=optimizer, 
                                    num_epochs=num_epochs,
                                    model_store_dir=model_store_dir, 
                                    model_name=model_name,
                                    checkpoint_interval=checkpoint_interval
                                    )
    
    
    with open(save_train_results_as, "w") as f:
        json.dump(result, f)
        
    if export_to_onnx:
        final_model_path = os.path.join(model_store_dir, f'{model_name}_epoch_{num_epochs}.pth')
        model.load_state_dict(torch.load(final_model_path))
        sample_img, _ = next(iter(train_dl))
        sample_image = sample_img[0]
        onnx_save_path = os.path.join(model_store_dir, f"{model_name}_epoch_{num_epochs}.onnx")
        export_onnx_model(model=model, image=sample_image, save_onnx_as=onnx_save_path)
        print(f"Exported ONNX model to: {onnx_save_path}")
        
    

if __name__ == "__main__":
    main()   