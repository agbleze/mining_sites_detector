import torch
from torch.onnx import TrainingMode


def export_onnx_model(model: torch.nn.Module, 
                      image,
                      save_onnx_as: str
                      ):

    with open(save_onnx_as, "wb") as f:
        torch.onnx.export(model=model,
                        args = (image,),
                        f = f, opset_version=16,
                        export_params=True,
                        input_names=["input"],
                        output_names=["boxes", "classes", "scores"],
                        training=TrainingMode.PRESERVE,
                        do_constant_folding=True,
                        dynamic_axes={
                            "input": {0: "batch_size", 1:"height", 2: "width"},
                            "classes": {0: "batch_size"},
                            "scores": {0: "batch_size"},
                            }
                        )
                
            
            
            