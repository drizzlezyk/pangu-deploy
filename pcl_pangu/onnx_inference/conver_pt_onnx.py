import numpy as np
import random
from .pangu.config import PanguConfig
import os


def convert_2_onnx(input_model_path, model, past):
    import torch
    batch_size= 1
    context_tokens = [1,2,3,4]
    past = torch.tensor(past)
    dummy_input = (torch.tensor(context_tokens, device="cpu", dtype=torch.int).unsqueeze(0).repeat(batch_size, 1),
                   None,None,None,past)
    # model_script = torch.jit.script(model)
    input_names = ["input_0","past"]
    output_names = ["output_0","output_1"]
    torch.onnx.export(model, dummy_input, input_model_path,input_names=input_names,
                      opset_version=11,
                      output_names=output_names,
                      use_external_data_format=True,
                      dynamic_axes={'input_0':[0,1],'past':[0,1,4],'output_0':[0,1,2]})
    print('@@@@ onnx model saved to: ', input_model_path)

def quantize_onnx(input_model_path: str, output_model_path: str):
    from onnxruntime.quantization import quantize_dynamic, QuantType

    quantized_model = quantize_dynamic(
        input_model_path,
        output_model_path,
        weight_type=QuantType.QUInt8,
        use_external_data_format=True,
        extra_options={
            'DisableShapeInference': True,
        }
    )
    print('@@@@ onnx quantized model saved to: ', output_model_path)


def pt_2_onnx8(model_name, pt_path, model_config, onnx_ckpt_root_dir):
    import torch
    from .pangu.model import Pangu
    from .pangu.utils import load_weight
    state_dict = torch.load(pt_path, map_location='cpu')

    device = "cpu"
    print('@@@@ model_name: ', model_name)

    path_int8 = f'onnx_int8_{model_name}'
    path_int8 = os.path.join(onnx_ckpt_root_dir, path_int8)
    assert not os.path.exists(path_int8), f'\n@@@@ onnx_int8_model_path exists: {path_int8}' \
                                          f'\n@@@@ The quantification process did not run, please remove it and run again!'

    config = PanguConfig(model_config)
    model = Pangu(config)
    model = load_weight(model, state_dict)
    model.to(device)
    model.eval()

    layer_past_path = f'{model_name}_layer_past.npy'
    layer_past_path = os.path.join(path_int8, layer_past_path)
    input_ids = torch.tensor([13], device=device, dtype=torch.long).unsqueeze(0).repeat(1, 1)
    logits, past = model(input_ids, past=None, position_ids=None, lm_labels=None, token_type_ids=None)
    np.save(layer_past_path, past.detach().cpu().numpy())

    path_onnx = f'onnx_{model_name}'
    path_onnx = os.path.join(path_int8, path_onnx)
    onnx_model_path = f'{path_onnx}/{model_name}.onnx'
    os.system(f'rm -rf {path_onnx}')
    os.system(f'mkdir -p {path_onnx}')
    convert_2_onnx(onnx_model_path, model, past)

    onnx_int8_model_path = f'{path_int8}/{model_name}_int8.onnx'
    os.system(f'rm -rf {path_int8}')
    os.system(f'mkdir -p {path_int8}')
    quantize_onnx(onnx_model_path, onnx_int8_model_path)

    os.system(f'rm -rf {path_onnx}')