import copy
import traceback
import numpy as np

import torch
import torch.nn as nn

from val import ModelVal
from layers import create_attention, create_layernorm, create_layernorm2d, create_gelu, create_piecewise_softmax_layers

# TODO
# - enforce kernel sizes

def trace_module_input_shapes(model, example_input):
    input_shapes = {}
    handles = []

    def register_hook(module):
        def hook(module, input, _):
            name = getattr(module, '__module_name__', None)
            if name is not None:
                input_shapes[name] = tuple(input[0].shape)
        if len(list(module.children())) == 0:
            handle = module.register_forward_hook(hook)
            handles.append(handle)
            
    for name, module in model.named_modules():
        module.__module_name__ = name
    model.eval()
    with torch.no_grad():
        model.apply(register_hook)
        model(example_input)
    for h in handles:
        h.remove()
    return input_shapes

class ViTCompile:
    def __init__(self, device_constraints=None):
        self.constraints = {
            'max_processors': 64,
            'memory_banks': ['0x0000', '0x2000', '0x4000'],
            'supported_ops': ['Conv1d', 'Conv2d', 'Linear', 'MaxPool1d', 'MaxPool2d', 'AvgPool1d', 'AvgPool2d', 'BatchNorm2d', 'Flatten', 'Identity', 'ReLU', 'Abs'], # TODO some constraints on flatten, conv2d, etc - must add
            'max_layers': 32}
        
        if device_constraints:
            self.constraints.update(device_constraints)
        
        self.module_patterns = {
            'attention': [
                r'.*attention.*', r'.*attn.*', r'.*self\_attn.*', r'.*mha.*',
                r'.*multihead.*', r'.*multi\_head.*'
            ],
            'norm': [
                r'.*norm.*', r'.*ln.*', r'.*layer\_norm.*'
            ],
            'ffn': [
                r'.*ffn.*', r'.*feed\_forward.*', r'.*mlp.*'
            ],
            'embedding': [
                r'.*embed.*', r'.*patch\_embed.*', r'.*token\_embed.*', 
                r'.*wte.*', r'.*word\_embeddings.*'
            ],
            'pos_embedding': [
                r'.*pos.*embed.*', r'.*position.*', r'.*wpe.*', 
                r'.*positional.*', r'.*pos\_enc.*'
            ]
        }
            
        self.op_mappings = {
            # 'MultiheadAttention': self._map_attention,
            'LayerNorm': self._map_layernorm,
            # 'LayerNorm2d': self.map_layernorm2d,
            'NonDynamicallyQuantizableLinear': self._map_linear,
            # 'Dense': self._map_linear,
            # 'GELU': self._map_gelu,
            # 'Dropout': self._map_dropout,
            # 'DropPath': self._map_droppath,
            # 'AdaptiveAvgPool2d': self._map_adaptiveavgpool2d
        }
        
        self.validator = ModelVal(rtol=1e-2, atol=1e-3)

    def _fuse_linear_layers(self, layers):
        if len(layers) < 2:
            return layers[0]

        fused = layers[0]
        for next_layer in layers[1:]:
            W1, b1 = fused.weight, fused.bias
            W2, b2 = next_layer.weight, next_layer.bias

            # fused weight: W = W2 @ W1
            W_fused = W2 @ W1

            # fused bias: b = W2 @ b1 + b2
            if b1 is not None:
                b_intermediate = W2 @ b1
            else:
                b_intermediate = torch.zeros(W2.shape[0], device=W2.device, dtype=W2.dtype)

            if b2 is not None:
                b_fused = b_intermediate + b2
            else:
                b_fused = b_intermediate

            fused = nn.Linear(fused.in_features, next_layer.out_features, bias=True)
            with torch.no_grad():
                fused.weight.copy_(W_fused)
                fused.bias.copy_(b_fused)

        return fused


    def fuse_linear_chains(self, model):
        def is_linear_only_module(mod):
            if isinstance(mod, nn.Linear):
                return True
            if isinstance(mod, nn.Sequential):
                return all(isinstance(m, nn.Linear) for m in mod)
            if isinstance(mod, nn.Module):
                children = list(mod.children())
                return len(children) > 0 and all(isinstance(m, nn.Linear) for m in children)
            return False

        def get_linear_chain(mod):
            if isinstance(mod, nn.Linear):
                return [mod]
            elif isinstance(mod, nn.Sequential):
                return [m for m in mod if isinstance(m, nn.Linear)]
            elif isinstance(mod, nn.Module):
                return [m for _, m in mod.named_children() if isinstance(m, nn.Linear)]
            return []

        def replace_with_fused(parent, name, fused_layer):
            setattr(parent, name, fused_layer)

        for _, module in model.named_modules():
            for child_name, child in list(module.named_children()):
                if is_linear_only_module(child):
                    linear_chain = get_linear_chain(child)
                    if len(linear_chain) >= 2:
                        fused = linear_chain[0]
                        for next_linear in linear_chain[1:]:
                            fused = self._fuse_linear_layers([fused, next_linear])
                        replace_with_fused(module, child_name, fused)
        return model

    def _map_adaptiveavgpool2d(self, module, input_shape=None, module_name=''):
        if input_shape is not None and len(input_shape) >= 2:
            dummy_input = torch.randn(*input_shape).float()
            
            with torch.no_grad():
                dummy_output = module(dummy_input)
                target_shape = dummy_output.shape
            
            input_h, input_w = input_shape[-2:]
            output_h, output_w = target_shape[-2:]
            
            if output_h == 1 and output_w == 1:
                return nn.AvgPool2d(kernel_size=(input_h, input_w), stride=1, padding=0)
            
            stride_h = input_h // output_h if output_h > 0 else 1
            stride_w = input_w // output_w if output_w > 0 else 1
            
            kernel_h = input_h - stride_h * (output_h - 1)
            kernel_w = input_w - stride_w * (output_w - 1)
            
            kernel_h = max(1, kernel_h)
            kernel_w = max(1, kernel_w)
            stride_h = max(1, stride_h)
            stride_w = max(1, stride_w)
            
            avgpool_layer = nn.AvgPool2d(
                kernel_size=(kernel_h, kernel_w),
                stride=(stride_h, stride_w),
                padding=0)
            
            return avgpool_layer

    def _map_linear(self, module, input_shape=None, module_name=''):
        new_linear = nn.Linear(
            in_features=module.in_features,
            out_features=module.out_features,
            bias=module.bias is not None,
            dtype=module.weight.dtype,
            device=module.weight.device)
        
        with torch.no_grad():
            new_linear.weight.copy_(module.weight)
            if module.bias is not None:
                new_linear.bias.copy_(module.bias)

        new_linear.train(module.training)
        return new_linear

    def _map_attention(self, original_module, input_shape=None, module_name=''):
        embed_dim = original_module.embed_dim
        num_heads = original_module.num_heads
        
        if hasattr(original_module, 'in_proj_weight'):
            qkv_weight = original_module.in_proj_weight.clone()
            qkv_bias = original_module.in_proj_bias.clone() if original_module.in_proj_bias is not None else None
            
            q_weight = qkv_weight[:embed_dim]
            k_weight = qkv_weight[embed_dim:2*embed_dim]
            v_weight = qkv_weight[2*embed_dim:]
            
            if qkv_bias is not None:
                q_bias = qkv_bias[:embed_dim]
                k_bias = qkv_bias[embed_dim:2*embed_dim]
                v_bias = qkv_bias[2*embed_dim:]
            else:
                q_bias = k_bias = v_bias = None
        else:
            q_weight = original_module.q_proj.weight.clone()
            k_weight = original_module.k_proj.weight.clone()
            v_weight = original_module.v_proj.weight.clone()
            q_bias = original_module.q_proj.bias.clone() if hasattr(original_module.q_proj, 'bias') and original_module.q_proj.bias is not None else None
            k_bias = original_module.k_proj.bias.clone() if hasattr(original_module.k_proj, 'bias') and original_module.k_proj.bias is not None else None
            v_bias = original_module.v_proj.bias.clone() if hasattr(original_module.v_proj, 'bias') and original_module.v_proj.bias is not None else None
        
        out_weight = original_module.out_proj.weight.clone()
        out_bias = original_module.out_proj.bias.clone() if original_module.out_proj.bias is not None else None
        
        q_proj = nn.Linear(embed_dim, embed_dim, bias=q_bias is not None)
        with torch.no_grad():
            q_proj.weight.copy_(q_weight)
            if q_bias is not None:
                q_proj.bias.copy_(q_bias)
                
        k_proj = nn.Linear(embed_dim, embed_dim, bias=k_bias is not None)
        with torch.no_grad():
            k_proj.weight.copy_(k_weight)
            if k_bias is not None:
                k_proj.bias.copy_(k_bias)
                
        v_proj = nn.Linear(embed_dim, embed_dim, bias=v_bias is not None)
        with torch.no_grad():
            v_proj.weight.copy_(v_weight)
            if v_bias is not None:
                v_proj.bias.copy_(v_bias)
                
        out_proj = nn.Linear(embed_dim, embed_dim, bias=out_bias is not None)
        with torch.no_grad():
            out_proj.weight.copy_(out_weight)
            if out_bias is not None:
                out_proj.bias.copy_(out_bias)
        
        return create_attention(q_proj, k_proj, v_proj, out_proj, num_heads, embed_dim)

    def _map_layernorm(self, module, input_shape, parent_name=''):
        if hasattr(module, 'normalized_shape'):
            normalized_shape = module.normalized_shape
            if isinstance(normalized_shape, (tuple, list)):
                if len(normalized_shape) == 1:
                    normalized_shape = normalized_shape[0]
                else:
                    normalized_shape = 1
                    for d in module.normalized_shape:
                        normalized_shape *= d
        else:
            normalized_shape = module.weight.shape[0]
        
        return create_layernorm(
            normalized_shape,
            module.weight.clone(),
            module.bias.clone() if module.bias is not None else None,
            module.eps)
    
    def map_layernorm2d(self, module, input_shape, parent_name=''):
        if hasattr(module, 'normalized_shape'):
            normalized_shape = module.normalized_shape
            if isinstance(normalized_shape, (tuple, list)):
                normalized_shape = normalized_shape[0]  
            else:
                normalized_shape = normalized_shape
        else:
            normalized_shape = module.weight.shape[0]

        return create_layernorm2d(
            normalized_shape,
            module.weight.clone(),
            module.bias.clone() if module.bias is not None else None,
            module.eps)

    def _map_gelu(self, module, input_shape, parent_name=''):
        return create_gelu()
    
    def _map_dropout(self, module, input_shape, parent_name=''):
        return nn.Identity()
    
    def _map_droppath(self, module, input_shape, parent_name=''):
        return nn.Identity()
    
    def _replace_unsupported_modules(self, module, parent_name='', input_shape_dict=None):
        module_class = module.__class__.__name__
        
        input_shape = None
        if input_shape_dict is not None and parent_name in input_shape_dict:
            input_shape = input_shape_dict[parent_name]

        if isinstance(module, nn.Dropout):
            return nn.Identity()

        if module_class in self.op_mappings:
            try:
                mapped = self.op_mappings[module_class](module, input_shape, parent_name)
                print(f"Mapped {module_class} at {parent_name} to {self.op_mappings[module_class]}")
                return mapped
            except Exception as e:
                print(f"Mapping failed for {module_class} at {parent_name}: {e}")
                traceback.print_exc()
                return nn.Identity()

        if len(list(module.children())) > 0:
            if isinstance(module, nn.Sequential):
                new_module = nn.Sequential()
                for idx, child in enumerate(module.children()):
                    new_child = self._replace_unsupported_modules(child, f"{parent_name}[{idx}]", input_shape_dict)
                    new_module.add_module(str(idx), new_child)
                return new_module
            elif isinstance(module, nn.ModuleList):
                new_module = nn.ModuleList()
                for idx, child in enumerate(module.children()):
                    new_child = self._replace_unsupported_modules(child, f"{parent_name}[{idx}]", input_shape_dict)
                    new_module.append(new_child)
                return new_module
            else:
                for name, child in module.named_children():
                    full_name = f"{parent_name}.{name}" if parent_name else name
                    new_child = self._replace_unsupported_modules(child, full_name, input_shape_dict)
                    if new_child is not child:
                        setattr(module, name, new_child)
                return module
        
        if module_class in self.constraints['supported_ops']:
            return module
        
        print(f"Unsupported module {module_class} at {parent_name}")
        return module

    def validate_conversion(self, original_model, converted_model, test_input, input_shape_dict):
        original_modules = dict(original_model.named_modules())
        converted_modules = dict(converted_model.named_modules())
        module_validation_results = []
        
        for name, original_module in original_modules.items():
            if name == "" or len(list(original_module.children())) > 0:
                continue 
            
            if name in converted_modules:
                converted_module = converted_modules[name]
                result = self.validator.validate_module(original_module, converted_module, name, input_shape_dict)
                module_validation_results.append(result['passed'])
            else:
                print(f"Module {name} was restructured")
        
        _ = self.validator.validate_full_model(original_model, converted_model, test_input)
        
        for name, module in converted_model.named_modules():
            if len(list(module.children())) == 0: 
                module_type = type(module).__name__
                if module_type not in self.constraints['supported_ops'] and module_type not in ['Identity', 'ReLU']:
                    print(f"Unsupported op found: {name} -> {module_type}")
        
        self.validator.print_validation_summary()

    def get_all_supported_operations(self, model):
        supported_ops = {}
        
        for name, module in model.named_modules():
            if len(list(module.children())) == 0: 
                module_type = type(module).__name__

                supported_ops[name] = {
                    'type': module_type,
                    'module': module}
                
                if isinstance(module, nn.Linear):
                    supported_ops[name]['in_features'] = module.in_features
                    supported_ops[name]['out_features'] = module.out_features
                    supported_ops[name]['has_bias'] = module.bias is not None
                elif isinstance(module, nn.Conv2d):
                    supported_ops[name]['in_channels'] = module.in_channels
                    supported_ops[name]['out_channels'] = module.out_channels
                    supported_ops[name]['kernel_size'] = module.kernel_size
                    supported_ops[name]['stride'] = module.stride
                    supported_ops[name]['padding'] = module.padding
                elif isinstance(module, nn.Conv1d):
                    supported_ops[name]['in_channels'] = module.in_channels
                    supported_ops[name]['out_channels'] = module.out_channels
                    supported_ops[name]['kernel_size'] = module.kernel_size
                    supported_ops[name]['stride'] = module.stride
                    supported_ops[name]['padding'] = module.padding
        
        return supported_ops

    def _analyze_model(self, model):
        model_info = {
            'embed_dim': getattr(model, 'embed_dim', None),
            'num_heads': getattr(model, 'num_heads', None),
            'depth': len(getattr(model, 'blocks', [])),
            'has_patch_embed': hasattr(model, 'patch_embed'),
            'mlp_ratio': getattr(model, 'mlp_ratio', 4.0),
            'input_resolution': getattr(model, 'image_size', 32),
            'patch_size': getattr(getattr(model, 'patch_embed', None), 'patch_size', 4),
        }
        return model_info

    def convert(self, original_model, example_input=None):
        model_copy = copy.deepcopy(original_model)
        input_shape_dict = None
        if example_input is not None:
            input_shape_dict = trace_module_input_shapes(model_copy, example_input)
        adapted = self._replace_unsupported_modules(model_copy, input_shape_dict=input_shape_dict)
        return adapted, input_shape_dict
    
    def print_torch_operators(self, model):
        torch_ops = {}
        for _, module in model.named_modules():
            if len(list(module.children())) == 0:
                op_type = type(module).__name__
                torch_ops[op_type] = torch_ops.get(op_type, 0) + 1
        print("\nOperator summary:")
        for op, count in sorted(torch_ops.items(), key=lambda x: x[1], reverse=True):
            print(f"{op:20s}: {count:3d}")

    def convert_model(self, model, dataset_name='CIFAR10', test_input=None):
        test_input = test_input.float()
        if test_input.shape[-1] in [1, 3]:  # TODO fix
            print("   Converting NHWC -> NCHW")
            test_input = test_input.permute(0, 3, 1, 2)

        adapted_model, input_shape_dict = self.convert(model, test_input)
        # adapted_model = self.fuse_linear_chains(adapted_model)

        _ = self.validate_conversion(model, adapted_model, test_input, input_shape_dict)

        supported_ops = self.get_all_supported_operations(adapted_model)

        print(f"- Op types: {set(op['type'] for op in supported_ops.values())}")
        self.print_torch_operators(adapted_model)

        print("\nModel breakdown:")
        for name, module in adapted_model.named_modules():
            if len(list(module.children())) == 0:
                print(f"  {name}: {type(module).__name__}")

        return adapted_model