import torch
import torch.nn as nn

class ModelVal:
    def __init__(self, rtol=1e-2, atol=1e-3):
        self.rtol = rtol 
        self.atol = atol 
        self.validation_results = {}

    def _generate_test_inputs(self, converted_module):
        test_inputs = []
        if isinstance(converted_module, nn.Linear):
            for batch_size in [1, 4]:
                for input_range in [(-1, 1), (-5, 5)]:
                    test_inputs.append(torch.randn(batch_size, converted_module.in_features) * input_range[1])

        elif isinstance(converted_module, nn.LayerNorm):
            shape = getattr(converted_module, 'normalized_shape', converted_module.weight.shape[0])
            shape = (shape,) if isinstance(shape, int) else tuple(shape)
            for batch_size in [1, 4]:
                for seq_len in [8, 16]:
                    for input_range in [(-2, 2), (-10, 10)]:
                        test_inputs.append(torch.randn(batch_size, seq_len, *shape) * input_range[1])

        elif isinstance(converted_module, nn.MultiheadAttention):
            embed_dim = converted_module.embed_dim
            batch_first = getattr(converted_module, 'batch_first', False)
            for batch_size in [1, 2]:
                for seq_len in [4, 8]:
                    shape = (batch_size, seq_len, embed_dim) if batch_first else (seq_len, batch_size, embed_dim)
                    test_inputs.append(torch.randn(*shape))

        elif isinstance(converted_module, (nn.GELU, nn.ReLU, nn.Softmax)):
            for shape in [(2, 10), (1, 5, 8), (2, 4, 16)]:
                for input_range in [(-3, 3), (-10, 10)]:
                    test_inputs.append(torch.randn(*shape) * input_range[1])

        elif isinstance(converted_module, nn.Conv2d):
            in_ch = converted_module.in_channels
            for batch_size in [1, 2]:
                for img_size in [16, 32]:
                    test_inputs.append(torch.randn(batch_size, in_ch, img_size, img_size))

        elif isinstance(converted_module, nn.Conv1d):
            in_ch = converted_module.in_channels
            for batch_size in [1, 2]:
                for seq_len in [16, 32]:
                    test_inputs.append(torch.randn(batch_size, in_ch, seq_len))
        return test_inputs
        
    def validate_module(self, original_module, converted_module, module_name, input_shape_dict, test_inputs=None):
        print(f"\nValidating: {module_name}")
        
        if module_name in input_shape_dict:
            input_shape = input_shape_dict[module_name]
            test_inputs = [torch.randn(input_shape)]
        else:
            test_inputs = self._generate_test_inputs(converted_module)

        original_module.eval()
        converted_module.eval()

        errors = []
        validation_passed = True
        with torch.no_grad():
            for test_input in test_inputs:
                try:
                    if isinstance(original_module, nn.MultiheadAttention):
                        if isinstance(test_input, tuple):
                            query, key, value = test_input
                        else:
                            query = key = value = test_input
                        orig_out, _ = original_module(query, key, value, need_weights=True)
                        conv_out, _ = converted_module(query, key, value, need_weights=True)
                    else:
                        orig_out = original_module(test_input)
                        conv_out = converted_module(test_input)

                    error = torch.mean(torch.abs(orig_out - conv_out)).item()
                    errors.append(error)
                    if not torch.allclose(orig_out, conv_out, rtol=self.rtol, atol=self.atol):
                        if error > 0.01:
                            validation_passed = False
                except Exception as e:
                    validation_passed = False
                    errors.append(float('inf'))

        mean_error = sum(errors) / len(errors) if errors else float('inf')

        result = {
            'passed': validation_passed,
            'mean_error': mean_error,
            'og_type': type(original_module).__name__,
            'new_type': type(converted_module).__name__}

        self.validation_results[module_name] = result

        status = "✅ PASS" if validation_passed else "❌ FAIL"
        print(f"   {status} - Mean error: {mean_error:.2e}")
        return result

    
    def validate_full_model(self, original_model, converted_model, test_input=None):
        print("\n Val full model...")

        try:
            original_model.eval()
            converted_model.eval()

            with torch.no_grad():
                if test_input is None:
                    if hasattr(original_model, 'image_size'):
                        img_size = original_model.image_size
                        test_input = torch.randn(1, 3, img_size, img_size)
                    else:
                        test_input = torch.randn(1, 3, 32, 32) # TODO fix
                test_input = test_input.clone().detach()
                if test_input.shape[-1] in [1, 3]:  # TODO fix
                    print("   Converting NHWC -> NCHW")
                    test_input = test_input.permute(0, 3, 1, 2)
                print(f"   Input shape: {test_input.shape}")

                first_conv = None
                for m in original_model.modules():
                    if isinstance(m, nn.Conv2d):
                        first_conv = m
                        break

                if first_conv:
                    expected_in_channels = first_conv.in_channels
                    if test_input.shape[1] != expected_in_channels:
                        raise ValueError(f"Input channels mismatch. Model expects {expected_in_channels}, got {test_input.shape[1]}. Check input format (NCHW vs NHWC).")

                orig_output = original_model(test_input.float())
                conv_output = converted_model(test_input.float())

                error = torch.abs(orig_output - conv_output)
                max_error = error.max().item()
                mean_error = error.mean().item()
                relative_error = (error / (torch.abs(orig_output) + 1e-8)).mean().item()

                outputs_close = torch.allclose(orig_output, conv_output, rtol=self.rtol * 10, atol=self.atol * 10)

                print(f"   Max error: {max_error:.2e}")
                print(f"   Mean error: {mean_error:.2e}")
                print(f"   Relative error: {relative_error:.2e}")

                if outputs_close or max_error < 0.01: 
                    print("   ✅ Full model validation PASSED")
                    model_validation_passed = True
                else:
                    print("   ❌ Full model validation FAILED")
                    model_validation_passed = False

                self.validation_results['__full_model__'] = {
                    'passed': model_validation_passed,
                    'max_error': max_error,
                    'mean_error': mean_error,
                    'relative_error': relative_error,
                    'input_shape': test_input.shape,
                    'output_shape': orig_output.shape}

                return model_validation_passed

        except Exception as e:
            print(f"   ❌ Full model val error: {str(e)}")
            import traceback
            traceback.print_exc()

            self.validation_results['__full_model__'] = {
                'passed': False,
                'error': str(e)}
            return False
    
    def print_validation_summary(self):
        total_modules = len([k for k in self.validation_results.keys() if k != '__full_model__'])
        passed_modules = len([v for k, v in self.validation_results.items() if k != '__full_model__' and v.get('passed', False)])
        
        print(f"Modules passed: {passed_modules}/{total_modules} ({passed_modules/total_modules*100 if total_modules > 0 else 0:.1f}%)")
        
        if '__full_model__' in self.validation_results:
            full_result = self.validation_results['__full_model__']
            status = "PASSED" if full_result.get('passed', False) else "FAILED"
            print(f"\nFull model validation: {status}")
            if 'max_error' in full_result:
                print(f"  Max error: {full_result['max_error']:.2e}")