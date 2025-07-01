opswap/main -> quantize -> ai8xize 

python ai8xize.py --verbose --test-dir demos --prefix ai85-vit --checkpoint-file og_model_new_patch_size_1_q8.pth.tar --config-file networks/ai8x-vit.yaml --device MAX78000 --overwrite --debug   
