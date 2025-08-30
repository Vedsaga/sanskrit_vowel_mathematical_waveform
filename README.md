### To run your GMM model:
cargo run --release -- --model-type gmm --model-path "<directory_path_where_models>"
### To run your new PyTorch/ONNX model:
cargo run --release -- --model-type onnx --model-path "<path_to_model>.onnx"
