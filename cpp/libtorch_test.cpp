#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

int main() {
    // 1. Setup Device
    torch::Device device(torch::kCPU);
    if (torch::cuda::is_available()) {
        std::cout << "Using CUDA/GPU for inference." << std::endl;
        device = torch::Device(torch::kCUDA);
    } else {
        std::cout << "CUDA not available. Falling back to CPU." << std::endl;
        return -1;
    }

    // 2. Load the Traced Model
    torch::jit::script::Module module;
    try {
        // Use the exact path where you saved the .pt file
        module = torch::jit::load("D:/Code/GNN/traced_models/gognn_traced.pt");
        module.to(device);
        module.eval();  
    } catch (const c10::Error& e) {
        std::cerr << "Error loading the model: " << e.msg() << std::endl;
        return -1;
    }

    // 3. Create Dummy Input (1 Piece of Data: 1 Board)
    // We use TensorOptions to ensure data is created directly on GPU to save time/VRAM
    auto float_opts = torch::TensorOptions().dtype(torch::kFloat32).device(device);
    auto long_opts = torch::TensorOptions().dtype(torch::kInt64).device(device);

    int N = 361; // Stones (19x19)
    int S = 20;  // Assume 20 strings/groups on board
    int G = 1;   // Global node

    // Node Features
    torch::Tensor stone_x = torch::randn({N, 18}, float_opts);
    torch::Tensor string_x = torch::randn({S, 2}, float_opts);
    torch::Tensor global_x = torch::randn({G, 19}, float_opts);

    // Edge Indices (Minimal edges to keep VRAM low)
    torch::Tensor e_s_a_s = torch::randint(0, N, {2, 100}, long_opts);
    torch::Tensor e_s_b_str = torch::stack({torch::randint(0, N, {50}, long_opts), 
                                            torch::randint(0, S, {50}, long_opts)});
    torch::Tensor e_str_c_s = torch::stack({torch::randint(0, S, {50}, long_opts), 
                                            torch::randint(0, N, {50}, long_opts)});
    torch::Tensor e_str_a_str = torch::randint(0, S, {2, 40}, long_opts);
    torch::Tensor e_str_r_g = torch::stack({torch::randint(0, S, {20}, long_opts), 
                                            torch::zeros({20}, long_opts)});
    torch::Tensor e_g_i_str = torch::stack({torch::zeros({20}, long_opts), 
                                            torch::randint(0, S, {20}, long_opts)});

    // 4. Wrap into IValue vector (The order must match your Python forward call)
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(stone_x);
    inputs.push_back(string_x);
    inputs.push_back(global_x);
    inputs.push_back(e_s_a_s);
    inputs.push_back(e_s_b_str);
    inputs.push_back(e_str_c_s);
    inputs.push_back(e_str_a_str);
    inputs.push_back(e_str_r_g);
    inputs.push_back(e_g_i_str);

    // 5. Run Inference
    std::cout << "Starting Inference..." << std::endl;
    
    // Disable gradient calculation for speed and VRAM
    torch::NoGradGuard no_grad; 
    
    auto output = module.forward(inputs).toTuple();

    // 6. Extract results
    torch::Tensor policy = output->elements()[0].toTensor();
    torch::Tensor pass_val = output->elements()[1].toTensor();
    torch::Tensor value = output->elements()[2].toTensor();
    torch::Tensor ownership = output->elements()[3].toTensor();

    // 7. Print verification
    std::cout << "Success!" << std::endl;
    std::cout << "Policy Shape: " << policy.sizes() << " (Expected [361])" << std::endl;
    std::cout << "Board Value: " << value.item<float>() << std::endl;
    std::cout << "Pass Prob: " << pass_val.item<float>() << std::endl;

    return 0;
}