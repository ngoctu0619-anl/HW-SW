import numpy as np
from IPython.display import FileLink

# Hàm xuất trọng số sang C++
def export_weights_to_c(model, filename="model_weights.h"):
    model.to("cpu")
    with open(filename, 'w') as f:
        f.write("#ifndef MODEL_WEIGHTS_H\n#define MODEL_WEIGHTS_H\n\n")
        f.write("#include <stdint.h>\n\n")
        
        for name, param in model.named_parameters():
            # Chuyển đổi và lượng tử hóa (nhân 128)
            weights = np.round(param.detach().numpy() * 128).astype(np.int32).flatten()
            c_name = name.replace('.', '_')
            f.write(f"const int32_t {c_name}[{len(weights)}] = {{")
            f.write(", ".join(map(str, weights)) + "};\n\n")
            
        f.write("#endif\n")
    print(f"Đã tạo file {filename} trong bộ nhớ.")

export_weights_to_c(model)

display(FileLink('model_weights.h'))
