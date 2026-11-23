import os
import kagglehub

output_dir = "./datasets"

# Download latest version
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    path = kagglehub.dataset_download("elmahy/pems-dataset", output_dir=output_dir)

print("✅ Dữ liệu đã được tải xuống thành công và lưu vào thư mục:", output_dir)