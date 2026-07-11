import os
import base64

files_to_bundle = [
    ("MODEL_GENERATOR_V2/core/hidden_state_bridge.py", "MODEL_GENERATOR_V2/core/hidden_state_bridge.py"),
    ("geometry-engine/heads/primitive.py", "geometry-engine/heads/primitive.py"),
    ("geometry-engine/heads/symmetry.py", "geometry-engine/heads/symmetry.py"),
    ("geometry-engine/heads/part.py", "geometry-engine/heads/part.py"),
    ("geometry-engine/heads/surface.py", "geometry-engine/heads/surface.py"),
    ("geometry-engine/heads/__init__.py", "geometry-engine/heads/__init__.py"),
    ("geometry-engine/graph/ggl_builder.py", "geometry-engine/graph/ggl_builder.py")
]

script_content = "#!/bin/bash\n"
script_content += "echo 'Starting update installation...'\n"
script_content += "mkdir -p MODEL_GENERATOR_V2/core geometry-engine/heads geometry-engine/graph\n\n"

for local_path, remote_path in files_to_bundle:
    full_local_path = os.path.join("C:/Users/proka/.gemini/antigravity/scratch", local_path)
    if os.path.exists(full_local_path):
        with open(full_local_path, "rb") as f:
            encoded = base64.b64encode(f.read()).decode("utf-8")
        
        script_content += f"echo 'Writing {remote_path}...'\n"
        script_content += f"base64 -d << 'EOF' > {remote_path}\n"
        # Wrap base64 lines for readability/terminal constraints
        for i in range(0, len(encoded), 76):
            script_content += encoded[i:i+76] + "\n"
        script_content += "EOF\n\n"
    else:
        print(f"Warning: file {full_local_path} not found")

script_content += "echo 'Updates successfully installed!'\n"

with open("C:/Users/proka/.gemini/antigravity/scratch/install_updates.sh", "w", newline="\n") as f:
    f.write(script_content)
print("install_updates.sh generated successfully.")
