# AI CAD OS: Autonomous Image-to-Parametric CAD Pipeline

**AI CAD OS** is a next-generation system designed to transform single 2D images into high-fidelity, parametric, and manufacturable CAD models. Unlike traditional image-to-mesh systems, AI CAD OS focuses on recovering the underlying **design intent** and **modeling history**, enabling engineers to edit and manufacture the output in industry-standard CAD software.

---

## 🏗️ Architecture

The system implements a four-stage pipeline that bridges generative diffusion models with deterministic engineering logic.

### 1. [MODEL_GENERATOR_V2](./MODEL_GENERATOR_V2) (Generative Core)
*   **Purpose**: High-quality 3D mesh generation.
*   **Engine**: Based on **Tencent Hunyuan3D-2.1** (Flow-matching DiT).
*   **Output**: High-resolution meshes and, crucially, internal **DiT hidden states** used for semantic probing.

### 2. [geometry-engine](./geometry-engine) (Geometric Probing)
*   **Purpose**: Extracts the **Geometry Graph Language (GGL)**.
*   **Key Innovation**: Probes the Latent Diffusion Model's internal representations to identify primitives (Cylinders, Boxes, Holes) and their relationships (Symmetry, Containment) before they are collapsed into a mesh.

### 3. [cad-planner](./cad-planner) (Reasoning & Planning)
*   **Purpose**: Transforms GGL into **CAD Action Language (CAL)**.
*   **Workflow**:
    *   **Intent Classification**: Assigns engineering meaning to geometric shapes.
    *   **Beam Search Planning**: Explores multiple construction sequences to find the most "idiomatic" CAD history.
    *   **Manufacturability Analysis**: Scores the design for production constraints.

### 4. [desktop-agent](./desktop-agent) (Execution & Feedback)
*   **Purpose**: Executes CAL actions in target CAD software.
*   **Current Support**: **FreeCAD** (via Python API).
*   **Function**: Orchestrates the actual modeling steps (Sketches, Extrusions, Fillets) and provides a verification loop.

---

## 🛠️ The Languages of Design

AI CAD OS uses two primary intermediate representations defined in the [shared-schemas](./shared-schemas) directory:

| Language | Type | Description |
| --- | --- | --- |
| **GGL** | Graph | **Geometry Graph Language**: Nodes (primitives) and Edges (relationships). Software-independent semantic representation. |
| **CAL** | Sequential | **CAD Action Language**: A list of modeling operations (e.g., `extrude`, `revolve`) with embedded reasoning and traceability. |

---

## 🚀 Getting Started

### Prerequisites
*   Python 3.8+
*   PyTorch (for `MODEL_GENERATOR_V2` and `geometry-engine`)
*   FreeCAD (for `desktop-agent` execution)

### Installation
```bash
# Clone the repository
git clone https://github.com/antigravity-research/ai-cad-os.git
cd ai-cad-os

# Install shared schemas first
pip install -e shared-schemas/

# Install components as needed
pip install -e MODEL_GENERATOR_V2/
pip install -e cad-planner/
pip install -e geometry-engine/
```

### Running the Pipeline
You can run a mock end-to-end planning simulation:
```bash
python cad-planner/scripts/run_pipeline.py
```

---

## 📐 Project Structure

```text
C:\Users\proka\.gemini\antigravity\scratch\
├── ai-backend-gateway/   # API orchestration
├── cad-planner/          # GGL -> CAL transformation logic
├── desktop-agent/        # CAD software automation (FreeCAD)
├── geometry-engine/      # Semantic extraction from DiT hidden states
├── MODEL_GENERATOR_V2/   # Generative 3D core (Hunyuan3D-2.1 based)
└── shared-schemas/       # Central GGL/CAL definitions
```

---

## 📜 Principles & Invariants

1.  **Semantic Primacy**: The system must never reconstruct CAD from raw meshes alone. It must always prioritize data derived from the generative model's hidden states (**GGL Source Integrity**).
2.  **Explainability**: Every CAL action must include a `reasoning` block explaining its engineering purpose.
3.  **Software Neutrality**: CAL and GGL are designed to be software-agnostic, allowing for future support of SolidWorks, Fusion 360, and others.

---

## ⚖️ License
This project utilizes components based on Tencent Hunyuan3D-2.1, subject to the TENCENT HUNYUAN NON-COMMERCIAL LICENSE AGREEMENT.
"# model" 
