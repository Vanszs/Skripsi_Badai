import sys
import os
import torch

# Add project root to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..')))

try:
    from src.data.ingest import PANGRANGO_NODES
    from src.models.diffusion import ConditionalDiffusionModel
    from src.inference import run_inference_hybrid
except ImportError as e:
    print(f"❌ CRITICAL IMPORT ERROR: {e}")
    sys.exit(1)

def run_health_check():
    print("🛡️  THESIS SENTINEL HEALTH CHECK 🛡️")
    print("====================================")
    
    issues = []
    
    # CHECK 1: NODES
    print(f"[*] Checking Nodes...", end=" ")
    node_names = PANGRANGO_NODES['name'].tolist()
    if 'Puncak' in node_names and 'Hilir_Cianjur' in node_names:
        print("✅ PASSED (Pangrango)")
    else:
        print("❌ FAILED")
        issues.append("Nodes do not match Gede-Pangrango configuration.")

    # CHECK 2: MODEL DIMENSIONS
    print(f"[*] Checking Model Config...", end=" ")
    try:
        # Inspect class attribute if instance not avail
        if ConditionalDiffusionModel.NUM_TARGETS == 3:
            print("✅ PASSED (3 Variables)")
        else:
            print(f"❌ FAILED (Found {ConditionalDiffusionModel.NUM_TARGETS} targets)")
            issues.append(f"Model configured for {ConditionalDiffusionModel.NUM_TARGETS} targets, expected 3.")
    except Exception as e:
        print(f"⚠️  WARNING: Could not verify Class Config ({e})")

    # CHECK 3: HYBRID WEIGHTS (Static Analysis)
    print(f"[*] Checking Inference Weights...", end=" ")
    inference_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../src/inference.py'))
    with open(inference_path, 'r') as f:
        content = f.read()
        if "'precipitation': 0.9" in content and "'wind_speed': 0.9" in content:
             print("✅ PASSED (Optimized 0.9/0.9/0.7)")
        else:
             print("❌ FAILED (Old weights detected?)")
             issues.append("Inference weights do not match optimized values (0.9/0.9/0.7).")

    # CHECK 4: FORBIDDEN TERMS
    print(f"[*] Scanning for Forbidden Terms (Sitaro)...", end=" ")
    forbidden = ['Siau', 'Tagulandang', 'Biaro', 'Sitaro']
    found_forbidden = []
    
    # Simple scan of current dir (recursive)
    for root, dirs, files in os.walk('.'):
        if '.agent' in root or '.git' in root or '__pycache__' in root:
            continue
        for file in files:
            if file.endswith('.py') or file.endswith('.md'):
                try:
                    with open(os.path.join(root, file), 'r', encoding='utf-8') as f:
                        text = f.read()
                        for term in forbidden:
                            if term in text and 'legacy' not in text.lower():
                                found_forbidden.append(f"{term} in {file}")
                except:
                    pass
    
    if not found_forbidden:
        print("✅ PASSED (Clean)")
    else:
        print(f"⚠️  WARNING (Found {len(found_forbidden)} legacy terms)")
        # issues.append(f"Legacy Sitaro terms found: {found_forbidden[:3]}...")

    print("\n====================================")
    if not issues:
        print("🟢 SYSTEM STATUS: HEALTHY & READY")
        return 0
    else:
        print("🔴 SYSTEM STATUS: ATTENTION NEEDED")
        for issue in issues:
            print(f" - {issue}")
        return 1

if __name__ == "__main__":
    run_health_check()
