"""Export Human Atlas dataset (atlas.json + binary geometry chunks) to a single GLB file and parts-info.json.

Requirements:
    pip install trimesh numpy

Usage:
    python scripts/export_glb.py
    python scripts/export_glb.py --atlas public/models/atlas.json --output-dir exported
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path
import numpy as np
import trimesh
from trimesh.visual.material import PBRMaterial

# Standard system metadata and curated colors (matching app/anatomy.ts)
SYSTEMS_METADATA = {
    'skeletal': {'name': 'Skeleton', 'color': [0.886, 0.851, 0.729, 1.0]},      # #e2d9ba
    'muscular': {'name': 'Muscles', 'color': [0.659, 0.357, 0.314, 1.0]},       # #a85b50
    'cardiac': {'name': 'Heart', 'color': [0.725, 0.404, 0.376, 1.0]},          # #b96760
    'sensory': {'name': 'Sensory organs', 'color': [0.690, 0.784, 0.808, 1.0]}, # #b0c8ce
    'arterial': {'name': 'Arteries', 'color': [0.753, 0.322, 0.271, 1.0]},      # #c05245
    'venous': {'name': 'Veins', 'color': [0.322, 0.486, 0.624, 1.0]},           # #527c9f
    'nervous': {'name': 'Nervous system', 'color': [0.847, 0.710, 0.396, 1.0]}, # #d8b565
    'respiratory': {'name': 'Respiratory', 'color': [0.725, 0.537, 0.569, 1.0]},# #b98991
    'digestive': {'name': 'Digestive', 'color': [0.722, 0.569, 0.420, 1.0]},    # #b8916b
    'urinary': {'name': 'Urinary', 'color': [0.706, 0.475, 0.380, 1.0]},        # #b47961
    'lymphatic': {'name': 'Lymphatic', 'color': [0.529, 0.624, 0.486, 1.0]},    # #879f7c
    'endocrine': {'name': 'Endocrine', 'color': [0.773, 0.627, 0.604, 1.0]},    # #c5a09a
    'reproductive': {'name': 'Reproductive', 'color': [0.741, 0.627, 0.596, 1.0]}, # #bda098
    'integumentary': {'name': 'Body surface', 'color': [0.729, 0.608, 0.490, 0.2]},# #ba9b7d (semi-transparent)
    'connective': {'name': 'Connective tissue', 'color': [0.682, 0.765, 0.733, 1.0]} # #aec3bb
}


def hex_to_rgba(hex_code, alpha=1.0):
    hex_code = hex_code.lstrip('#')
    r = int(hex_code[0:2], 16) / 255.0
    g = int(hex_code[2:4], 16) / 255.0
    b = int(hex_code[4:6], 16) / 255.0
    return [r, g, b, alpha]


def load_atlas_and_chunks(atlas_path: Path):
    with open(atlas_path, 'r', encoding='utf-8') as f:
        atlas = json.load(f)

    models_dir = atlas_path.parent
    chunks_data = {}
    for i, chunk_info in enumerate(atlas['chunks']):
        chunk_rel = chunk_info['url'].lstrip('/').split('models/')[-1]
        chunk_file = models_dir / chunk_rel

        if not chunk_file.exists():
            # Fallback to absolute relative resolution
            chunk_file = atlas_path.parents[1] / chunk_info['url'].lstrip('/')

        if not chunk_file.exists():
            raise FileNotFoundError(f"Missing chunk binary file: {chunk_file}")

        with open(chunk_file, 'rb') as f:
            chunks_data[i] = f.read()

    return atlas, chunks_data


def export_atlas_to_glb(atlas_path: Path, output_dir: Path, output_name: str = "human-atlas.glb"):
    start_time = time.time()
    output_dir.mkdir(parents=True, exist_ok=True)
    glb_path = output_dir / output_name
    info_path = output_dir / "parts-info.json"

    print(f"Reading atlas manifest: {atlas_path}")
    atlas, chunks_data = load_atlas_and_chunks(atlas_path)

    parts = atlas['parts']
    print(f"Found {len(parts)} anatomical parts across {len(chunks_data)} binary chunks.")

    # Create scene
    scene = trimesh.Scene()

    # Pre-create system parent hierarchy nodes in scene graph
    present_systems = sorted(list(set(p.get('system', 'connective') for p in parts)))
    system_materials = {}

    for sys_id in present_systems:
        meta = SYSTEMS_METADATA.get(sys_id, {'name': sys_id.capitalize(), 'color': [0.7, 0.7, 0.7, 1.0]})
        sys_node_name = f"System_{meta['name'].replace(' ', '_')}"
        scene.graph.update(frame_to=sys_node_name, frame_from='world')

        # Create reusable PBR Material for each anatomical system
        color = meta['color']
        mat = PBRMaterial(
            name=f"Mat_{sys_id}",
            baseColorFactor=color,
            metallicFactor=0.08,
            roughnessFactor=0.53,
            alphaMode='BLEND' if color[3] < 1.0 else 'OPAQUE'
        )
        system_materials[sys_id] = mat

    parts_info = []

    print("Decoding geometries and assembling scene...")
    for idx, p in enumerate(parts):
        chunk_idx = p['chunk']
        buf = chunks_data[chunk_idx]
        v_count = p['vertexCount']
        i_count = p['indexCount']

        # 1. Decode positions (Float32Array)
        positions = np.frombuffer(
            buf, dtype=np.float32, count=v_count * 3, offset=p['positions']
        ).reshape((v_count, 3))

        # 2. Decode and dequantize normals (Int16Array quantized by 32767)
        normals = np.frombuffer(
            buf, dtype=np.int16, count=v_count * 3, offset=p['normals']
        ).astype(np.float32).reshape((v_count, 3)) / 32767.0

        # Normalize normal vectors
        norm_len = np.linalg.norm(normals, axis=1, keepdims=True)
        norm_len[norm_len == 0] = 1.0
        normals = normals / norm_len

        # 3. Decode indices (Uint32Array)
        indices = np.frombuffer(
            buf, dtype=np.uint32, count=i_count, offset=p['indices']
        ).reshape((i_count // 3, 3))

        # 4. Create Trimesh geometry
        mesh = trimesh.Trimesh(
            vertices=positions,
            faces=indices,
            vertex_normals=normals,
            process=False
        )

        sys_id = p.get('system', 'connective')
        mesh.visual.material = system_materials.get(sys_id)

        # Node name sanitized for Unity / 3D DCC tools
        part_id = p['id']
        part_name = p['name']
        sanitized_name = part_name.replace('/', '_').replace('\\', '_')
        node_name = f"{part_id}_{sanitized_name}"
        mesh.metadata['id'] = part_id
        mesh.metadata['name'] = part_name
        mesh.metadata['conceptId'] = p.get('conceptId', '')
        mesh.metadata['system'] = sys_id

        sys_meta = SYSTEMS_METADATA.get(sys_id, {'name': sys_id.capitalize()})
        sys_node_name = f"System_{sys_meta['name'].replace(' ', '_')}"

        # Add geometry to scene graph under its anatomical system parent
        scene.add_geometry(
            mesh,
            node_name=node_name,
            geom_name=node_name,
            parent_node_name=sys_node_name
        )

        # Build parts-info entry
        parts_info.append({
            'id': part_id,
            'name': part_name,
            'conceptId': p.get('conceptId', ''),
            'system': sys_id,
            'systemName': sys_meta['name'],
            'vertexCount': v_count,
            'triangleCount': i_count // 3,
            'bounds': p.get('bounds', [])
        })

        if (idx + 1) % 500 == 0 or (idx + 1) == len(parts):
            print(f"Processed {idx + 1}/{len(parts)} parts...")

    # Write simplified parts-info.json
    print(f"Saving parts metadata: {info_path}")
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump({
            'version': atlas.get('version', 'BodyParts3D 4.0'),
            'totalParts': len(parts_info),
            'parts': parts_info
        }, f, indent=2)

    # Export to GLB
    print(f"Exporting GLB scene to {glb_path} (this may take a few moments)...")
    glb_bytes = scene.export(file_type='glb')
    with open(glb_path, 'wb') as f:
        f.write(glb_bytes)

    size_mb = len(glb_bytes) / (1024 * 1024)
    elapsed = time.time() - start_time
    print(f"Export completed in {elapsed:.2f}s!")
    print(f"Output GLB: {glb_path} ({size_mb:.2f} MB)")
    print(f"Output JSON: {info_path}")


def main():
    parser = argparse.ArgumentParser(description="Export human atlas models to a single hierarchy-preserved GLB.")
    parser.add_argument('--atlas', type=str, default='public/models/atlas.json', help="Path to atlas.json")
    parser.add_argument('--output-dir', type=str, default='exported', help="Output directory")
    parser.add_argument('--output-name', type=str, default='human-atlas.glb', help="Output GLB filename")
    args = parser.parse_args()

    atlas_path = Path(args.atlas).resolve()
    output_dir = Path(args.output_dir).resolve()

    export_atlas_to_glb(atlas_path, output_dir, args.output_name)


if __name__ == '__main__':
    main()
