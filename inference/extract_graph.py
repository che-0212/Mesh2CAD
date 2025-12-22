import argparse
import json
import sys
from pathlib import Path

# Import existing modules
from repl_env import ReplEnv
from search import Search


def extract_cad_graph_simple(input_file):
    """
    Simplified CAD graph structure extraction function
    Output filename automatically uses the same name as input file
    """
    try:
        # 1. Use the same environment setup as main.py
        env = ReplEnv(host="127.0.0.1", port=8080, launch_gym=True)

        # 2. Create search instance (no logging needed)
        search = Search(env, log_dir=None)
        
        # 3. Set target file
        input_path = Path(input_file)
        if not input_path.exists():
            raise FileNotFoundError(f"File not found: {input_file}")
            
        print(f"Processing CAD file: {input_path.name}")
        
        # 4. Extract graph structure
        target_graph, target_bounding_box = search.set_target(input_path)
        
        # 5. Analyze and output results
        nodes = target_graph["nodes"]
        valid_planar_nodes = []
        surface_types = {}
        
        for node in nodes:
            surface_type = node.get("surface_type", "Unknown")
            surface_types[surface_type] = surface_types.get(surface_type, 0) + 1
            
            if surface_type == "PlaneSurfaceType":
                valid_planar_nodes.append(node["id"])
        
        # 6. Prepare output data
        output_data = {
            "file_info": {
                "input_file": str(input_path),
                "file_name": input_path.name,
                "file_stem": input_path.stem
            },
            "graph_statistics": {
                "total_nodes": len(nodes),
                "total_links": len(target_graph.get("links", [])),
                "planar_nodes_count": len(valid_planar_nodes),
                "surface_type_distribution": surface_types
            },
            "target_graph": target_graph,
            "target_bounding_box": target_bounding_box,
            "valid_planar_nodes": valid_planar_nodes
        }
        
        # 7. Console output
        print("\n" + "="*50)
        print("CAD Model Graph Structure Analysis Results")
        print("="*50)
        print(f"File name: {input_path.name}")
        print(f"Total nodes: {len(nodes)}")
        print(f"Total links: {len(target_graph.get('links', []))}")
        print(f"Planar nodes: {len(valid_planar_nodes)}")
        print("\nSurface type distribution:")
        for surface_type, count in surface_types.items():
            print(f"  {surface_type}: {count}")
        print(f"\nBounding box: {target_bounding_box}")
        
        # 8. Auto-generate output filename (use same filename but change extension to .json)
        output_file = input_path.parent / f"{input_path.stem}.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed graph structure saved to: {output_file}")
        
        return output_data
        
    except Exception as e:
        print(f"Error: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="Extract graph structure from CAD file")
    parser.add_argument("--input", type=str, required=True, 
                       help="Input CAD file path (.step or .stp format)")
    
    args = parser.parse_args()
    
    # Execute graph structure extraction
    result = extract_cad_graph_simple(input_file=args.input)
    
    if result is None:
        print("Graph structure extraction failed!")
        sys.exit(1)
    else:
        print("\nGraph structure extraction completed!")
        sys.exit(0)


if __name__ == "__main__":
    main()