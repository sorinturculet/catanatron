"""
Verify that the GNN implementation has correct mappings.
Run this before training to catch any issues.
"""

from catanatron.models.board import get_edges, STATIC_GRAPH
from catanatron.models.map import NUM_NODES, NUM_EDGES
from catanatron_gym.board_tensor_features import get_node_and_edge_maps

def verify_mappings():
    print("=" * 60)
    print("VERIFYING GNN MAPPINGS")
    print("=" * 60)

    node_map, edge_map = get_node_and_edge_maps()
    land_edges = get_edges(frozenset(range(NUM_NODES)))

    print(f"\nExpected:")
    print(f"  Land nodes: {NUM_NODES} (IDs 0-53)")
    print(f"  Land edges: {NUM_EDGES}")

    print(f"\nActual:")
    print(f"  node_map has {len(node_map)} nodes")
    print(f"  edge_map has {len(edge_map)} entries (includes both directions)")
    print(f"  land_edges has {len(land_edges)} edges")

    # Check 1: Are all land nodes (0-53) in node_map?
    print(f"\n--- Check 1: Land nodes in node_map ---")
    missing_land_nodes = []
    for node_id in range(NUM_NODES):
        if node_id not in node_map:
            missing_land_nodes.append(node_id)

    if missing_land_nodes:
        print(f"  FAIL: {len(missing_land_nodes)} land nodes missing from node_map!")
        print(f"  Missing: {missing_land_nodes}")
    else:
        print(f"  PASS: All {NUM_NODES} land nodes (0-53) are in node_map")

    # Check 2: Are all land edges in edge_map?
    print(f"\n--- Check 2: Land edges in edge_map ---")
    missing_edges = []
    for (src, dst) in land_edges:
        if (src, dst) not in edge_map and (dst, src) not in edge_map:
            missing_edges.append((src, dst))

    if missing_edges:
        print(f"  FAIL: {len(missing_edges)} land edges missing from edge_map!")
        print(f"  Missing: {missing_edges[:10]}...")  # Show first 10
    else:
        print(f"  PASS: All {NUM_EDGES} land edges are in edge_map")

    # Check 3: What extra nodes are in node_map (water nodes)?
    print(f"\n--- Check 3: Extra nodes in node_map (water) ---")
    water_nodes = [n for n in node_map.keys() if n >= NUM_NODES]
    print(f"  Water nodes in node_map: {len(water_nodes)}")
    print(f"  Water node IDs: {sorted(water_nodes)}")

    # Check 4: Verify land node coordinates are valid
    print(f"\n--- Check 4: Coordinate validity ---")
    WIDTH, HEIGHT = 21, 11
    invalid_coords = []
    for node_id in range(NUM_NODES):
        if node_id in node_map:
            x, y = node_map[node_id]
            if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
                invalid_coords.append((node_id, x, y))

    if invalid_coords:
        print(f"  FAIL: {len(invalid_coords)} nodes have invalid coordinates!")
        print(f"  Invalid: {invalid_coords[:10]}...")
    else:
        print(f"  PASS: All land node coordinates are within bounds (0-20, 0-10)")

    # Check 5: Edge coordinates
    print(f"\n--- Check 5: Edge coordinate validity ---")
    invalid_edge_coords = []
    for (src, dst) in land_edges:
        if (src, dst) in edge_map:
            x, y = edge_map[(src, dst)]
        elif (dst, src) in edge_map:
            x, y = edge_map[(dst, src)]
        else:
            continue
        if x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT:
            invalid_edge_coords.append(((src, dst), x, y))

    if invalid_edge_coords:
        print(f"  FAIL: {len(invalid_edge_coords)} edges have invalid coordinates!")
    else:
        print(f"  PASS: All land edge coordinates are within bounds")

    # Summary
    print("\n" + "=" * 60)
    all_pass = (
        len(missing_land_nodes) == 0 and
        len(missing_edges) == 0 and
        len(invalid_coords) == 0 and
        len(invalid_edge_coords) == 0
    )
    if all_pass:
        print("ALL CHECKS PASSED - GNN implementation should work!")
    else:
        print("SOME CHECKS FAILED - GNN implementation needs fixes!")
    print("=" * 60)

    return all_pass

if __name__ == "__main__":
    verify_mappings()
