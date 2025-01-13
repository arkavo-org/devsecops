// Constants for configuration
const GRID_SIZE: u32 = 128u;
const WORKGROUP_SIZE: u32 = 8u;

// Struct for grid dimensions and simulation parameters
struct SimParams {
    grid_size: vec2<u32>,
    wrap_edges: u32,  // 0 or 1 to control edge behavior
    birth_rule: u32,  // Typically 3
    survival_min: u32, // Typically 2
    survival_max: u32, // Typically 3
}

@group(0) @binding(0) var<storage, read> src_cells: array<u32>;
@group(0) @binding(1) var<storage, read_write> dst_cells: array<u32>;
@group(0) @binding(2) var<uniform> params: SimParams;

// Utility function to get array index with bounds checking
fn get_cell_index(pos: vec2<u32>) -> u32 {
    let grid_size = params.grid_size;
    if (pos.x >= grid_size.x || pos.y >= grid_size.y) {
        return 0u;
    }
    return pos.y * grid_size.x + pos.x;
}

// Function to get cell state with configurable edge behavior
fn get_cell_state(pos: vec2<u32>) -> u32 {
    let grid_size = params.grid_size;
    var adjusted_pos = pos;

    if (params.wrap_edges == 1u) {
        // Wrap-around behavior
        adjusted_pos.x = pos.x % grid_size.x;
        adjusted_pos.y = pos.y % grid_size.y;
    } else {
        // Dead cells beyond borders
        if (pos.x >= grid_size.x || pos.y >= grid_size.y) {
            return 0u;
        }
    }

    return src_cells[get_cell_index(adjusted_pos)];
}

// Optimized neighbor counting using unrolled loop
fn count_neighbors(pos: vec2<u32>) -> u32 {
    var count = 0u;
    
    // Unrolled loop for better performance
    // Top row
    count += get_cell_state(vec2<u32>(pos.x - 1u, pos.y - 1u));
    count += get_cell_state(vec2<u32>(pos.x, pos.y - 1u));
    count += get_cell_state(vec2<u32>(pos.x + 1u, pos.y - 1u));
    
    // Middle row
    count += get_cell_state(vec2<u32>(pos.x - 1u, pos.y));
    count += get_cell_state(vec2<u32>(pos.x + 1u, pos.y));
    
    // Bottom row
    count += get_cell_state(vec2<u32>(pos.x - 1u, pos.y + 1u));
    count += get_cell_state(vec2<u32>(pos.x, pos.y + 1u));
    count += get_cell_state(vec2<u32>(pos.x + 1u, pos.y + 1u));
    
    return count;
}

// Function to determine next cell state based on rules
fn compute_next_state(current: u32, neighbors: u32) -> u32 {
    // Birth rule
    if (current == 0u && neighbors == params.birth_rule) {
        return 1u;
    }
    
    // Survival rule
    if (current == 1u && 
        neighbors >= params.survival_min && 
        neighbors <= params.survival_max) {
        return 1u;
    }
    
    // Death rule
    return 0u;
}

@compute @workgroup_size(WORKGROUP_SIZE, WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>, 
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    
    let pos = vec2<u32>(global_id.x, global_id.y);
    
    // Early return if outside grid bounds
    if (pos.x >= params.grid_size.x || pos.y >= params.grid_size.y) {
        return;
    }

    // Get current cell state and neighbor count
    let current_state = get_cell_state(pos);
    let neighbor_count = count_neighbors(pos);
    
    // Compute next state
    let next_state = compute_next_state(current_state, neighbor_count);
    
    // Write result
    dst_cells[get_cell_index(pos)] = next_state;
}