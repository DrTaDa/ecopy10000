import cv2
import numpy
import secrets
import glob
from scipy.spatial.distance import cdist
import time
from numba import jit
import matplotlib.pyplot as plt
import pathlib
import math

from pawn import Blob, Plant, WaterSource, Water, spawn_around, Wall

cell_size = 300 # Tied to vision_distance
world_size_global = 0 # Will be set in main()
grid_cols = 0
grid_rows = 0
spatial_grid = {} # The main grid dictionary: {(col, row): [entity1, entity2, ...]}

def initialize_grid(world_size):
    """Sets up grid dimensions based on world size."""
    global world_size_global, grid_cols, grid_rows, spatial_grid, cell_size
    world_size_global = world_size
    # Ensure cell_size is at least 1 to avoid division by zero
    if cell_size <= 0:
        cell_size = 1 # Or default to vision_distance if available
    grid_cols = math.ceil(world_size_global / cell_size)
    grid_rows = math.ceil(world_size_global / cell_size)
    spatial_grid = {}
    print(f"Initialized grid: {grid_cols}x{grid_rows} cells, cell size {cell_size}px")

def get_cell_coords(position):
    """Calculates the (col, row) grid cell for a given world position."""
    col = int(position[0] // cell_size)
    row = int(position[1] // cell_size)
    # Clamp coordinates to be within the grid boundaries
    col = max(0, min(col, grid_cols - 1))
    row = max(0, min(row, grid_rows - 1))
    return (col, row)

def add_entity_to_grid(entity):
    """Adds an entity to the spatial grid and sets its coords attribute."""
    if not hasattr(entity, 'position'): # Ensure it's a spatial entity
        return
    coords = get_cell_coords(entity.position)
    spatial_grid.setdefault(coords, []).append(entity)
    # Dynamically add the attribute to track the entity's cell
    entity._current_cell_coords = coords

def remove_entity_from_grid(entity):
    """Removes an entity from the spatial grid using its stored coords."""
    if not hasattr(entity, '_current_cell_coords') or entity._current_cell_coords is None:
        # If entity was never added or already removed, do nothing
        return

    coords = entity._current_cell_coords
    if coords in spatial_grid:
        try:
            spatial_grid[coords].remove(entity)
            # Optional: Clean up empty lists
            if not spatial_grid[coords]:
                del spatial_grid[coords]
        except ValueError:
            # Entity might have already been removed somehow, ignore
            pass
    entity._current_cell_coords = None # Mark as removed

def update_grid_for_entity(entity):
    """Checks if an entity moved cells and updates the grid accordingly."""
    if not hasattr(entity, 'position'):
        return
    
    new_coords = get_cell_coords(entity.position)
    old_coords = getattr(entity, '_current_cell_coords', None)

    if new_coords != old_coords:
        # Remove from old cell (if it was in one)
        if old_coords is not None and old_coords in spatial_grid:
             try:
                 spatial_grid[old_coords].remove(entity)
                 if not spatial_grid[old_coords]:
                     del spatial_grid[old_coords]
             except ValueError:
                 pass # Wasn't in the old cell list, maybe already removed

        # Add to new cell
        spatial_grid.setdefault(new_coords, []).append(entity)
        entity._current_cell_coords = new_coords


def draw_pawns(canvas, entities, blobs, oldest_blob):

    for e in entities:
        e.draw(canvas)
    for b in blobs:
        b.draw(canvas, draw_vision=b is oldest_blob)


def det(a, b):
    return a[0] * b[1] - a[1] * b[0]


def line_intersection(line1, line2):

    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

    div = det(xdiff, ydiff)
    # --- Safety check for parallel lines ---
    if abs(div) < 1e-9: # Lines are parallel or collinear
        return None # Or handle appropriately (e.g., raise error, return sentinel)

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


@jit(nopython=True, fastmath=True)
def jit_find_closest_index(distances_to_others, distances_to_ray_end_others, rays_dx, rays_dy, diff_distances_others, out_ray_others):
    """
    Optimized Numba function to find close entities along a ray.
    Note: Inputs are now distances/diffs relative to *other* nearby entities (excluding self).
    """
    min_distance_to_end = numpy.minimum(distances_to_others, distances_to_ray_end_others)
    distance_to_seg = numpy.abs((rays_dy * diff_distances_others[:, 0]) - (rays_dx * diff_distances_others[:, 1])) / 300.0

    shortest_distances_to_ray = numpy.where(out_ray_others, min_distance_to_end, distance_to_seg)
    # Find indices where distance is small enough
    # NOTE: The returned indices are relative to the input arrays (distances_to_others, etc.)
    return (shortest_distances_to_ray < 12).nonzero()[0]


def compute_vision(blobs, vision_distance, world_size):
    """
    Batch processing of ray casting and vision using Spatial Grid.
    WARNING: HARD CODED LENGTH AND DISTANCES (partially mitigated by grid)
    """
    if not blobs: # Skip if no blobs
        return

    # Pre-calculate vision angles relative to blob forward direction
    # Angle arrays are now calculated once if they are static properties of Blob
    # Assuming all blobs have the same vision setup:
    blob_vision_angles = blobs[0].vision_angles # Get from first blob
    n_vision_rays = blobs[0].n_vision_rays

    for i, b in enumerate(blobs):

        # 1. Get nearby entities from the spatial grid
        nearby_entities = []
        blob_coords = getattr(b, '_current_cell_coords', None)
        if blob_coords is None: # Should not happen if grid updates are correct
            blob_coords = get_cell_coords(b.position) # Fallback
            b._current_cell_coords = blob_coords # Try to fix it

        blob_col, blob_row = blob_coords
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                check_col = blob_col + dc
                check_row = blob_row + dr
                cell_key = (check_col, check_row)
                # Use .get() to safely handle empty/non-existent cells bordering the world
                entities_in_cell = spatial_grid.get(cell_key, [])
                nearby_entities.extend(entities_in_cell)

        # Filter out the blob itself from its nearby list
        nearby_entities_others = [e for e in nearby_entities if e is not b]

        # Reset vision for the current blob
        b.closest_collision_per_ray = [vision_distance] * n_vision_rays # Default to max distance
        b.closest_object_per_ray = [None] * n_vision_rays

        if not nearby_entities_others:
            # No other entities nearby, only check walls
            positions_others = numpy.empty((0, 2)) # Empty array
        else:
            # Create position array for nearby entities (excluding self)
            positions_others = numpy.asarray([e.position for e in nearby_entities_others])

        # 2. Calculate ray endpoints for the current blob
        angle_rays = blob_vision_angles + b.direction # Absolute world angles for rays
        rays_dx = vision_distance * numpy.cos(angle_rays)
        rays_dy = vision_distance * numpy.sin(angle_rays)
        ray_end_points_x = b.position[0] + rays_dx
        ray_end_points_y = b.position[1] + rays_dy
        ray_end = numpy.stack([ray_end_points_x, ray_end_points_y], axis=-1) # Shape (n_rays, 2)

        # Only perform distance calculations if there are nearby entities
        if positions_others.shape[0] > 0:
            # 3. Calculate distances between blob and nearby entities
            # distances_blob_to_others: shape (n_nearby,)
            distances_blob_to_others = cdist(b.position.reshape(1, 2), positions_others)[0]

            # 4. Calculate distances between ray endpoints and nearby entities
            # distances_ray_end_to_others: shape (n_rays, n_nearby)
            distances_ray_end_to_others = cdist(ray_end, positions_others)

            # 5. Vectorized checks for entities near rays
            # diff_distances_others: shape (n_nearby, 2), vector from blob to others
            diff_distances_others = positions_others - b.position
            # ray_end_offset: shape (n_rays, 2), vector from blob to ray end
            ray_end_offset = ray_end - b.position

            # Loop through each ray for the current blob
            for r in range(n_vision_rays):
                # dets: shape (n_nearby,), dot product check projection
                # Using ray_end_offset[r] specifically for this ray
                dets = numpy.dot(ray_end_offset[r], diff_distances_others.T)

                # out_ray_others: boolean array (n_nearby,), True if entity is behind or too far along ray extension
                # Vision distance squared = 300*300 = 90000
                out_ray_others = numpy.logical_or(dets < 0, dets > vision_distance**2)

                # Find indices of nearby entities that are close to this specific ray 'r'
                close_indices_relative = jit_find_closest_index(
                    distances_blob_to_others,           # Dist blob to others
                    distances_ray_end_to_others[r],     # Dist ray 'r' end to others
                    rays_dx[r],                         # Ray 'r' x component
                    rays_dy[r],                         # Ray 'r' y component
                    diff_distances_others,              # Vector blob to others
                    out_ray_others                      # Check if entity is roughly 'in front' of ray
                )

                # If any entities are close to the ray
                if len(close_indices_relative) > 0:
                    # Find the closest one among them based on actual distance to the blob
                    # argsort returns indices to sort distances_blob_to_others[close_indices_relative]
                    # We take the first one [0] which corresponds to the minimum distance
                    closest_idx_among_close = numpy.argsort(distances_blob_to_others[close_indices_relative])[0]
                    # Get the index in the original nearby_entities_others list
                    original_nearby_index = close_indices_relative[closest_idx_among_close]

                    # Store the collision distance and the entity object
                    b.closest_collision_per_ray[r] = distances_blob_to_others[original_nearby_index]
                    b.closest_object_per_ray[r] = nearby_entities_others[original_nearby_index]

        # 6. Check for collisions with walls *only if* no closer entity was found for that ray
        for r in range(n_vision_rays):
            if b.closest_object_per_ray[r] is None: # Check wall only if ray is 'clear' so far
                closest_wall_collision = vision_distance # Start assuming no wall collision within range

                ray_start = b.position
                ray_end_point = ray_end[r] # Specific endpoint for this ray

                # Define wall lines (slightly simplified, assumes origin at 0,0)
                walls = [
                    [[0, 0], [world_size, 0]],             # Bottom wall
                    [[world_size, 0], [world_size, world_size]], # Right wall
                    [[world_size, world_size], [0, world_size]], # Top wall
                    [[0, world_size], [0, 0]]              # Left wall
                ]

                # Check intersection with each wall
                for wall in walls:
                    intersection_point = line_intersection([ray_start, ray_end_point], wall)

                    if intersection_point is not None:
                        # Check if intersection point is actually ON the wall segment AND the ray segment
                        ix, iy = intersection_point
                        # Check wall segment (allowing for small float inaccuracies)
                        wall_on_segment = False
                        if wall[0][0] == wall[1][0]: # Vertical wall
                           wall_on_segment = min(wall[0][1], wall[1][1]) - 1e-6 <= iy <= max(wall[0][1], wall[1][1]) + 1e-6
                        else: # Horizontal wall
                           wall_on_segment = min(wall[0][0], wall[1][0]) - 1e-6 <= ix <= max(wall[0][0], wall[1][0]) + 1e-6

                        # Check ray segment (intersection must be between start and end point of the vision ray)
                        # Dot product check: (intersection - start) dot (end - start) should be between 0 and |end-start|^2
                        vec_ray = ray_end_point - ray_start
                        vec_intersect = intersection_point - ray_start
                        dot_prod = numpy.dot(vec_intersect, vec_ray)
                        ray_on_segment = (0 - 1e-6 <= dot_prod <= numpy.dot(vec_ray, vec_ray) + 1e-6)

                        if wall_on_segment and ray_on_segment:
                            distance_collision = numpy.linalg.norm(intersection_point - ray_start) # Faster than manual sqrt
                            if distance_collision < closest_wall_collision:
                                closest_wall_collision = distance_collision

                # If a wall collision was found closer than vision_distance
                if closest_wall_collision < vision_distance:
                    b.closest_collision_per_ray[r] = closest_wall_collision
                    b.closest_object_per_ray[r] = Wall() # Assign Wall object

        # 7. Compute the final vision vector input for the brain (moved from Blob.do_something)
        # This should ideally be in Blob class, but placed here to keep Blob class unchanged
        current_vision_input = []
        for r in range(n_vision_rays):
            # Create vision vector: [distance_encoded, is_Blob, is_Plant, is_Water, is_Wall, can_mate, similarity, empty, empty]
            # Length is length_vision_vector = 9 as defined in original Blob init
            vision_vec = [0.] * b.length_vision_vector # Initialize vector

            # Use stored closest object and distance
            distance = b.closest_collision_per_ray[r]
            obj = b.closest_object_per_ray[r]

            if obj is not None and distance < vision_distance:
                distance_encoded = 1.0 - (distance / vision_distance)
                vision_vec[0] = distance_encoded

                if isinstance(obj, Blob):
                    vision_vec[1] = 1.0
                    vision_vec[5] = 1.0 if obj.can_mate else 0.0 # Check can_mate
                    # Calculate similarity (careful with performance if called very often)
                    similarity = numpy.linalg.norm(obj.brain.flatten - b.brain.flatten)
                    vision_vec[6] = 1.0 - numpy.clip(similarity / 2.0, 0.0, 1.0) # Normalize similarity
                elif isinstance(obj, Plant):
                    vision_vec[2] = 1.0
                elif isinstance(obj, Water): # Assuming WaterSource is not seen directly, only Water
                    vision_vec[3] = 1.0
                elif isinstance(obj, Wall):
                    vision_vec[4] = 1.0
            # Else: obj is None or too far, vector remains all zeros

            current_vision_input.extend(vision_vec) # Append the vector for this ray

        # Store the computed vision input directly on the blob instance for use in do_something
        # This assumes Blob class uses self.current_vision for its input later
        b.current_vision = current_vision_input


def plot_population_history(population_blobs, population_plants, population_waters, t, world_size):

    canvas_hist = numpy.full((200, world_size, 3), 0, numpy.uint8)

    if t < 2:
        return canvas_hist

    # Prevent division by zero if all populations are zero
    max_pop = max(max(population_blobs) if population_blobs else 1,
                  max(population_plants) if population_plants else 1,
                  max(population_waters) if population_waters else 1)
    if max_pop == 0: max_pop = 1 # Avoid division by zero

    base_color = [150, 150, 150]
    populations_data = [population_waters, population_plants, population_blobs]

    for i, pop in enumerate(populations_data):
        if not pop: continue # Skip if population list is empty
        color = base_color[:]
        color[i] = 255 # Highlight the current population type
        # Ensure history fits the canvas width
        tmp_hist = pop[-world_size:] if len(pop) > world_size else pop
        # Normalize points to canvas height (0-199)
        points = []
        for step, count in enumerate(tmp_hist):
            x = step
            y = 199 - int(199. * count / max_pop) # Invert Y-axis for drawing
            points.append((x,y))

        # Draw lines if there's more than one point
        if len(points) > 1:
            pts = numpy.array(points, numpy.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(canvas_hist, [pts], isClosed=False, color=color, thickness=2) # Use polylines for efficiency

    return canvas_hist


def print_summary(blobs, population_plants, population_waters, t, oldest_blob, targets, to_add):

    print()
    print()
    print(f"############ Timestep {t} ############")
    print()
    print(f"Population blob: {len(blobs)}")
    # Ensure population lists are not empty before accessing [-1]
    print(f"Plants: {population_plants[-1] if population_plants else 0}")
    print(f"Waters: {population_waters[-1] if population_waters else 0}")
    print(f"Plant eaten: {len([t for t in targets if isinstance(targets[t][0], Plant)])}")
    print(f"Water drunk: {len([t for t in targets if isinstance(targets[t][0], Water)])}")

    if blobs:
        print(f"Max hunger: {max([b.hunger for b in blobs])}")
        print(f"Max thirst: {max([b.thirst for b in blobs])}")
        print(f"New blobs: {len([b for b in to_add if b is not None])}")

    if oldest_blob:
        print()
        print(oldest_blob.name)
        print(f"Age: {oldest_blob.age}")
        print(f"Action: {getattr(oldest_blob, 'action', 'N/A')}") # Use getattr for safety
        print(f"Movement: {getattr(oldest_blob, 'movement', 'N/A')}")
        print("Health: {:.2f} / {}".format(oldest_blob.health, oldest_blob.health_max))
        print("Hunger: {:.2f} / {}".format(oldest_blob.hunger, oldest_blob.hunger_max))
        print("Fat: {:.2f} / {}".format(oldest_blob.fat, oldest_blob.fat_max))
        print("Thirst: {:.2f} / {}".format(oldest_blob.thirst, oldest_blob.thirst_max))
        print(f"Generation: {oldest_blob.generation}")
        print(f"Is mature: {oldest_blob.mature}")
        print(f"Can mate: {oldest_blob.can_mate}")
        if oldest_blob.parents:
            print(f"Parents: {oldest_blob.parents}")
        if oldest_blob.partners:
            print(f"Partners: {oldest_blob.partners}")
        if oldest_blob.offsprings:
            print(f"Offsprings: {oldest_blob.offsprings}")


def filter_spawned_entity(new_entities, entities, min_distance=48):
    """Filter out the new entities that might spawn too close to already existing entities."""
    # This function could also benefit from the spatial grid for faster proximity checks
    # if it becomes a bottleneck with many spawns.

    if not new_entities: # No new entities to filter
        return []
    if not entities: # No existing entities to check against
        return new_entities

    positions_new_entities = numpy.asarray([ne.position for ne in new_entities])
    positions_others = numpy.asarray([w.position for w in entities])

    distances = cdist(positions_new_entities, positions_others)

    # Keep only entities where the minimum distance to any existing entity is > min_distance
    return [ne for i, ne in enumerate(new_entities) if distances[i].min() > min_distance]


def main(world_size, t_spawn_blobs, n_water_source, n_starting_blob, vision_distance):

    # --- Initialize Spatial Grid ---
    global cell_size # Allow modification if needed
    cell_size = vision_distance # Link cell size to vision distance
    initialize_grid(world_size)
    # ---

    blobs = []
    water_sources = [WaterSource(world_size) for i in range(n_water_source)]
    # Initial entities (plants)
    entities = [Plant(world_size) for i in range(int(0.6 * n_starting_blob))]

    # --- Populate Initial Grid ---
    for ws in water_sources: add_entity_to_grid(ws) # Water sources might not move, but good practice
    for e in entities: add_entity_to_grid(e)
    # Blobs will be added when spawned
    # ---

    population_blobs = []
    population_plants = []
    population_waters = []
    gen_tree = [] # Unused?

    # Main simulation loop
    for t in range(100000):

        # --- Spawn initial blobs ---
        if t == t_spawn_blobs:
            newly_spawned_blobs = [Blob(world_size, parents=None) for i in range(n_starting_blob)]
            for b in newly_spawned_blobs:
                add_entity_to_grid(b) # Add new blobs to grid
            blobs.extend(newly_spawned_blobs)
            print(f"Spawned {n_starting_blob} blobs at t={t}")

        # --- Save experienced blobs ---
        #for i, b in enumerate(blobs):
        #   if not b.saved and len(b.offsprings) >= 9: # Threshold check
        #        try:
        #            pathlib.Path("./models/").mkdir(parents=True, exist_ok=True) # Ensure dir exists
        #            b.brain.save(b.name, t)
        #            blobs[i].saved = True
        #            # print(f"Saved brain for blob {b.name}") # Optional log
        #        except Exception as e:
        #            print(f"Error saving blob {b.name}: {e}")

        # --- Compute vision for all blobs using the spatial grid ---
        if blobs:
            compute_vision(blobs, vision_distance, world_size) # Uses the optimized function

        # --- Record Population Stats ---
        population_blobs.append(len(blobs))
        population_plants.append(len([1 for e in entities if isinstance(e, Plant)]))
        population_waters.append(len([1 for e in entities if isinstance(e, Water)]))

        # --- Reporting and Visualization ---
        oldest_blob = None
        if blobs:
            # Find blob with most offsprings
            oldest_blob = max(blobs, key=lambda b: len(b.offsprings))

        if t > t_spawn_blobs: # Only print after blobs are spawned
            # Prepare `to_add` temporarily for printing summary, it's reset later
            _temp_to_add_for_print = [] # We don't have to_add yet at this stage
            _temp_targets_for_print = {} # We don't have targets yet at this stage
            if t % 10 == 0:
                print_summary(blobs, population_plants, population_waters, t, oldest_blob,
                              _temp_targets_for_print, _temp_to_add_for_print)

        # Draw and display canvas
        if t > t_spawn_blobs:
            canvas = numpy.full((world_size, world_size, 3), 100, numpy.uint8)
            # Combine entities and water sources for drawing non-blobs
            drawable_entities = entities + water_sources
            draw_pawns(canvas, drawable_entities, blobs, oldest_blob)
            canvas_hist = plot_population_history(population_blobs, population_plants, population_waters, t, world_size)
            final_canvas = numpy.concatenate([canvas, canvas_hist], axis=0)
            # Resize for display
            target_height = 1000 # Adjust target display height if needed
            scale = target_height / final_canvas.shape[0]
            resized = cv2.resize(
                final_canvas,
                (int(final_canvas.shape[1] * scale), target_height),
                interpolation = cv2.INTER_NEAREST # Use INTER_NEAREST for speed and pixelated look
            )
            cv2.imshow('Ecopy Simulation', resized) # Window title
            # cv2.imwrite(f"./frames/frame_{t:06d}.jpg", resized)

            key = cv2.waitKey(1)

            if key == 27: # Allow exit with ESC key
                print("ESC pressed, exiting simulation.")
                break

        # --- Blob Actions and Interactions ---
        to_add_blobs = [] # Blobs born this turn
        targets = {}      # Entities targeted for consumption/interaction {target_name: [target_entity, acting_blob]}

        # Reset acted flag
        for b in blobs:
            b.acted = False

        # Let blobs perform actions
        for b in blobs:
            if b.acted: continue # Skip if already acted (e.g., mated)

            # Vision input is now pre-calculated and stored in b.current_vision by compute_vision
            # Blob.do_something() needs to use b.current_vision directly
            tmp_add, target = b.do_something() # Assumes do_something uses the precomputed vision

            if tmp_add is not None:
                to_add_blobs.append(tmp_add)

            # Store target if one exists
            if target:
                # Ensure target is not already targeted by another blob this turn?
                # Current logic allows multiple blobs targeting one entity. First one processed wins?
                # Or maybe the Blob.do_something handles proximity better now.
                if target.name not in targets:
                    targets[target.name] = [target, b] # Store target and the blob targeting it

        # --- Process Interactions (Consumption) ---
        consumed_target_names = set()
        for target_name in targets:
            target_entity = targets[target_name][0]
            acting_blob = targets[target_name][1]

            # Check if target still exists (wasn't consumed by another blob processed earlier)
            # And check if blob is still alive
            if target_entity.name not in consumed_target_names and not acting_blob.is_dead():
                 # Perform consumption
                 acting_blob.consume(target_entity)
                 consumed_target_names.add(target_entity.name)
                 # Mark target entity for removal later (don't remove from grid yet)

        # --- Blob Metabolism and Health Update ---
        for b in blobs:
            # Simplified metabolism logic, potentially happens within do_something now?
            # Let's assume do_something handles basic metabolism (hunger/thirst decrease)
            # We only need the thresholding after potential consumption
            b.threshold_metabolism() # Clamp values and convert excess hunger to fat

        # --- Update Entity Lists (Removal) ---
        # Remove consumed entities from main list and grid
        original_entity_count = len(entities)
        entities_to_keep = []
        for e in entities:
            if e.name in consumed_target_names:
                remove_entity_from_grid(e) # Remove from spatial grid
            else:
                entities_to_keep.append(e)
        entities = entities_to_keep

        # Remove dead blobs from main list and grid
        blobs_to_keep = []
        for b in blobs:
            if b.is_dead():
                remove_entity_from_grid(b) # Remove from spatial grid
                # print(f"Blob {b.name} died at age {b.age}") # Optional log
            else:
                blobs_to_keep.append(b)
        blobs = blobs_to_keep

        # --- Add Newly Born Blobs ---
        for new_b in to_add_blobs:
            add_entity_to_grid(new_b) # Add to spatial grid
            blobs.append(new_b)

        # --- Non-Blob Entity Actions (Spawning Plants/Water) ---
        newly_spawned_entities = []
        # Include water sources in the update loop if they spawn water
        entities_and_sources = entities + water_sources
        for e in entities_and_sources:
            spawned = e.do_something() # Plant spreading, WaterSource dripping
            if spawned is not None:
                # Check position validity? (do_something should handle it)
                newly_spawned_entities.append(spawned)
                # Don't add to grid immediately, filter first

        # --- Filter and Add New Entities ---
        if newly_spawned_entities:
            # Combine existing blobs and entities for filtering check
            all_existing_movable = blobs + entities # Water sources are static
            filtered_new_entities = filter_spawned_entity(newly_spawned_entities, all_existing_movable)
            for fe in filtered_new_entities:
                add_entity_to_grid(fe)
                entities.append(fe)

        # --- Update Grid for Moved Blobs ---
        # This needs to happen AFTER blobs have moved (in do_something) and BEFORE next compute_vision
        for b in blobs:
            update_grid_for_entity(b)
        # Note: Plants/Water usually don't move, so no update needed unless they do

        # --- Simulation End Condition ---
        # Check if population died out after initial spawn phase
        if t > t_spawn_blobs + 500 and not blobs:
            print(f"All blobs died out by timestep {t}. Extinction event. T_T")
            break

    cv2.destroyAllWindows()
    print("Simulation finished.")


if __name__ == "__main__":

    chunk_size = 300
    vision_distance = chunk_size

    n_chunks = 13
    world_size = n_chunks * chunk_size
    n_pixels = world_size * world_size

    n_water_sources_per_million_pixels = 1.5
    n_blobs_per_million_pixels = 120 # normal: 100, restart from models: 20

    n_water_source = numpy.max([int(round(n_water_sources_per_million_pixels * n_pixels / 1e6)), 1])
    n_starting_blob = int(round(n_blobs_per_million_pixels * n_pixels / 1e6))

    t_spawn_blobs = 2000
    main(world_size, t_spawn_blobs, n_water_source, n_starting_blob, vision_distance)
