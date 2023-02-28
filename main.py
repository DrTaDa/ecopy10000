import cv2
import numpy
import secrets
import glob
from scipy.spatial.distance import cdist
import time
from numba import jit
import matplotlib.pyplot as plt
import pathlib

from pawn import Blob, Plant, WaterSource, Water, spawn_around, Wall
from chunk import Chunk


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

    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div

    return x, y


@jit(nopython=True, fastmath=True)
def jit_find_closest_index(distances, distances_to_ray_end, rays_dx, rays_dy, diff_distances, out_ray):

    min_distance_to_end = numpy.minimum(distances, distances_to_ray_end)
    distance_to_seg = numpy.abs((rays_dy * diff_distances[:, 0]) - (rays_dx * diff_distances[:, 1])) / 300
    shortest_distances_to_ray = numpy.where(out_ray, min_distance_to_end, distance_to_seg)
    return (shortest_distances_to_ray < 12).nonzero()[0]


def compute_vision(blobs, entities, vision_distance, world_size):
    """Batch processing of ray casting and vision.

    WARNING: HARD CODED LENGTH AND DISTANCES"""

    entities_ =  blobs + entities
    positions = numpy.asarray([e.position for e in entities_])
    angle_rays = blobs[0].vision_angles[:, numpy.newaxis] + numpy.asarray([e.direction for e in blobs])

    distances = cdist(positions[:len(blobs)], positions)

    rays_dx = vision_distance * numpy.cos(angle_rays.T)
    rays_dy = vision_distance * numpy.sin(angle_rays.T)
    ray_end_offset = numpy.stack([rays_dx, rays_dy], axis=-1)
    ray_end_points_x = positions[:len(blobs), 0, numpy.newaxis] + rays_dx
    ray_end_points_y = positions[:len(blobs), 1, numpy.newaxis] + rays_dy
    ray_end = numpy.stack([ray_end_points_x, ray_end_points_y], axis=-1)

    for i, b in enumerate(blobs):

        diff_distances = positions - b.position
        dets = numpy.dot(ray_end_offset[i], diff_distances.T)
        out_ray = numpy.logical_or(dets < 0, dets > 90000)
        distances_to_ray_end = cdist(ray_end[i], positions)

        blobs[i].closest_collision_per_ray = [100000] * b.n_vision_rays
        blobs[i].closest_object_per_ray = [None] * b.n_vision_rays

        for r in range(b.n_vision_rays):

            close_indexes = jit_find_closest_index(
                distances[i], distances_to_ray_end[r], rays_dx[i, r], rays_dy[i, r], diff_distances, out_ray[r]
            )

            if len(close_indexes) > 1:
                j = numpy.argsort(distances[i, close_indexes])[1]
                blobs[i].closest_collision_per_ray[r] = distances[i, close_indexes[j]]
                blobs[i].closest_object_per_ray[r] = entities_[close_indexes[j]]

            # Check for collisions with walls
            if blobs[i].closest_object_per_ray[r] is None:

                closest_collision = 1000
                if ray_end_points_x[i][r] < 0:
                    line1 = [blobs[i].position, ray_end[i][r]]
                    line2 = [[0, 0], [0, world_size]]
                    point_collision = line_intersection(line1, line2)
                    distance_collision = ((blobs[i].position[0] - point_collision[0])**2 + (blobs[i].position[1] - point_collision[1])**2)**0.5
                    if distance_collision <= closest_collision:
                        closest_collision = distance_collision
                if ray_end_points_x[i][r] > world_size:
                    line1 = [blobs[i].position, ray_end[i][r]]
                    line2 = [[world_size, 0], [world_size, world_size]]
                    point_collision = line_intersection(line1, line2)
                    distance_collision = ((blobs[i].position[0] - point_collision[0])**2 + (blobs[i].position[1] - point_collision[1])**2)**0.5
                    if distance_collision <= closest_collision:
                        closest_collision = distance_collision
                if ray_end_points_y[i][r] < 0:
                    line1 = [blobs[i].position, ray_end[i][r]]
                    line2 = [[0, 0], [world_size, 0]]
                    point_collision = line_intersection(line1, line2)
                    distance_collision = ((blobs[i].position[0] - point_collision[0])**2 + (blobs[i].position[1] - point_collision[1])**2)**0.5
                    if distance_collision <= closest_collision:
                        closest_collision = distance_collision
                if ray_end_points_y[i][r] > world_size:
                    line1 = [blobs[i].position, ray_end[i][r]]
                    line2 = [[0, world_size], [world_size, world_size]]
                    point_collision = line_intersection(line1, line2)
                    distance_collision = ((blobs[i].position[0] - point_collision[0])**2 + (blobs[i].position[1] - point_collision[1])**2)**0.5
                    if distance_collision <= closest_collision:
                        closest_collision = distance_collision

                if closest_collision <= 300:
                    blobs[i].closest_collision_per_ray[r] = closest_collision
                    blobs[i].closest_object_per_ray[r] = Wall()


def plot_population_history(population_blobs, population_plants, population_waters, t, world_size):

    canvas_hist = numpy.full((200, world_size, 3), 0, numpy.uint8)

    if t < 2:
        return canvas_hist

    max_pop = max(max(population_blobs), max(population_plants), max(population_waters))
    base_color = [150, 150, 150]
    for i, pop in enumerate([population_waters, population_plants, population_blobs]):
        color = base_color[:]
        color[i] = 255
        tmp_hist = pop if t < world_size else pop[-world_size:]
        for i in range(0, len(tmp_hist) - 1, 1):
            start_point = (i, 200 - int((200. * tmp_hist[i] / max_pop)))
            end_point = (i + 1, 200 - int((200. * tmp_hist[i+1] / max_pop)))
            cv2.line(canvas_hist, start_point, end_point, color, 3)

    return canvas_hist


def print_summary(blobs, population_plants, population_waters, t, oldest_blob, targets, to_add):

    print()
    print()
    print(f"############ Timestep {t} ############")
    print()
    print(f"Population blob: {len(blobs)}")
    print(f"Plants: {population_plants[-1]}")
    print(f"Waters: {population_waters[-1]}")
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
        print(f"Action: {oldest_blob.action}")
        print(f"Movement: {oldest_blob.movement}")
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

    if len(entities) == 0:
        return new_entities

    elif new_entities:
        positions_new_entities = [ne.position for ne in new_entities]
        positions_others = [w.position for w in entities]
        distances = cdist(positions_new_entities, positions_others)
        return [ne for i, ne in enumerate(new_entities) if distances[i].min() > min_distance]

    else:
        return []


def main(world_size, n_chunks, chunk_size, t_spawn_blobs, t_recordings, length_recordings, n_water_source, n_starting_blob, vision_distance):

    blobs = []
    water_sources = [WaterSource(world_size) for i in range(n_water_source)]

    entities = [Plant(world_size) for i in range(int(0.6 * n_starting_blob))]

    population_blobs = []
    population_plants = []
    population_waters = []
    gen_tree = []

    to_add = []
    targets = {}

    currently_recording = False
    end_recording = 0

    for t in range(70000):

        if t == t_spawn_blobs:
            #blobs = []
            #paths = glob.glob("./models/*.npy")
            #for i in range(n_starting_blob):
            #    blobs.append(Blob(world_size, parents=None))
            #    if numpy.random.random() > 0.1:
            #        path = numpy.random.choice(paths)
            #        print(f"starting blob from {path}")
            #        blobs[-1].brain.load(path)
            blobs += [Blob(world_size, parents=None) for i in range(n_starting_blob)]

        for i, b in enumerate(blobs):
            if not b.saved and len(b.offsprings) >= 9:
                b.brain.save(b.name, t)
                blobs[i].saved = True

        # Compute vision
        if blobs:
            compute_vision(blobs, entities, vision_distance, world_size)

        # Draw population curve
        oldest_blob = None
        if blobs:
            oldest_blob = blobs[numpy.argmax([len(b.offsprings) for b in blobs])]

        population_blobs.append(len(blobs))
        population_plants.append(len([1 for e in entities if isinstance(e, Plant)]))
        population_waters.append(len([1 for e in entities if isinstance(e, Water)]))
        
        if t > t_spawn_blobs:
            print_summary(blobs, population_plants, population_waters, t, oldest_blob, targets, to_add)

        # Draw and display canvas
        if t > t_spawn_blobs and t % t_recordings == 0:
            currently_recording = True
            end_recording = t + length_recordings
        if t == end_recording:
            currently_recording = False

        # if currently_recording:
        if t > t_spawn_blobs and currently_recording:
            canvas = numpy.full((world_size, world_size, 3), 100, numpy.uint8)
            draw_pawns(canvas, entities, blobs, oldest_blob)
            canvas_hist = plot_population_history(population_blobs, population_plants, population_waters, t, world_size)
            final_canvas = numpy.concatenate([canvas, canvas_hist], axis=0)
            resized = cv2.resize(
                final_canvas,
                (int(final_canvas.shape[1]*1500/final_canvas.shape[0]),1500),
                interpolation = cv2.INTER_AREA
            )
            cv2.imshow('Ecopy 10000', resized)
            cv2.imwrite(f"./frames/{t}.jpg", resized)
            cv2.waitKey(1)

        # Actions
        to_add = []
        targets = {}

        for b in blobs:
            b.acted = False

        for b in blobs:

            tmp_add, target = b.do_something()
            to_add.append(tmp_add)

            # I store the target as the first element of the array which is strange
            if target:
                if target.name not in targets:
                    targets[target.name] = [target]
                targets[target.name].append(b)

        for target_name in targets:
            targets[target_name][1].consume(targets[target_name][0])
        
        for b in blobs:
            b.threshold_metabolism()

        # Update entity lists
        entities = [e for e in entities if e.name not in targets]
        blobs = [b for b in blobs if not b.is_dead()] + [b for b in to_add if b is not None]

        new_entities = []
        for e in entities + water_sources:
            tmp = e.do_something()
            if tmp is not None:
                new_entities.append(tmp)

        if new_entities:
            entities += filter_spawned_entity(new_entities, entities)

        if t > 5000 and not len(blobs):
            print("Old blobs dead ... T_T")
            exit()


def benchmark_mating_model(path, n_chunk, chunk_size, world_size, vision_distance, n_blobs=4):

    world_size = n_chunk * chunk_size
    blobs = [Blob(world_size, parents=None) for i in range(n_blobs)]
    for i in range(n_blobs):
        blobs[i].brain.load(path)

    entities = [Plant(world_size) for i in range(120)] + [Water(world_size) for i in range(120)]
    
    cloning = 0
    mating = 0
    
    for t in range(10000):

        compute_vision(blobs, entities, vision_distance, world_size)
        
        for i in range(len(blobs)):
            blobs[i].acted = False

        """canvas = numpy.full((world_size, world_size, 3), 100, numpy.uint8)
        draw_pawns(canvas, entities, blobs, blobs[0])
        cv2.imshow('Ecopy 10000', canvas)
        cv2.waitKey(30)"""
        
        targets = {}
        for b in blobs:

            tmp_add, target = b.do_something()
            if tmp_add is not None:
                if len(tmp_add.parents) == 2:
                    mating += 1
                else:
                    cloning += 1

            # I store the target as the first element of the array which is strange
            if target:
                if target.name not in targets:
                    targets[target.name] = [target]
                targets[target.name].append(b)

        for target_name in targets:
            targets[target_name][1].consume(targets[target_name][0])
        
        blobs = [b for b in blobs if not b.is_dead()]
        entities = [e for e in entities if e.name not in targets]

        if targets:
            new_entities = []
            for target_name in targets:
                if isinstance(targets[target_name][0], Water):
                    new_entities.append(Water(world_size=world_size))
                elif isinstance(targets[target_name][0], Plant):
                    new_entities.append(Plant(world_size=world_size))

            entities += filter_spawned_entity(new_entities, entities)
           
        if not blobs:
            break

    return t, cloning, mating


if __name__ == "__main__":

    chunk_size = 300
    vision_distance = chunk_size

    n_chunks = 21
    world_size = n_chunks * chunk_size
    n_pixels = world_size * world_size

    n_water_sources_per_million_pixels = 1.5
    n_blobs_per_million_pixels = 120 # normal: 100, restart from models: 20

    n_water_source = numpy.max([int(round(n_water_sources_per_million_pixels * n_pixels / 1e6)), 1])
    n_starting_blob = int(round(n_blobs_per_million_pixels * n_pixels / 1e6))

    t_spawn_blobs = 2000
    t_recordings = 70000
    length_recordings = 2000

    main(world_size, n_chunks, chunk_size, t_spawn_blobs, t_recordings, length_recordings, n_water_source, n_starting_blob, vision_distance)
