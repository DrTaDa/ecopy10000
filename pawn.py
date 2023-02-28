import numpy
import secrets
import cv2
import pickle
from numba import jit
import time

from brain import Brain

TWO_PI = 2 * numpy.pi

def spawn_around(position, scale=150):
    angle = numpy.random.random() * TWO_PI
    distance = numpy.random.normal(loc=0, scale=scale)
    position = position + distance * numpy.asarray([numpy.cos(angle), numpy.sin(angle)])
    return position


def degrees_to_radians(angle):
    _ = angle * numpy.pi / 180.
    while _ < 0:
        _ += 2. * numpy.pi
    while _ > 2. * numpy.pi:
        _ -= 2. * numpy.pi
    return _


class Wall:

    color = (50, 50, 50)

    def __init__(self):
        pass


class Pawn:

    radius = 0
    color = (254, 254, 254)

    def __init__(self, position, direction, world_size):

        self.position = position
        self.direction = direction

        self.world_size = world_size
        self.name = secrets.token_urlsafe(10)

        self.clip_position()
        self.clip_direction()

    @property
    def x(self):
        return self.position[0]

    @property
    def y(self):
        return self.position[1]

    def clip_position(self):
        self.position = numpy.clip(self.position, 0, self.world_size)

    def clip_direction(self):
        if self.direction >= TWO_PI:
            self.direction -= TWO_PI
        elif self.direction < 0:
            self.direction += TWO_PI

    def draw(self, canvas):
        pass

    def do_something(self):
        pass


class Plant(Pawn):

    radius = 12
    color = (0, 170, 0)

    def __init__(self, world_size, position=None, parent=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size
        )

        self.color_contour = (int(self.color[0] + 50), int(self.color[1]) + 50, int(self.color[2] + 50))

        self.parent = parent

        if self.parent is None:
            self.age = numpy.random.randint(0, 2000)
            self.cooldown = numpy.random.randint(0, self.cooldown_spread)
        else:
            self.age = 0
            self.cooldown = 0

        self.spawn_max_distance = 150

    @property
    def cooldown_spread(self):
        return max(500 - int(self.age / 4), 10)

    def draw(self, canvas):
        cv2.circle(canvas, (int(self.x), int(self.y)), 12 - 1, self.color, -1)
        cv2.circle(canvas, (int(self.x), int(self.y)), 12 - 1, self.color_contour, 2)

    def do_something(self):

        new_plant = None

        if self.age > 50 and self.cooldown > self.cooldown_spread:
            position = spawn_around(self.position, scale=self.spawn_max_distance)
            if 0 < position[0] < self.world_size and 0 < position[1] < self.world_size:
                new_plant = Plant(self.world_size, position, self)
                self.cooldown = 0

        self.age += 1
        self.cooldown += 1

        return new_plant


class WaterSource(Pawn):

    radius = 12
    color = (150, 0, 0)

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.asarray([
                100 + numpy.random.random() * (world_size - 200),
                100 + numpy.random.random() * (world_size - 200)]
            )

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size
        )

        self.color_contour = (int(self.color[0] + 50), int(self.color[1]) + 50, int(self.color[2] + 50))

        self.parent = None

        self.spawn_max_distance = 150

    def do_something(self):

        new_water = None

        position = spawn_around(self.position, scale=self.spawn_max_distance)
        if 0 < position[0] < self.world_size and 0 < position[1] < self.world_size:
            new_water = Water(world_size=self.world_size, position=position)
            self.cooldown = 0

        return new_water

    def draw(self, canvas):

        cv2.circle(canvas, (int(self.x), int(self.y)), int(self.radius) - 1, self.color, -1)
        cv2.circle(canvas, (int(self.x), int(self.y)), int(self.radius) - 1, self.color_contour, 2)


class Water(Pawn):


    radius = 12
    color = (170, 20, 20)

    def __init__(self, world_size, position=None):

        if position is None:
            position = numpy.random.random(2) * world_size

        Pawn.__init__(
            self,
            position=position,
            direction=0.,
            world_size=world_size
        )

        self.color_contour = (int(self.color[0] + 50), int(self.color[1]) + 50, int(self.color[2] + 50))

        self.parent = None

    def draw(self, canvas):

        cv2.circle(canvas, (int(self.x), int(self.y)), int(self.radius) - 1, self.color, -1)
        cv2.circle(canvas, (int(self.x), int(self.y)), int(self.radius) - 1, self.color_contour, 2)


@jit(nopython=True, fastmath=True)
def jit_move_forward_clip(position, speed, direction, world_size):

    new_position = numpy.clip(
        position + speed * numpy.asarray([numpy.cos(direction), numpy.sin(direction)]),
        0,
        world_size
    )

    return new_position, numpy.abs(speed / 4.0)**2.5


class Blob(Pawn):

    radius = 12
    vision_distance = 300

    def __init__(self, world_size, parents=None, generation=0, model=None):

        Pawn.__init__(
            self,
            position=numpy.random.random(2) * world_size,
            direction=numpy.random.random() * TWO_PI,
            world_size=world_size
        )

        self.acted = False

        self.health = 50
        self.hunger = 50
        self.thirst = 50
        self.fat = 0
        self.health_max = 100
        self.thirst_max = 100
        self.hunger_max = 100
        self.fat_max = 100
        self.ratio_hunger_to_fat = 0.5
        self.threshold_for_fat_use = 50

        self.age = 0
        self.age_limit = 10000
        self.age_maturity = 50
        self.generation = generation
        self.parents = parents
        self.offsprings = []
        self.partners = []

        self.max_rotation = 0.25
        self.max_speed = 8.

        self.base_food_consumption = 0.18

        self.n_vision_rays = 13
        self.length_vision_vector = 9
        self.vision_angles = numpy.asarray([-90, -75, -60, -45, -30, -15, 0, 15, 30, 45, 60, 75, 90])
        self.vision_angles = self.vision_angles * numpy.pi / 180
        self.current_vision = [0] * self.n_vision_rays * self.length_vision_vector
        self.closest_collision_per_ray = [self.vision_distance] * self.n_vision_rays
        self.closest_object_per_ray = [None] * self.n_vision_rays

        self.n_actions = 1
        self.n_other_inputs = 5
        self.n_memory_neuron = 12
        self.memory = [0.] * self.n_memory_neuron
        self.action = 0
        self.movement = [0., 0.]
        self.n_hidden = 20

        self.brain = Brain(
            n_vision_rays=self.n_vision_rays,
            length_vision_vector=self.length_vision_vector,
            n_other_inputs=self.n_other_inputs,
            n_hidden=self.n_hidden,
            n_memory=self.n_memory_neuron,
            n_actions=self.n_actions,
            model=model
        )

        self.saved = False

    @property
    def mature(self):
        return self.age > self.age_maturity

    @property
    def can_mate(self):
        return self.mature and self.hunger > 40 and self.thirst > 40

    @property
    def can_clone(self):
        return self.mature and self.hunger > 50 and self.fat > 50 and self.thirst > 50

    @property
    def color(self):
        return (180, 180, 180)

    def draw(self, canvas, draw_vision=False):

        start_point = (int(self.x), int(self.y))
        movement_direction = [numpy.cos(self.direction), numpy.sin(self.direction)]
        end_point = (
            int(self.x + ((self.radius + 3) * movement_direction[0])),
            int(self.y + ((self.radius + 3) * movement_direction[1]))
        )

        cv2.line(canvas, start_point, end_point, self.color, 9)
        cv2.circle(canvas, start_point, int(self.radius), self.color, -1)
 
        if self.action == 0:
            cv2.circle(canvas, start_point, 5, (250, 50, 50), -1)
        elif self.action == 1:
            cv2.circle(canvas, start_point, 5, (50, 50, 250), -1)
        
        if draw_vision:
            for i, d_angle in enumerate(self.vision_angles):
                angle_ray = self.direction + d_angle
                ray_direction = [numpy.cos(angle_ray), numpy.sin(angle_ray)]

                length = self.closest_collision_per_ray[i]
                if self.closest_collision_per_ray[i] > self.vision_distance:
                    length = self.vision_distance

                end_point = (
                    int(self.x + (length * ray_direction[0])),
                    int(self.y + (length * ray_direction[1]))
                )

                if self.closest_object_per_ray[i]:
                    if isinstance(self.closest_object_per_ray[i], Wall):
                        cv2.line(canvas, start_point, end_point, (0, 0, 255), 1)
                    else:
                        cv2.line(canvas, start_point, end_point, (0, 255, 255), 1)
                        end_point2 = (int(self.closest_object_per_ray[i].x), int(self.closest_object_per_ray[i].y))
                        cv2.line(canvas, end_point, end_point2, (0, 255, 255), 1)
                else:
                    cv2.line(canvas, start_point, end_point, (255, 255, 255), 1)

            perc_health = int((self.health / self.health_max) * 60)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 19), (start_point[0] + 30, start_point[1] + 19), (0, 0, 0), 3)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 19), (start_point[0] - 30 + perc_health, start_point[1] + 19), (70, 70, 240), 10)

            perc_hunger = int((self.hunger / self.hunger_max) * 60)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 29), (start_point[0] + 30, start_point[1] + 29), (0, 0, 0), 3)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 29), (start_point[0] - 30 + perc_hunger, start_point[1] + 29), (70, 240, 70), 10)

            perc_fat = int((self.fat / self.fat_max) * 60)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 39), (start_point[0] + 30, start_point[1] + 39), (0, 0, 0), 3)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 39), (start_point[0] - 30 + perc_fat, start_point[1] + 39), (50, 180, 180), 10)

            perc_thirst = int((self.thirst / self.thirst_max) * 60)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 49), (start_point[0] + 30, start_point[1] + 49), (0, 0, 0), 3)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 49), (start_point[0] - 30 + perc_thirst, start_point[1] + 49), (240, 70, 70), 10)

            perc_age = int((self.age / self.age_limit) * 60)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 59), (start_point[0] + 30, start_point[1] + 59), (0, 0, 0), 3)
            cv2.line(canvas, (start_point[0] - 30, start_point[1] + 59), (start_point[0] - 30 + perc_age, start_point[1] + 59), (190, 190, 190), 10)

    def compute_vision(self):

        self.current_vision = []
        for r in range(self.n_vision_rays):

            # Create vision vector: [distance, Blob, Plant, Water, Wall, can_mate, similarity]
            if self.closest_collision_per_ray[r] < self.vision_distance:
                distance_encoded = 1 - (self.closest_collision_per_ray[r] / self.vision_distance)
            else:
                distance_encoded = 0

            if self.closest_object_per_ray[r] is None:
                self.current_vision = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            elif isinstance(self.closest_object_per_ray[r], Blob):
                similarity = numpy.linalg.norm(self.closest_object_per_ray[r].brain.flatten - self.brain.flatten)
                similarity = 1. - numpy.clip(similarity / 2, 0, 1)
                self.current_vision = [distance_encoded, 1, 0, 0, 0, self.closest_object_per_ray[r].can_mate, similarity]
            elif isinstance(self.closest_object_per_ray[r], Plant):
                self.current_vision = [distance_encoded, 0, 1, 0, 0, 0, 0]
            elif isinstance(self.closest_object_per_ray[r], Water):
                self.current_vision = [distance_encoded, 0, 0, 1, 0, 0, 0]
            elif isinstance(self.closest_object_per_ray[r], Wall):
                self.current_vision = [distance_encoded, 0, 0, 0, 1, 0, 0]

    def give_birth(self, partner):

        new_b = Blob(
            world_size=self.world_size,
            parents=[self.name],
            generation=self.generation + 1
        )
        new_b.position = numpy.copy(self.position)

        self.offsprings.append(new_b.name)

        if partner is None:
            dfat = 0.7 * self.fat
            dthirst = 0.5 * self.thirst
            self.fat -= dfat
            self.thirst -= dthirst
            new_b.hunger = dfat
            new_b.thirst = dthirst
            new_b.brain.copy_weights_and_mutate_vertical(self.brain, None)

        else:
        
            new_b.parents.append(partner.name)
            self.partners.append(partner.name)
            partner.partners.append(self.name)
            partner.offsprings.append(new_b.name)

            dhunger1 = 0.7 * self.hunger
            dthirst1 = 0.7 * self.thirst
            dhunger2 = 0.7 * partner.hunger
            dthirst2 = 0.7 * partner.thirst
            self.hunger -= dhunger1
            self.thirst -= dthirst1
            partner.hunger -= dhunger2
            partner.thirst -= dthirst2
            new_b.hunger = dhunger1 + dhunger2
            new_b.thirst = dthirst1 + dthirst2
                
            new_b.brain.copy_weights_and_mutate_vertical(self.brain, partner.brain)

        return new_b

    def do_something(self):

        input_brain = numpy.concatenate(
                [
                    self.current_vision,
                    [1. - (self.health / self.health_max)],
                    [1. - (self.hunger / self.hunger_max)],
                    [1. - (self.thirst / self.thirst_max)],
                    [1. - (self.fat / self.fat_max)],
                    [self.can_mate],
                    #self.memory
                ]
        )

        output_test = self.brain.evaluate(input_brain)
        self.movement = output_test[0]
        self.action = output_test[1] # 0 mate or clone, 1 attack or consume
        #self.memory = output_test[2]

        new_b = None
        action_target = None

        if self.is_dead():
            self.acted = True
            return new_b, action_target

        self.direction += self.movement[1] * self.max_rotation
        
        speed = (0.5 * (1 + self.movement[0])) * self.max_speed
        self.position, food_consumption_ratio = jit_move_forward_clip(self.position, speed, self.direction, self.world_size)

        self.hunger -= (0.5 + food_consumption_ratio) * self.base_food_consumption
        self.thirst -= (0.5 + food_consumption_ratio) * self.base_food_consumption

        if self.fat and self.hunger <= 0:
            delta = numpy.min([self.fat, food_consumption_ratio])
            self.hunger += delta
            self.fat -= delta

        if self.hunger <= 0:
            self.health -= self.base_food_consumption
            self.hunger = 0
        if self.thirst <= 0:
            self.health -= self.base_food_consumption
            self.thirst = 0
        if self.hunger > 0 and self.thirst > 0:
            self.health += 2. * self.base_food_consumption
            if self.health > self.health_max:
                self.health = self.health_max

        for r in numpy.argsort(self.closest_collision_per_ray):
            if not self.acted:

                if self.can_clone and self.action == 0:
                    new_b = self.give_birth(None)
                    self.acted = True

                elif self.closest_object_per_ray[r] is not None and self.closest_collision_per_ray[r] < 12:
                    if isinstance(self.closest_object_per_ray[r], Plant) and self.action == 1:
                        action_target = self.closest_object_per_ray[r]
                        self.acted = True
                    elif isinstance(self.closest_object_per_ray[r], Water) and self.action == 1:
                        action_target = self.closest_object_per_ray[r]
                        self.acted = True
                    elif isinstance(self.closest_object_per_ray[r], Blob):
                        if self.action == 0:
                            if self.can_mate and self.closest_object_per_ray[r].can_mate and not self.closest_object_per_ray[r].acted:
                                new_b = self.give_birth(self.closest_object_per_ray[r])
                                self.acted = True
                                self.closest_object_per_ray[r].acted = True
                        elif self.action == 1:
                            pass

        self.clip_direction()

        self.age += 1

        return new_b, action_target

    def is_dead(self):

        if self.health <= 0 or self.age >= self.age_limit:
            return True

        return False

    def threshold_metabolism(self):
        if self.thirst > self.thirst_max:
            self.thirst = self.thirst_max
        elif self.thirst < 0:
            self.thirst = 0
        if self.hunger > self.hunger_max:
            delta = self.hunger - self.hunger_max
            self.fat += self.ratio_hunger_to_fat * delta
            if self.fat > self.fat_max:
                self.fat = self.fat_max
            self.hunger = self.hunger_max
        if self.hunger < 0:
            self.hunger = 0

    def consume(self, food):
        if isinstance(food, Plant):
            self.hunger += 35
            self.thirst += 1
        elif isinstance(food, Water):
            self.thirst = self.thirst_max
        #elif isinstance(food, Blob):
        #    self.fat +=  food.fat
        #    self.hunger += food.hunger
        #    self.thirst += food.thirst
        self.threshold_metabolism()
