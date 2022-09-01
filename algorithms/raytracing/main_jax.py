import math
import random as pyrandom
import jax.numpy as jnp

from jax import jit, vmap, random, devices, device_put

from jax import lax

from functools import partial
from typing import Optional, List, Tuple
from tqdm import tqdm


# Questions:
# 1) How to represent individual operations with Jax? Should they be done assuming vectors along the dimension?
# - vmap() can be used to map over pixel positions (forget first 2 dimension)
# - so in principle you can think of a single ray at a time (in the ray-tracing code)
# - 

# 2) 
# 3)

CPU_DEVICE = devices("cpu")[0]


def unit(vector: jnp.array) -> jnp.array:
	return vector / length(vector)


def length(vector: jnp.array) -> float:
	return jnp.sqrt(jnp.dot(vector, vector))


def vector(x: float, y: float, z: float) -> jnp.array:
	return jnp.array([x, y, z], dtype=jnp.float32)


def color(r: float, g: float, b: float) -> jnp.array:
	return vector(r, g, b)


class Ray:

	def __init__(self, origin: jnp.array, direction: jnp.array):
		self.origin = origin
		self.direction = direction

	def at(self, t: float) -> jnp.array:
		return self.origin + t * self.direction


# dataclass
# todo: add ser/deser methods to keep things inside jax arrays
class HitRecord:

	def __init__(self, t: float, point: jnp.array):
		self.t = t
		self.point = point


class Hittable:

	def hit(self, ray: Ray, t_min: float, t_max: float) -> float:
		raise NotImplemented()


# TODO: implement
class Sphere(Hittable):

	def __init__(self, origin: jnp.array, radius: float):
		self.origin = origin
		self.radius = radius

	# 
	def hit(self, ray: Ray, t_min: float, t_max: float) -> float:
		sphere_origin = self.origin
		sphere_radius = self.radius

		diff_v = ray.origin - sphere_origin

		a = jnp.dot(ray.direction, ray.direction)
		b = jnp.dot(ray.direction, diff_v)
		c = jnp.dot(diff_v, diff_v) - self.radius ** 2

		discriminant = b ** 2 - a * c

		@jit
		def hit_record(t):
			hitpoint = ray.at(t)
			outward_normal = (hitpoint - sphere_origin) / sphere_radius

			normal = lax.cond(
				jnp.dot(ray.direction, outward_normal) < 0,
				lambda: outward_normal,
				lambda: -outward_normal
			)

			return jnp.concatenate((jnp.array((1.0, t)), hitpoint, normal))

		missed_hit_record = jnp.concatenate((jnp.array((0.0, jnp.inf)), jnp.zeros((6,))))

		@jit
		def determine_hitpoint(delta):
			delta_sqrt = jnp.sqrt(delta)
			t1 = (-b + delta_sqrt) / a
			t2 = (-b - delta_sqrt) / a
			return lax.cond(
				(t_min < t1) & (t1 < t2), 
				lambda: hit_record(t1), 
				lambda: lax.cond(
					t_min < t2,
					lambda: hit_record(t2),
					lambda: missed_hit_record
				))
		return lax.cond(
			discriminant > 0, 
			lambda: determine_hitpoint(discriminant), 
			lambda: missed_hit_record
		)


# TODO: implement
class World(Hittable):

	T_COLUMN_INDEX = 1

	def __init__(self, objects: List[Hittable]):
		self.objects = objects

	def hit(self, ray: Ray, t_min: float, t_max: float) -> float:
		hits = jnp.stack([hittable.hit(ray, t_min, t_max) for hittable in self.objects])
		min_t_object_index = jnp.argmin(hits[:, self.T_COLUMN_INDEX])
		return hits[min_t_object_index, :]


class Camera:

	def __init__(self, 
		origin: jnp.array, 
		image_size: Tuple[int, int],
		viewport_size: Tuple[int, int],
		focal_length: float = 1.0
	):
		@jit
		def compute_ray_direction(relative_y, relative_x):
			return relative_y * self.__vertical + relative_x * self.__horizontal

		image_width, image_height = image_size
		viewport_width, viewport_height = viewport_size

		self.origin = origin
		self.__image_width = image_width
		self.__image_height = image_height

		self.__vertical = vector(0, viewport_height, 0)
		self.__horizontal = vector(viewport_width, 0, 0)
		self.__bottom_left_corner = origin - self.__vertical / 2 - self.__horizontal / 2 - vector(0, 0, focal_length)

		self.__vertical_indices = jnp.arange(self.__image_height).reshape(1, self.__image_height, 1)
		self.__horizontal_indices = jnp.arange(self.__image_width).reshape(1, 1, self.__image_width)

		self.__compute_ray_direction = vmap(vmap(vmap(compute_ray_direction)))

	def get_ray_directions(self, prng_key, batch_size: int) -> jnp.array:
		return self.__compute_ray_directions(prng_key, batch_size)

	def __compute_ray_directions(self, prng_key, batch_size):
		shape = (batch_size, self.__image_height, self.__image_width)

		zeros = jnp.zeros(shape, dtype=jnp.float32)

		key, subkey = random.split(prng_key)
		# TODO: compute relative unit to move that computation after relative_y is computed
		vertical_indices = self.__vertical_indices + random.uniform(key, shape=shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
		relative_y = zeros.at[::].set(vertical_indices / (self.__image_height - 1))

		# TODO: compute relative unit to move that computation after relative_y is computed
		horizontal_indices = self.__horizontal_indices + random.uniform(subkey, shape=shape, dtype=jnp.float32, minval=0.0, maxval=1.0)
		relative_x = zeros.at[::].set(horizontal_indices / (self.__image_width - 1))

		ray_directions = self.__compute_ray_direction(relative_y, relative_x)
		return ray_directions + self.__bottom_left_corner


# it's important for this function to be jittable, so it could be executed on a target device as a single kernel
def compute_ray_color(ray_direction, seed, ray_origin, world, max_scatter_steps: int = 4) -> jnp.array:
	@jit
	def scatter(hit_record, key):
		hit_point = hit_record[2:5]
		normal = hit_record[5:8]
		
		random_unit_vector = unit(random.uniform(key, (3,), minval=-1.0, maxval=1.0))
		scattered_ray_direction = hit_point + normal + random_unit_vector

		scattered_ray = Ray(hit_point, scattered_ray_direction)
		return world.hit(scattered_ray, 0.001, jnp.inf)

	prng_key = random.PRNGKey(seed[0])
	ray = Ray(ray_origin, ray_direction)
	initial_hit_record = world.hit(ray, 0.001, jnp.inf)

	all_hit_records = jnp.zeros((max_scatter_steps + 1, 8))
	all_hit_records = all_hit_records.at[0].set(initial_hit_record)

	prng_keys = random.split(prng_key, num=max_scatter_steps + 1)

	def loop_body(input_tuple):
		scatter_step, hit_record, all_hit_records = input_tuple
		new_hit_record = lax.cond(
			hit_record[0] > 0,
			lambda: scatter(hit_record, prng_keys[scatter_step]),
			lambda: jnp.zeros((8,))
		)
		return scatter_step + 1, new_hit_record, all_hit_records.at[scatter_step + 1].set(new_hit_record)

	_last_step_index, last_hit_record, all_hit_records = lax.while_loop(
		lambda input_tuple: (input_tuple[0] < max_scatter_steps) & (input_tuple[1][0] > 0),
		loop_body,
		(0, initial_hit_record, all_hit_records)
	)

	initial_color = lax.cond(
		(all_hit_records[-1][0] > 0),
		lambda: color(0, 0, 0),
		lambda: background_color(ray)
	)

	final_color = initial_color
	for hit_record in all_hit_records[::-1]:
		final_color = lax.cond(
			hit_record[0] > 0,
			lambda c: c * 0.5,
			lambda c: c,
			final_color
		)
	return final_color


def background_color(ray):
	t = (unit(ray.at(1))[1] + 1) / 2
	return (1 - t) * color(0, 0, 1) + t * color(1, 1, 1)


def write_ppm_image(image: jnp.array, max_color: int = 255) -> str:
	height, width, _ = image.shape

	image *= 255.999
	image = image.astype(int)

	device_put(image, CPU_DEVICE)

	image = image.to_py()

	output_lines = ["P3\n{width} {height} \n{max_color}\n".format(width=width, height=height, max_color=max_color)]

	for j in tqdm(range(height - 1, -1, -1), desc="writing pixel values"):
		for i in range(0, width):
			r, g, b = image[j, i, :]
			output_lines.append("{r} {g} {b}\n".format(r=r, g=g, b=b))

	return "".join(output_lines)


def main():
	seed = 0
	aspect_ratio = 16 / 9
	image_width = 512
	image_height = int(image_width / aspect_ratio)

	viewport_height = 2
	viewport_width = viewport_height * aspect_ratio
	channels_num = 3

	starting_key = random.PRNGKey(seed)
	camera = Camera(origin=vector(0, 0, 0), image_size=(image_width, image_height), viewport_size=(viewport_width, viewport_height))

	# starting_scene = [
	# 	Sphere(vector(0, 0, -1), 0.5),
 #    	Sphere(vector(0.0, -100.5, -1.0), 100)  # ground
	# ]

	second_scene = [
		Sphere(vector(-1, 0, -0.75), 0.2),
		Sphere(vector(1, 0, -1), 0.5),
		Sphere(vector(0, -0.2, -1.5), 0.25),
    	Sphere(vector(0.0, -100.5, -1.0), 100)  # ground
	]

	world = World(second_scene)

	samples_per_pixel = 100
	batch_size = 25

	pixel_colors = jnp.zeros(shape=(batch_size, image_height, image_width, channels_num), dtype=jnp.float32)

	compute_ray_color_in_a_given_world = jit(partial(compute_ray_color, ray_origin=camera.origin, world=world))
	compute_ray_color_vectorized = vmap(vmap(vmap(compute_ray_color_in_a_given_world)))

	batches_num = int(samples_per_pixel / batch_size) + math.ceil(samples_per_pixel % batch_size)
	batch_prng_keys = random.split(starting_key, num=batches_num)

	seeds = (random.uniform(starting_key, (batch_size, image_height, image_width, 1)) * 1000).astype(int)

	for batch_index in tqdm(range(batches_num), desc="sampling rays"):
	  key = batch_prng_keys[batch_index]
	  ray_directions = camera.get_ray_directions(key, batch_size)  # B x H x W x DIRECTION_DIM (3)
	  pixel_colors += compute_ray_color_vectorized(ray_directions, seeds)

	pixel_colors = pixel_colors.sum(axis=0) / batch_size / batches_num
	pixel_colors = jnp.sqrt(pixel_colors)  # gamma correction (gamma 2)

	print(write_ppm_image(pixel_colors))

main()
