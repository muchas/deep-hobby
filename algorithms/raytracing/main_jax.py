import math
import random as pyrandom
import jax.numpy as jnp

from jax import jit, lax, vmap, random, devices, device_put, tree_map, xla_computation

from collections import namedtuple
from cached_property import cached_property
from functools import partial
from typing import Optional, List, Tuple, NamedTuple
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


def reflect(ray_direction: jnp.array, normal: jnp.array) -> jnp.array:
    return ray_direction - 2 * jnp.dot(normal, ray_direction) * normal


def refract(vector: jnp.array, normal: jnp.array, refraction_ratio: float) -> jnp.array:
    cos_theta = min(jnp.dot(-vector, normal), 1.0)
    r_out_perp = refraction_ratio * (vector + cos_theta * normal)
    r_out_parallel = -jnp.sqrt(jnp.abs(1.0 - jnp.dot(r_out_perp, r_out_perp))) * normal
    return r_out_perp + r_out_parallel


class Ray:
    def __init__(self, origin: jnp.array, direction: jnp.array):
        self.origin = origin
        self.direction = direction

    def at(self, t: float) -> jnp.array:
        return self.origin + t * self.direction


class ScatteredRay:
    def __init__(self, ray: Ray, attenuation: jnp.array):
        self.ray = ray
        self.attenuation = attenuation


class Hittable:
    def hit(self, prng_key, ray: Ray, t_min: float, t_max: float) -> float:
        raise NotImplemented()


class HitRecordV2(NamedTuple):
    was_hit: bool
    ray_direction: jnp.array
    t: float
    hit_point: jnp.array
    normal: jnp.array
    scattered_ray_direction: jnp.array
    scattered_ray_attenuation: jnp.array


MISSED_HIT_RECORD = HitRecordV2(
    was_hit=jnp.array([False], dtype=jnp.int32),
    ray_direction=vector(0, 0, 0),
    t=jnp.inf,
    hit_point=vector(0, 0, 0),
    normal=vector(0, 0, 0),
    scattered_ray_direction=vector(0, 0, 0),
    scattered_ray_attenuation=vector(0, 0, 0),
)       


class SphereV2(Hittable):
    def __init__(self, origin: jnp.array, radius: float, material):
        self.origin = origin
        self.radius = radius
        self.material = material

    def hit(self, prng_key, ray: Ray, t_min: float, t_max: float) -> HitRecordV2:
        material = self.material
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
                lambda: -outward_normal,
            )

            scattered_ray = material.scatter(prng_key, hitpoint, normal, ray.direction)

            return HitRecordV2(
                was_hit=jnp.array([True], dtype=jnp.int32),
                ray_direction=ray.direction,
                t=t,
                hit_point=hitpoint,
                normal=normal,
                scattered_ray_direction=scattered_ray.ray.direction,
                scattered_ray_attenuation=scattered_ray.attenuation,
            )

        @jit
        def determine_hitpoint(delta):
            delta_sqrt = jnp.sqrt(delta)
            t1 = (-b + delta_sqrt) / a
            t2 = (-b - delta_sqrt) / a
            return lax.cond(
                (t_min < t1) & (t1 < t2),
                lambda: hit_record(t1),
                lambda: lax.cond(
                    t_min < t2, lambda: hit_record(t2), lambda: MISSED_HIT_RECORD
                ),
            )

        return lax.cond(
            discriminant > 0,
            lambda: determine_hitpoint(discriminant),
            lambda: MISSED_HIT_RECORD,
        )


class WorldV2:
    def __init__(self, objects: List[Hittable]):
        self.objects = objects

    def hit(self, prng_key, ray: Ray, t_min: float, t_max: float) -> HitRecordV2:
        hit_records = [hittable.hit(prng_key, ray, t_min, t_max) for hittable in self.objects]
        min_t_object_index = jnp.argmin(jnp.stack([hit_record.t for hit_record in hit_records]))

        conditions = []
        for obj_idx in range(len(hit_records)):
            conditions.append(min_t_object_index == obj_idx)

        return tree_map(lambda *x: jnp.select(conditions, x), *hit_records)


class Camera:
    def __init__(
        self,
        origin: jnp.array,
        image_size: Tuple[int, int],
        viewport_size: Tuple[int, int],
        focal_length: float = 1.0,
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
        self.__bottom_left_corner = (
            origin
            - self.__vertical / 2
            - self.__horizontal / 2
            - vector(0, 0, focal_length)
        )

        self.__vertical_indices = jnp.arange(self.__image_height).reshape(
            1, self.__image_height, 1
        )
        self.__horizontal_indices = jnp.arange(self.__image_width).reshape(
            1, 1, self.__image_width
        )

        self.__compute_ray_direction = vmap(vmap(vmap(compute_ray_direction)))

    def get_ray_directions(self, prng_key, batch_size: int) -> jnp.array:
        return self.__compute_ray_directions(prng_key, batch_size)

    def __compute_ray_directions(self, prng_key, batch_size):
        shape = (batch_size, self.__image_height, self.__image_width)

        zeros = jnp.zeros(shape, dtype=jnp.float32)

        key, subkey = random.split(prng_key)
        # TODO: compute relative unit to move that computation after relative_y is computed
        vertical_indices = self.__vertical_indices + random.uniform(
            key, shape=shape, dtype=jnp.float32, minval=0.0, maxval=1.0
        )
        relative_y = zeros.at[::].set(vertical_indices / (self.__image_height - 1))

        # TODO: compute relative unit to move that computation after relative_y is computed
        horizontal_indices = self.__horizontal_indices + random.uniform(
            subkey, shape=shape, dtype=jnp.float32, minval=0.0, maxval=1.0
        )
        relative_x = zeros.at[::].set(horizontal_indices / (self.__image_width - 1))

        ray_directions = self.__compute_ray_direction(relative_y, relative_x)
        return ray_directions + self.__bottom_left_corner


class Material:
    #  TODO(smucha): extract hit_point, normal and ray_direction into a separate primitive
    def scatter(self, prgn_key, hit_point, normal, ray_direction) -> ScatteredRay:
        raise NotImplemented()


class Lambertian(Material):
    def __init__(self, albedo):
        self.albedo = albedo

    def scatter(self, prgn_key, hit_point, normal, _ray_direction) -> ScatteredRay:
        random_unit_vector = unit(
            random.uniform(prgn_key, (3,), minval=-1.0, maxval=1.0)
        )
        scattered_ray_direction = hit_point + normal + random_unit_vector

        return ScatteredRay(
            ray=Ray(hit_point, scattered_ray_direction), attenuation=self.albedo
        )


class Metal(Material):
    def __init__(self, albedo, fuzziness: float = 0.0):
        self.albedo = albedo
        self.fuzziness = fuzziness

    def scatter(self, prgn_key, hit_point, normal, ray_direction) -> ScatteredRay:
        random_unit_vector = unit(
            random.uniform(prgn_key, (3,), minval=-1.0, maxval=1.0)
        )
        reflected_ray_direction = (
            reflect(ray_direction, normal) + self.fuzziness * random_unit_vector
        )
        return ScatteredRay(
            ray=Ray(hit_point, reflected_ray_direction), attenuation=self.albedo
        )


class Dielectric(Material):
    def __init__(self, index_of_refraction):
        self.index_of_refraction = index_of_refraction

    def scatter(self, prngn_key, hit_point, normal, ray_direction) -> ScatteredRay:
        color = vector(1.0, 1.0, 1.0)

        refraction_ratio = (
            (1.0 / self.index_of_refraction)
            if hit_record.front_face
            else self.index_of_refraction
        )

        unit_direction = unit(ray.direction)
        cos_theta = min(np.dot(-unit_direction, hit_record.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        cannot_refract = refraction_ratio * sin_theta > 1.0

        if (
            cannot_refract
            or self.__reflectance(cos_theta, refraction_ratio) > random.random()
        ):
            target_direction = reflect(unit_direction, hit_record.normal)
        else:
            target_direction = refract(
                unit_direction, hit_record.normal, refraction_ratio
            )

        return ScatteredRay(
            ray=Ray(hit_record.point, target_direction), attenuation=color
        )

    def __reflectance(self, cosine: float, refraction_ratio: float) -> float:
        # Schlick's approximation for reflectance
        r_0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
        r_0 = r_0 * r_0
        return r_0 + (1 - r_0) * ((1 - cosine) ** 5)


# it's important for this function to be jittable, so it could be executed on a target device as a single kernel
def compute_ray_color(
    ray_direction, seed, ray_origin, world, max_scatter_steps: int = 4
) -> jnp.array:
    ray = Ray(ray_origin, ray_direction)

    prng_key = random.PRNGKey(seed[0])
    prng_keys = random.split(prng_key, num=max_scatter_steps + 1)

    hit_record = world.hit(prng_key, ray, 0.001, jnp.inf)
    all_hit_records = [hit_record]

    for scatter_step in range(max_scatter_steps):
        new_hit_record = lax.cond(
            jnp.all(hit_record.was_hit),
            lambda: world.hit(prng_keys[scatter_step], Ray(hit_record.hit_point, hit_record.scattered_ray_direction), 0.001, jnp.inf),
            lambda: MISSED_HIT_RECORD,
        )
        all_hit_records.append(new_hit_record)
        hit_record = new_hit_record

    initial_color = lax.cond(
        jnp.all(all_hit_records[-1].was_hit),
        lambda: color(0, 0, 0),
        lambda: background_color(ray),
    )

    final_color = initial_color
    for hit_record in all_hit_records[::-1]:
        final_color = lax.cond(
            jnp.all(hit_record.was_hit),
            lambda c: hit_record.scattered_ray_attenuation * c,
            lambda c: c,
            final_color,
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

    output_lines = [
        "P3\n{width} {height} \n{max_color}\n".format(
            width=width, height=height, max_color=max_color
        )
    ]

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
    camera = Camera(
        origin=vector(0, 0, 0),
        image_size=(image_width, image_height),
        viewport_size=(viewport_width, viewport_height),
    )

    # starting_scene = [
    # 	Sphere(vector(0, 0, -1), 0.5),
    #    	Sphere(vector(0.0, -100.5, -1.0), 100)  # ground
    # ]

    material_ground = Lambertian(vector(0.8, 0.8, 0.0))
    material_center = Lambertian(vector(0.1, 0.2, 0.5))
    # material_left = Dielectric(1.5)
    material_right = Metal(vector(0.8, 0.6, 0.2), fuzziness=0.1)

    second_scene = [
        SphereV2(vector(-1, 0, -0.75), 0.2, material_center),
        SphereV2(vector(1, 0, -1), 0.5, material_center),
        SphereV2(vector(0, -0.2, -1.5), 0.25, material_right),
        SphereV2(vector(-0.6, -0.2, -1.5), 0.25, material_right),
        SphereV2(vector(0.0, -100.5, -1.0), 100, material_ground),  # ground
    ]

    world = WorldV2(second_scene)

    samples_per_pixel = 50
    batch_size = 10

    pixel_colors = jnp.zeros(
        shape=(batch_size, image_height, image_width, channels_num), dtype=jnp.float32
    )

    compute_ray_color_in_a_given_world = jit(
        partial(compute_ray_color, ray_origin=camera.origin, world=world)
    )
    compute_ray_color_vectorized = vmap(vmap(vmap(compute_ray_color_in_a_given_world)))

    batches_num = int(samples_per_pixel / batch_size) + math.ceil(
        samples_per_pixel % batch_size
    )
    batch_prng_keys = random.split(starting_key, num=batches_num)

    seeds = (
        random.uniform(starting_key, (batch_size, image_height, image_width, 1)) * 1000
    ).astype(int)

    for batch_index in tqdm(range(batches_num), desc="sampling rays"):
        key = batch_prng_keys[batch_index]
        ray_directions = camera.get_ray_directions(
            key, batch_size
        )  # B x H x W x DIRECTION_DIM (3)
        pixel_colors += compute_ray_color_vectorized(ray_directions, seeds)

    pixel_colors = pixel_colors.sum(axis=0) / batch_size / batches_num
    pixel_colors = jnp.sqrt(pixel_colors)  # gamma correction (gamma 2)

    print(write_ppm_image(pixel_colors))


def save_computation_graph():
    seed = 0
    aspect_ratio = 16 / 9
    image_width = 512
    image_height = int(image_width / aspect_ratio)

    viewport_height = 2
    viewport_width = viewport_height * aspect_ratio
    channels_num = 3

    starting_key = random.PRNGKey(seed)
    camera = Camera(
        origin=vector(0, 0, 0),
        image_size=(image_width, image_height),
        viewport_size=(viewport_width, viewport_height),
    )

    # starting_scene = [
    #   Sphere(vector(0, 0, -1), 0.5),
    #       Sphere(vector(0.0, -100.5, -1.0), 100)  # ground
    # ]

    second_scene = [
        Sphere(vector(-1, 0, -0.75), 0.2, Lambertian()),
        Sphere(vector(1, 0, -1), 0.5, Lambertian()),
        Sphere(vector(0, -0.2, -1.5), 0.25, Metal()),
        Sphere(vector(-0.6, -0.2, -1.5), 0.25, Metal()),
        Sphere(vector(0.0, -100.5, -1.0), 100, Lambertian()),  # ground
    ]

    world = World(second_scene)

    samples_per_pixel = 50
    batch_size = 10

    pixel_colors = jnp.zeros(
        shape=(batch_size, image_height, image_width, channels_num), dtype=jnp.float32
    )

    compute_ray_color_in_a_given_world = jit(
        partial(compute_ray_color, ray_origin=camera.origin, world=world)
    )
    compute_ray_color_vectorized = vmap(vmap(vmap(compute_ray_color_in_a_given_world)))

    batches_num = int(samples_per_pixel / batch_size) + math.ceil(
        samples_per_pixel % batch_size
    )
    batch_prng_keys = random.split(starting_key, num=batches_num)

    seeds = (
        random.uniform(starting_key, (batch_size, image_height, image_width, 1)) * 1000
    ).astype(int)

    key = batch_prng_keys[0]
    ray_directions = camera.get_ray_directions(
        key, batch_size
    )  # B x H x W x DIRECTION_DIM (3)

    computation = xla_computation(compute_ray_color_vectorized)(ray_directions, seeds)

    import pdb

    pdb.set_trace()

    with open("computation_graph.dot", "w") as f:
        f.write(computation.as_hlo_dot_graph())


main()
