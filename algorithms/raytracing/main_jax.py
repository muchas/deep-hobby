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


class Ray(NamedTuple):
    origin: jnp.array
    direction: jnp.array


def at(ray, t):
    return ray.origin + t * ray.direction


class ScatteredRay(NamedTuple):
    ray: Ray
    attenuation: jnp.array


class Intersection(NamedTuple):
    was_hit: bool
    ray_direction: jnp.array
    t: float
    hit_point: jnp.array
    normal: jnp.array
    material_index: int


class HitRecord(NamedTuple):
    intersection: Intersection
    scattered_ray: ScatteredRay

    @property
    def was_hit(self):
        return self.intersection.was_hit

    @property
    def ray_direction(self):
        return self.intersection.ray_direction

    @property
    def t(self):
        return self.intersection.t

    @property
    def hit_point(self):
        return self.intersection.hit_point

    @property
    def normal(self):
        return self.intersection.normal


EMPTY_INTERSECTION = Intersection(
    was_hit=jnp.array([False], dtype=jnp.int32),
    ray_direction=vector(0, 0, 0),
    t=jnp.inf,
    hit_point=vector(0, 0, 0),
    normal=vector(0, 0, 0),
    material_index=0,
)

MISSED_HIT_RECORD = HitRecord(
    intersection=EMPTY_INTERSECTION,
    scattered_ray=ScatteredRay(
        ray=Ray(origin=vector(0, 0, 0), direction=vector(0, 0, 0),),
        attenuation=vector(0, 0, 0),
    ),
)


class SceneObject:
    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Intersection:
        raise NotImplemented()


class Sphere(SceneObject):
    def __init__(self, origin: jnp.array, radius: float, material_index: int):
        self.origin = origin
        self.radius = radius
        self.material_index = material_index

    @partial(jit, static_argnums=(0,))
    def intersect(self, ray: Ray, t_min: float, t_max: float) -> Intersection:
        material_index = self.material_index
        sphere_origin = self.origin
        sphere_radius = self.radius

        diff_v = ray.origin - sphere_origin

        a = jnp.dot(ray.direction, ray.direction)
        b = jnp.dot(ray.direction, diff_v)
        c = jnp.dot(diff_v, diff_v) - self.radius ** 2

        discriminant = b ** 2 - a * c

        @jit
        def intersection(t):
            hitpoint = at(ray, t)
            outward_normal = (hitpoint - sphere_origin) / sphere_radius

            normal = lax.cond(
                jnp.dot(ray.direction, outward_normal) < 0,
                lambda: outward_normal,
                lambda: -outward_normal,
            )
            return Intersection(
                was_hit=jnp.array([True], dtype=jnp.int32),
                ray_direction=ray.direction,
                t=t,
                hit_point=hitpoint,
                normal=normal,
                material_index=material_index,
            )

        @jit
        def determine_hitpoint(delta):
            delta_sqrt = jnp.sqrt(delta)
            t1 = (-b + delta_sqrt) / a
            t2 = (-b - delta_sqrt) / a
            return lax.cond(
                (t_min < t1) & (t1 < t2),
                lambda: intersection(t1),
                lambda: lax.cond(
                    t_min < t2, lambda: intersection(t2), lambda: EMPTY_INTERSECTION
                ),
            )

        return lax.cond(
            discriminant > 0,
            lambda: determine_hitpoint(discriminant),
            lambda: EMPTY_INTERSECTION,
        )


class World:
    def __init__(self, scene_objects: List[SceneObject]):
        self.scene_objects = scene_objects

    @partial(jit, static_argnums=(0,))
    def find_closest_intersection(
        self, ray: Ray, t_min: float, t_max: float
    ) -> Intersection:
        intersections = [
            scene_object.intersect(ray, t_min, t_max)
            for scene_object in self.scene_objects
        ]
        min_t_object_index = jnp.argmin(
            jnp.stack([intersection.t for intersection in intersections])
        )
        conditions = [
            min_t_object_index == obj_idx for obj_idx in range(len(intersections))
        ]
        return tree_map(lambda *x: jnp.select(conditions, x), *intersections)


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


class Lambertian(NamedTuple):
    albedo: jnp.array

    # @partial(jit, static_argnums=(0,))
    def scatter(self, prgn_key, hit_point, normal, _ray_direction) -> ScatteredRay:
        random_unit_vector = unit(
            random.uniform(prgn_key, (3,), minval=-1.0, maxval=1.0)
        )
        scattered_ray_direction = hit_point + normal + random_unit_vector

        return ScatteredRay(
            ray=Ray(hit_point, scattered_ray_direction), attenuation=self.albedo
        )


class Metal(NamedTuple):
    albedo: jnp.array
    fuzziness: float = 0.0

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


# @partial(jit, static_argnames=["materials"])
def scatter_with_material_type(
    prng_key, intersection: Intersection, materials
) -> ScatteredRay:
    conditions = [
        intersection.material_index == material_index
        for material_index in range(len(materials))
    ]
    material = tree_map(lambda *x: jnp.select(conditions, x), *materials)
    return material.scatter(
        prng_key,
        intersection.hit_point,
        intersection.normal,
        intersection.ray_direction,
    )


def scatter(
    prng_key, intersection: Intersection, lambertian_materials, metal_materials
) -> ScatteredRay:
    # TODO(smucha): improve disaptch extensibility to multiple material types
    return lax.cond(
        intersection.material_index < len(lambertian_materials),
        lambda: scatter_with_material_type(
            prng_key, intersection, lambertian_materials
        ),
        lambda: scatter_with_material_type(
            prng_key,
            intersection._replace(
                material_index=intersection.material_index - len(lambertian_materials)
            ),
            metal_materials,
        ),
    )


# it's important for this function to be jittable, so it could be executed on a target device as a single kernel
def compute_ray_color(
    ray_direction,
    seed,
    ray_origin,
    world,
    lambertian_materials,
    metal_materials,
    max_scatter_steps: int = 4,
) -> jnp.array:
    ray = Ray(ray_origin, ray_direction)

    prng_key = random.PRNGKey(seed[0])
    prng_keys = random.split(prng_key, num=max_scatter_steps + 1)

    intersection = world.find_closest_intersection(ray, 0.001, jnp.inf)
    scattered_ray = scatter(
        prng_keys[0], intersection, lambertian_materials, metal_materials
    )
    all_hit_records = [HitRecord(intersection, scattered_ray)]

    for scatter_step in range(1, max_scatter_steps):
        last_hit_record = all_hit_records[-1]
        intersection = lax.cond(
            jnp.all(last_hit_record.was_hit),
            lambda: world.find_closest_intersection(
                last_hit_record.scattered_ray.ray, 0.001, jnp.inf
            ),
            lambda: EMPTY_INTERSECTION,
        )
        scattered_ray = scatter(
            prng_keys[scatter_step], intersection, lambertian_materials, metal_materials
        )
        all_hit_records.append(HitRecord(intersection, scattered_ray))

    initial_color = lax.cond(
        jnp.all(all_hit_records[-1].was_hit),
        lambda: color(0, 0, 0),
        lambda: background_color(ray),
    )

    final_color = initial_color
    for hit_record in all_hit_records[::-1]:
        final_color = lax.cond(
            jnp.all(hit_record.was_hit),
            lambda c: hit_record.scattered_ray.attenuation * c,
            lambda c: c,
            final_color,
        )
    return final_color


def background_color(ray):
    t = (unit(at(ray, 1))[1] + 1) / 2
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


def generate_random_lambertian(prng_key):
    albedo = random.uniform(
        prng_key, shape=(3,), dtype=jnp.float32, minval=0.05, maxval=0.95
    )
    return Lambertian(albedo)


def generate_random_metal(prng_key):
    albedo_key, fuzziness_key = random.split(prng_key, num=2)

    albedo = random.uniform(
        albedo_key, shape=(3,), dtype=jnp.float32, minval=0.05, maxval=0.95
    )
    fuzziness = random.uniform(
        fuzziness_key, shape=(1,), dtype=jnp.float32, minval=0.0, maxval=0.3
    )

    return Metal(albedo, fuzziness=fuzziness)


def generate_random_sphere(prng_key, material_idx):
    x_key, z_key, size_key, material_key = random.split(prng_key, num=4)

    x = random.uniform(x_key, minval=-5, maxval=5)
    z = random.uniform(z_key, minval=0.0, maxval=2.0)
    size = random.uniform(size_key, minval=0.1, maxval=0.25)
    # material_idx = random.choice(material_key, jnp.array(list(range(len(materials))), dtype=jnp.int32))

    return Sphere(vector(x, -0.3, -0.3 - z), size, material_idx)


def generate_random_scene(
    prng_key,
    number_of_spheres: int,
    number_of_lambertian_materials: int,
    number_of_metal_materials: int,
):
    # split prng keys
    lambertian_prng_key, metal_prng_key, scene_prng_key = random.split(prng_key, num=3)

    lambertian_prng_keys = random.split(
        lambertian_prng_key, num=number_of_lambertian_materials
    )
    lambertian_materials = [
        generate_random_lambertian(key) for key in lambertian_prng_keys
    ]
    metal_prng_keys = random.split(metal_prng_key, num=number_of_metal_materials)
    metal_materials = [generate_random_metal(key) for key in metal_prng_keys]
    materials = lambertian_materials + metal_materials
    scene_prng_keys = random.split(scene_prng_key, num=number_of_spheres)
    scene = [
        generate_random_sphere(key, material_idx=(i % len(materials))) for i, key in enumerate(scene_prng_keys)
    ]
    return scene, lambertian_materials, metal_materials


def main():
    seed = 7
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

    scene, lambertian_materials, metal_materials = generate_random_scene(
        starting_key,
        number_of_spheres=15,
        number_of_lambertian_materials=1,
        number_of_metal_materials=6,
    )

    material_ground = Lambertian(vector(0.8, 0.8, 0.0))

    lambertian_materials.append(material_ground)
    scene.append(Sphere(vector(0.0, -100.5, -1.0), 100, len(lambertian_materials) - 1))

    materials = lambertian_materials + metal_materials

    world = World(scene)

    samples_per_pixel = 50
    batch_size = 10

    pixel_colors = jnp.zeros(
        shape=(batch_size, image_height, image_width, channels_num), dtype=jnp.float32
    )

    compute_ray_color_in_concrete_world = jit(
        partial(
            compute_ray_color,
            ray_origin=camera.origin,
            world=world,
            lambertian_materials=lambertian_materials,
            metal_materials=metal_materials,
        )
    )
    compute_ray_color_vectorized = vmap(vmap(vmap(compute_ray_color_in_concrete_world)))

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
