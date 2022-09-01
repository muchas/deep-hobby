import random
import numpy as np
import logging

from typing import Tuple, Optional, List
from tqdm import tqdm


class PythonImage:
    def __init__(self, width: int, height: int):
        assert width > 1
        assert height > 1

        self.width = width
        self.height = height

    # returns (r, g, b)
    def get_colour(self, x: int, y: int) -> Tuple[int, int, int]:
        assert 0 <= x <= self.width
        assert 0 <= y <= self.height

        r = x / (self.width - 1)
        g = y / (self.height - 1)
        b = 0.1

        return ((int)(255.999 * r), (int)(255.999 * g), (int)(255.999 * b))



class NumpyImage:

    def __init__(self, width: int, height: int):
        assert width > 1
        assert height > 1

        self.width = width
        self.height = height
        self.array = np.fromfunction(self.__construct_image, (height, width, 3), dtype=np.float32)


    def get_colour(self, x: int, y: int) -> Tuple[int, int, int]:
        assert 0 <= x <= self.width
        assert 0 <= y <= self.height

        r = self.array[y, x, 1]
        g = self.array[y, x, 0]
        b = self.array[y, x, 2]

        return int(r), int(g), int(b)


    def __construct_image(self, y: int, x: int, _channel: int):
        result = y / (self.height - 1)
        result[:, :, 1] = (x / (self.width - 1))[:, :, 1]
        result[:, :, 2] = 0.9

        result *= 255.999

        return result


def render(image, max_color: int = 255) -> str:
    result = "P3\n {width} {height} \n{max_color}\n".format(
        width=image.width, height=image.height, max_color=max_color
    )

    for j in tqdm(range(image.height - 1, -1, -1)):
        for i in range(0, image.width):
            r, g, b = image.get_colour(i, j)

            assert 0 <= r <= max_color
            assert 0 <= g <= max_color
            assert 0 <= b <= max_color

            result += "{r} {g} {b} \n".format(r=r, g=g, b=b)
    return result



class Ray:

    def __init__(self, origin: Tuple[float, float, float], direction: Tuple[float, float, float]):
        self.origin = np.fromiter(origin, dtype=np.float64)
        self.direction = np.fromiter(direction, dtype=np.float64)

    def at(self, t: float):
        return self.origin + t * self.direction



def unit(vector):
    # dot(x, x) = ||x|| ^ 2
    return vector / length(vector)


def length(vector: np. array):
    return np.sqrt(np.dot(vector, vector))


def vector(x: float, y: float, z: float) -> np.array:
    return np.fromiter([x, y, z], dtype=np.float64)


def reflect(vector: np.array, mirror: np.array) -> np.array:
    unit_mirror = unit(mirror)
    return vector - 2 * np.dot(unit_mirror, vector) * unit_mirror


def refract(vector: np.array, normal: np.array, refraction_ratio: float) -> np.array:
    cos_theta = min(np.dot(-vector, normal), 1.0)
    r_out_perp = refraction_ratio * (vector + cos_theta * normal)
    r_out_parallel = -np.sqrt(np.abs(1.0 - np.dot(r_out_perp, r_out_perp))) * normal
    return r_out_perp + r_out_parallel


class HitRecord:

    def __init__(
        self,
        t: float, 
        point: np.array, 
        outward_normal: np.array,
        ray_direction: np.array,
        material
    ):
        self.t = t
        self.point = point
        self.front_face = np.dot(ray_direction, outward_normal) < 0
        self.normal = outward_normal if self.front_face else -outward_normal
        self.material = material



class ScatteredRay:

    def __init__(self, ray: Ray, attenuation: np.array):
        self.ray = ray
        self.attenuation = attenuation


class Material:

    def scatter(self, ray: Ray, hit_record: HitRecord) -> ScatteredRay:
        raise NotImplemented()


class Lambertian(Material):

    def __init__(self, albedo: np.array):
        self.albedo = albedo

    def scatter(self, _ray: Ray, hit_record: HitRecord) -> ScatteredRay:
        target = hit_record.point + hit_record.normal + random_unit_vector()
        direction = target - hit_record.point
        if np.allclose(direction, np.zeros(shape=direction.shape)):
            return ScatteredRay(ray=Ray(hit_record.point, hit_record.normal), attenuation=self.albedo)

        return ScatteredRay(ray=Ray(hit_record.point, direction), attenuation=self.albedo)


class Metal(Material):

    def __init__(self, albedo: np.array, fuzziness: float = 0.0):
        self.albedo = albedo
        self.fuzziness = fuzziness

    def scatter(self, ray: Ray, hit_record: HitRecord) -> ScatteredRay:
        reflected = reflect(ray.direction, hit_record.normal) + self.fuzziness * random_unit_vector()
        return ScatteredRay(ray=Ray(hit_record.point, reflected), attenuation=self.albedo)


class Dielectric(Material):

    def __init__(self, index_of_refraction: float):
        self.index_of_refraction = index_of_refraction

    def scatter(self, ray: Ray, hit_record: HitRecord) -> ScatteredRay:
        color = vector(1.0, 1.0, 1.0)

        refraction_ratio =  (1.0 / self.index_of_refraction) if hit_record.front_face else self.index_of_refraction

        unit_direction = unit(ray.direction)
        cos_theta = min(np.dot(-unit_direction, hit_record.normal), 1.0)
        sin_theta = np.sqrt(1.0 - cos_theta * cos_theta)
        cannot_refract = refraction_ratio * sin_theta > 1.0

        if cannot_refract or self.__reflectance(cos_theta, refraction_ratio) > random.random():
            target_direction = reflect(unit_direction, hit_record.normal)
        else:
            target_direction = refract(unit_direction, hit_record.normal, refraction_ratio)

        return ScatteredRay(ray=Ray(hit_record.point, target_direction), attenuation=color)

    def __reflectance(self, cosine: float, refraction_ratio: float) -> float:
        # Schlick's approximation for reflectance
        r_0 = (1.0 - refraction_ratio) / (1.0 + refraction_ratio)
        r_0 = r_0 * r_0
        return r_0 + (1 - r_0) * ((1 - cosine) ** 5)



class Hittable:

    def material():
        raise NotImplemented()

    def hit(self, ray: Ray) -> Optional[HitRecord]:
        raise NotImplemented()


class Sphere(Hittable):
    def __init__(self, origin, radius, material: Material):
        self.origin = origin
        self.radius = radius
        self.material = material

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        oc = ray.origin - self.origin

        a = np.dot(ray.direction, ray.direction)
        b = np.dot(ray.direction, oc)
        c = np.dot(oc, oc) - self.radius ** 2

        discriminant = np.dot(b, b) - a * c

        # note: np.dot(x, y) * np.dot(x, y) != np.dot(x, x) * np.dot(y, y)
        # therefore, below formula is false:
        # discriminant = 4 * (sphere_radius ** 2) * np.dot(ray.direction, ray.direction)

        if discriminant < 0:
            return None

        sqrted_discriminant = np.sqrt(discriminant)

        t = (-b - sqrted_discriminant) / a

        if t_min <= t <= t_max:
            intersection_point = ray.at(t)
            return HitRecord(t=t, point=intersection_point, outward_normal=(intersection_point - self.origin) / self.radius, ray_direction=ray.direction, material=self.material)

        t = (-b + sqrted_discriminant) / a

        if t_min <= t <= t_max:
            intersection_point = ray.at(t)
            return HitRecord(t=t, point=intersection_point, outward_normal=(intersection_point - self.origin) / self.radius, ray_direction=ray.direction, material=self.material)

        return None



class World(Hittable):

    def __init__(self, objects: List[Hittable]):
        self.objects = objects

    def hit(self, ray: Ray, t_min: float, t_max: float) -> Optional[HitRecord]:
        closest_hit_record = None

        for hittable in self.objects:
            hit_record = hittable.hit(ray, t_min, t_max if closest_hit_record is None else closest_hit_record.t)
            if hit_record is not None:
                closest_hit_record = hit_record

        return closest_hit_record



def random_point_in_unit_sphere() -> np.array:
    while True:
        point = (np.random.random(size=(3,)) * 2) - 1
        if length(point) <= 1:
            return point


def random_unit_vector() -> np.array:
    return unit(random_point_in_unit_sphere());


# def random_point_in_sphere(origin: np.array, radius: float) -> np.array:
#     perturbation = (np.random.random(size=origin.shape) * 2) - 1
#     point = origin + perturbation

#     while length(point - origin) > radius:
#         perturbation = (np.random.random(size=origin.shape) * 2) - 1
#         point = origin + perturbation

#     return point


max_depth = 10


def ray_color(world: Hittable, ray: Ray, step=0, t_min=0.001, t_max=99999999999) -> np.array:
    if step >= max_depth:
        return vector(0.0, 0.0, 0.0)

    hit_record = world.hit(ray, t_min, t_max)
    if hit_record is not None:
        scattered_ray = hit_record.material.scatter(ray, hit_record)
        return scattered_ray.attenuation * ray_color(world, scattered_ray.ray, step=step+1)
        # return 0.5 * (hit_record.normal + vector(1, 1, 1))

    unit_direction = unit(ray.direction)
    t = 0.5 * (unit_direction[1] + 1.0)
    return (1.0 - t) * vector(1.0, 1.0, 1.0) + t * vector(0.5, 0.7, 1.0)


class Camera:

    def __init__(self, 
        lookfrom: np.array, 
        lookat: np.array,
        vup: np.array,
        vertical_fov_angle: float, 
        aspect_ratio: float = 16 / 9, 
    ):
        h = np.tan(vertical_fov_angle / 2)
        viewport_height = 2 * h
        viewport_width = aspect_ratio * viewport_height

        w = unit(lookfrom - lookat)
        u = unit(np.cross(vup, w))
        v = np.cross(w, u)

        # logging.info("vertical field of view angle: {angle}\nviewport_width:{w}\nviewport_height:{h}".format(angle=vertical_fov_angle, w=viewport_width, h=viewport_height))

        self.origin = lookfrom
        self.__horizontal = u * viewport_width
        self.__vertical = v * viewport_height
        self.__lower_left_corner = lookfrom - self.__horizontal / 2 - self.__vertical / 2 - w

    def get_ray(self, u: float, v: float) -> Ray:
        return Ray(self.origin, self.__lower_left_corner +  u * self.__vertical + v * self.__horizontal - self.origin)



def render_rays(camera: Camera, world: World, width, height, max_color=255):
    result = "P3\n {width} {height} \n{max_color}\n".format(
        width=width, height=height, max_color=max_color
    )

    for j in tqdm(range(height - 1, -1, -1)):
        for i in range(0, width):

            color = vector(0, 0, 0)
            for sample_num in range(1, samples_per_pixel + 1):
                u = (j + random.random()) / (height - 1)
                v = (i + random.random()) / (width - 1)

                ray = camera.get_ray(u, v)

                color += ray_color(world, ray) 

            # import pdb; pdb.set_trace()
            color /= samples_per_pixel

            # gamma 2
            r, g, b = np.sqrt(color)

            ir, ig, ib = (int(x * 255.999) for x in (r, g, b))
            # import pdb; pdb.set_trace()

            assert 0 <= ir <= max_color, "{ir} is too large".format(ir=ir)
            assert 0 <= ig <= max_color, "{ig} is too large".format(ig=ig)
            assert 0 <= ib <= max_color, "{ib} is too large".format(ib=ib)

            result += "{r} {g} {b} \n".format(r=ir, g=ig, b=ib)
    return result



# def render_rays_vectorized(width, height):


#     colors = np.array()  # height x width x 3

#     for sample_num in range(samples_per_pixel):
#         colors += 

#     pass


# ray = Ray((-0.2, 0.4, 0.67), (1, 1, 1))

# import pdb; pdb.set_trace()


# image = NumpyImage(IMAGE_WIDTH, IMAGE_HEIGHT)


ASPECT_RATIO = 16 / 9

IMAGE_WIDTH = 256
IMAGE_HEIGHT = int(IMAGE_WIDTH / ASPECT_RATIO)

samples_per_pixel = 20

# far_camera = Camera(lookfrom=vector(-2, 2, 1), lookat=vector(0,0,-1), vup=vector(0, 1, 0), vertical_fov_angle=np.pi * 0.5, aspect_ratio=ASPECT_RATIO)
camera = Camera(lookfrom=vector(-2, 2, 1), lookat=vector(0,0,-1), vup=vector(0,1,0), vertical_fov_angle=np.pi * 0.11, aspect_ratio=ASPECT_RATIO)


material_ground = Lambertian(vector(0.8, 0.8, 0.0))
material_center = Lambertian(vector(0.1, 0.2, 0.5))
material_left = Dielectric(1.5)
material_right = Metal(vector(0.8, 0.6, 0.2), fuzziness=1.0)

world = World([
    Sphere(vector(0.0, -100.5, -1.0), 100, material_ground),
    Sphere(vector(0.0, 0.0, -1.0), 0.5, material_center),
    Sphere(vector(-1.0, 0.0, -1.0), 0.5, material_left),
    Sphere(vector(-1.0, 0.0, -1.0), -0.45, material_left),
    Sphere(vector(1.0, 0.0, -1.0), 0.5, material_right)
])

print(render_rays(camera, world, IMAGE_WIDTH, IMAGE_HEIGHT))
