#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

#define num_objs 5
#define num_samples 5
#define max_bounces 5

#define PI 3.14

float g_seed = 0.0;

//
// Base object types
//

struct ray{
	vec3 origin;
	vec3 dir;
};

struct camera{
	ray transform;
	float focal_length;
	float eye_radius;
};

struct material{
	vec3 color;
	float roughness;
	float refractive;
};

struct sphere{
	vec3 center;
	float radius;

	vec3 color;
};

struct disk{
	vec3 center;
	vec3 normal;
	float radius;

	vec3 color;
};

struct hittable{
	// enum of what type of object this is
	// 0: sphere
	// 1: disk
	// 2: point_light
	int type;

	vec3 center;
	vec3 normal;

	float radius;

	material mat;
};

struct hit{
	float dist;
	int type;

	vec3 position;
	vec3 normal;
	vec3 reflection;

	material mat;
};

//
// Utilities
//

// random for point
// https://stackoverflow.com/a/10625698
float random(vec2 pos){
    vec2 K1 = vec2(
        23.14069263277926, // e^pi (Gelfond's constant)
         2.665144142690225 // 2^sqrt(2) (Gelfondâ€“Schneider constant)
    );
    return fract( cos( dot(pos,K1) ) * 12345.6789 );
}

vec2 hash_vec2(vec2 pos){
    vec2 K1 = vec2(
        23.14069263277926, // e^pi (Gelfond's constant)
         2.665144142690225 // 2^sqrt(2) (Gelfondâ€“Schneider constant)
    );
	return vec2(
				fract( cos( dot(pos,K1) ) * 12345.6789 ),
				fract( cos( dot(pos,K1) ) * 34567.8912 )
			);
}

vec3 hash_vec3(vec2 pos){
    vec2 K1 = vec2(
        23.14069263277926, // e^pi (Gelfond's constant)
         2.665144142690225 // 2^sqrt(2) (Gelfondâ€“Schneider constant)
    );
	return vec3(
				fract( cos( dot(pos,K1) ) * 12345.6789 ),
				fract( cos( dot(pos,K1) ) * 34567.8912 ),
				fract( cos( dot(pos,K1) ) * 23456.7891 )
			);
}

vec3 random_in_unit_sphere(vec2 seed) {
	vec3 h = hash_vec3(seed) * vec3(2.,6.28318530718,1.)-vec3(1,0,0);
    float phi = h.y;
    float r = pow(h.z, 1./3.);
	return r * vec3(sqrt(1.-h.x*h.x)*vec2(sin(phi),cos(phi)),h.x);
}

vec3 random_on_hemisphere(vec3 normal, vec2 seed){
	vec3 base_vector = random_in_unit_sphere(seed);
	if(dot(base_vector, normal) > 0.0){
		return base_vector;
	} else{
		return -base_vector;
	}
}

vec4 color_mix(vec4 base_col, vec4 top_col, float top_factor){
	return (base_col * (1.0 - top_factor)) + (top_col * top_factor);
}

vec3 pos_along_ray(ray r, float t){
	return r.origin + (r.dir * t);
}

//
// Intersection Functions
//

float intersect_sphere(hittable obj, ray r){
	vec3 oc = obj.center - r.origin;
	float a = dot(r.dir, r.dir);
    float b = -2.0 * dot(r.dir, oc);
    float c = dot(oc, oc) - obj.radius*obj.radius;
    float discriminant = b*b - 4.0*a*c;

	// if we miss the sphere, return -1.0
	if(discriminant < 0.0){
		return -1.0;

	// else return where along the ray we hit the sphere
	} else{
		return (-b - sqrt(discriminant)) / (2.0 * a);
	}
}

float intersect_disk(hittable obj, ray r){
	float denom = dot(obj.normal, r.dir);

	// if the ray isn't parallel the plane
	if(abs(denom) > 0.00001){
		float t = dot(obj.center - r.origin, obj.normal) / denom;
		if(t < 0.0){
			return -1.0;
		} else{
			vec3 hit_point = pos_along_ray(r, t);
			float dist_from_disk_center = distance(hit_point, obj.center);
			
			if(dist_from_disk_center < obj.radius){
				return t;
			} else{
				return -1.0;
			}
		}
	}
}

hit intersect_world(ray r, hittable objs[num_objs], hittable light_obj){

	bool hit_something = false;
	hit min_hit = hit(
				1000.0,			// dist (t)
				-1,				// type
				vec3(0.0),		// location
				vec3(0.0),		// normal
				vec3(0.0),		// reflect
				material(
						vec3(0.0),
						1.0,
						0.0
					)		// color
			);

	for(int obj = 0; obj < num_objs; obj++){
		
		float t = 0.0;
		if(objs[obj].type == 0 || objs[obj].type == 2){
			t = intersect_sphere(objs[obj], r);
		} else if(objs[obj].type == 1){
			t = intersect_disk(objs[obj], r);
		}
		if(t > 0.0001){
			hit_something = true;
			
			if(t < min_hit.dist){
				min_hit.dist = t;
				min_hit.position = pos_along_ray(r, t);
				min_hit.mat = objs[obj].mat;
				min_hit.type = objs[obj].type;
				
				if(objs[obj].type == 0 || objs[obj].type == 2){
					min_hit.normal = min_hit.position - objs[obj].center;
				} else if(objs[obj].type == 1){
					min_hit.normal = objs[obj].normal;
				}
				
				min_hit.reflection = r.dir - 2.0*dot(r.dir, min_hit.normal)*min_hit.normal;
			}
		}
	}

	float t = 0.0;
	if(light_obj.type == 0 || light_obj.type == 2){
		t = intersect_sphere(light_obj, r);
	} else if(light_obj.type == 1){
		t = intersect_disk(light_obj, r);
	}
	if(t > 0.0001){
		hit_something = true;
		
		if(t < min_hit.dist){
			min_hit.dist = t;
			min_hit.position = pos_along_ray(r, t);
			min_hit.mat = light_obj.mat;
			min_hit.type = light_obj.type;
			
			if(light_obj.type == 0 || light_obj.type == 2){
				min_hit.normal = min_hit.position - light_obj.center;
			} else if(light_obj.type == 1){
				min_hit.normal = light_obj.normal;
			}
			
			min_hit.reflection = r.dir - 2.0*dot(r.dir, min_hit.normal)*min_hit.normal;
		}
	}


	if(!hit_something){
		min_hit.dist = -1.0;;
	}
	
	return min_hit;
}

bool in_shadow(hit in_hit, hittable objs[num_objs], hittable light_obj){
	
	float rand_influence = 0.4;

	vec3 light_dir = normalize(light_obj.center - in_hit.position);

	if(dot(light_dir, in_hit.normal) < 0.0){
		return true;
	}
	
	ray light_ray = ray(
				in_hit.position,
				(light_dir * (1.0 - rand_influence)) +
				random_on_hemisphere(
					light_dir,
					light_dir.xy * g_seed
				) * rand_influence
			);

	hit check_light_hit = intersect_world(light_ray, objs, light_obj);

	if(check_light_hit.type == 2){
		return false;
	}

	return true;
}

ray get_brdf_ray(int sample_num, hit in_hit, ray in_ray, inout bool specular){
	
	ray new_ray = in_ray;

	float rand = random(in_ray.dir.xy * g_seed);

	new_ray = ray(in_hit.position,
					(in_hit.reflection * (1.0 - in_hit.mat.roughness)) +
					random_on_hemisphere(
						in_hit.normal,
						in_ray.dir.xy + g_seed
					) * in_hit.mat.roughness
				);
	
	if(in_hit.mat.roughness < 0.5){
		specular = true;
	}

	
	return new_ray;
}

vec3 sample_trace_path(int sample_num, ray start_ray, hittable objs[num_objs], hittable light_obj){
	
	vec3 sample_color = vec3(0.0);
	vec3 factor = vec3(2.0);

	ray cur_ray = start_ray;

	bool first_hit = true;
	bool specular = false;

	for(int bounce = 0; bounce < max_bounces; bounce++){
		hit cur_hit = intersect_world(cur_ray, objs, light_obj);

		// if we hit something
		if(cur_hit.dist > 0.0){

			// is it a light?
			if(cur_hit.type == 2){
				sample_color += factor * cur_hit.mat.color;
				return sample_color;
			}
			// else{
			// 	sample_color = cur_hit.mat.color;
			// 	return sample_color;
			// }

			// every time we hit something, we change the amount
			// of light in different colors that comes through
			factor *= cur_hit.mat.color;

			// gets the next bounce ray
			// also updates the `specular` boolean value
			cur_ray = get_brdf_ray(sample_num, cur_hit, cur_ray, specular);

			// // direct light sampling
			if(!specular && !in_shadow(cur_hit, objs, light_obj)){
				cur_ray = ray(
							cur_hit.position,
								normalize(light_obj.center - cur_hit.position)
						);
			}

			first_hit = false;

			
			
		// if we hit nothing
		} else{

			vec4 top_col = vec4(0.18, 0.18, 0.227, 1.0);
			vec4 bot_col = vec4(0.737, 0.365, 0.18, 1.0);
			vec4 gradient = color_mix(bot_col, top_col, 0.5 + (cur_ray.dir.y * 0.5)) * 0.3;

			sample_color += factor * gradient.xyz;
			return sample_color;
		}

	}

	return sample_color;

}

void mainImage(out vec4 frag_color, in vec2 frag_coord)
{
	// random seed
	g_seed = float(random(frag_coord)) + u_time + 20.0;
    
	// Normalized pixel coordinates (from 0 to 1)
    vec2 uv = frag_coord / u_resolution.xy;
	vec2 mouse_uv = u_mouse / u_resolution.xy;
	
	// Map the coordinates to have square pixels
	float smaller_side = min(u_resolution.x, u_resolution.y);
	vec2 mapped_coords = frag_coord / smaller_side;
	vec2 mapped_res = u_resolution / smaller_side;
	vec2 mouse_mapped = u_mouse / smaller_side;

	if(smaller_side == u_resolution.x){
		float larger_excess = (u_resolution.y/smaller_side) - 1.0;
		mapped_coords = vec2(mapped_coords.x, mapped_coords.y - (0.5 * larger_excess));
		mapped_res = vec2(mapped_res.x, mapped_res.y - (0.5 * larger_excess));
		mouse_mapped = vec2(mouse_mapped.x, mouse_mapped.y - (0.5 * larger_excess));
	} else if(smaller_side == u_resolution.y){
		float larger_excess = (u_resolution.x/smaller_side) - 1.0;
		mapped_coords = vec2(mapped_coords.x - (0.5 * larger_excess), mapped_coords.y);
		mapped_res = vec2(mapped_res.x - (0.5 * larger_excess), mapped_res.y);
		mouse_mapped = vec2(mouse_mapped.x - (0.5 * larger_excess), mouse_mapped.y);
	}

	// The base color of the output image
	vec4 base_color = vec4(0.0);

	// scene camera definition
	//vec3 cam_origin = vec3(-sin(u_time/2.0)*10.0, 3.0, 40.0);
	vec3 cam_origin = vec3(-sin(u_time/2.0)*40.0, 3.0, cos(u_time/2.0)*40.0);
	//vec3 cam_origin = vec3(0.0, 3.0, 40.0);
	camera cam = camera(
					ray(
						cam_origin,	// origin
						-normalize(vec3(cam_origin.x, 0.0, cam_origin.z))
					),
					3.0,							// focal length
					0.003							// eye radius
				);
	//
	// direction to each pixel from the camera
	//

	vec3 up_dir = normalize(vec3(0.0, 1.0, 0.0));
	vec3 left_dir = normalize(cross(cam.transform.dir, up_dir));

	float pixel_dist_h = 2.0 * mapped_coords.x - 1.0;
	float pixel_dist_v = 2.0 * mapped_coords.y - 1.0;

	// The direction to the current pixel
	//
	// Randomizing where the initial vector starts from within the eye radius
	// kind of approximates depth of field
    vec3 pixel_dir = (
				(cam.transform.dir * cam.focal_length) + 
				left_dir*pixel_dist_h + up_dir*pixel_dist_v
			);
	ray pixel_ray = ray(
				cam.transform.origin,
				normalize(pixel_dir)
			);


	// Define the scene
	
	hittable objs[num_objs];

	#define wall_size 30.0

	material white_mat = material(
				vec3(0.9, 0.9, 0.9),
				0.9,
				0.0
			);
	material a_mat = material(
				vec3(0.953, 0.259, 0.075),
				0.9,
				0.0
			);
	material b_mat = material(
				vec3(0.498, 0.471, 0.906),
				1.0,
				0.0
			);
	material c_mat = material(
				vec3(0.0, 0.0, 0.9),
				0.9,
				0.0
			);
	material light_mat = material(
				vec3(1.0, 0.992, 0.31),
				0.9,
				0.0
			);
	material mirror_mat = material(
				vec3(0.9, 0.9, 0.9),
				0.4,
				0.0
			);

	hittable light_sphere = hittable(	// mouse-tracker sphere
				2,
				//vec3(sin(u_time)*8.0, 3.0, cos(u_time)*8.0),
				vec3(8.0, 3.0, 8.0),
				vec3(0.0, 0.0, 0.0),
				2.0,
				light_mat
			);

	objs[0] = hittable(
				0,
				vec3(4.0, 1.0, -4.0),
				vec3(0.0, 0.0, 0.0),
				3.0,
				mirror_mat
			);
	objs[1] = hittable(
				0,
				vec3(-5.0, 4.0, 6.0),
				vec3(0.0, 0.0, 0.0),
				2.0,
				a_mat
			);
	objs[2] = hittable(					// floor
				1,						// type: disk
				vec3(0.0, -2.0, 0.0),	// center
				vec3(0.0, 1.0, 0.0),	// normal
				wall_size,				// radius
				white_mat				// color
			);
	objs[3] = hittable(
				0,
				vec3(-10.0, 1.0, 4.0),
				vec3(0.0, 0.0, 0.0),
				3.0,
				b_mat
			);

	// Background gradient
	vec4 top_col = vec4(0.3, 0.3, 0.7, 1.0);
	vec4 bot_col = vec4(0.8, 0.8, 1.0, 1.0);
	vec4 gradient = color_mix(bot_col, top_col, 0.5 + (pixel_ray.dir.y * 0.5));


	//
    // Handle intersections
	//

	vec3 total_color = vec3(0.1);

	for(int sample_num = 0; sample_num < num_samples; sample_num++){
		total_color += sample_trace_path(sample_num, pixel_ray, objs, light_sphere);
	}
	total_color /= float(num_samples);
	
	frag_color = vec4(total_color, 1.0);
}

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
