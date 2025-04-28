#ifdef GL_ES
precision mediump float;
#endif

uniform float u_time;
uniform vec2 u_resolution;
uniform vec2 u_mouse;

#define NUM_OBJS 5
#define NUM_SAMPLES 10
#define MAX_BOUNCES 3

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
	float diffuse;
	float reflective;
	float refractive;
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

float reflectance(float cosine, float refraction_index) {
	// Use Schlick's approximation for reflectance.
	float r0 = (1.0 - refraction_index) / (1.0 + refraction_index);
	r0 = r0*r0;
	return r0 + (1.0-r0)*pow((1.0 - cosine),5.0);
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

hit intersect_world(ray r, hittable objs[NUM_OBJS], hittable light_obj){

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
						0.0,
						0.0
					)		// color
			);

	for(int obj = 0; obj < NUM_OBJS; obj++){
		
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

vec3 get_background_gradient(ray in_ray){
	vec4 bot_col = vec4(0.18, 0.18, 0.227, 1.0);
	vec4 top_col = vec4(0.737, 0.365, 0.18, 1.0);
	
	//vec4 top_col = vec4(1.0, 1.0, 1.0, 1.0);
	
	top_col = vec4(0.0, 0.0, 0.0, 1.0);
	bot_col = vec4(0.0, 0.0, 0.0, 1.0);
	vec4 gradient = color_mix(bot_col, top_col, 0.2 + (in_ray.dir.y * 0.5));

	return gradient.xyz;
}



vec3 sample_trace_path(ray start_ray, hittable objs[NUM_OBJS], hittable light_obj){
	
	vec3 sample_color = vec3(0.0);
	vec3 attenuation = vec3(1.0);

	ray cur_ray = start_ray;

	for(int bounce = 0; bounce < MAX_BOUNCES; bounce++){
		hit cur_hit = intersect_world(cur_ray, objs, light_obj);
		ray next_ray = cur_ray;

		// if we hit something
		if(cur_hit.dist > 0.0){

			if(cur_hit.type == 2){
				return attenuation*cur_hit.mat.color;
			}
			
			if(cur_hit.mat.reflective > 0.0){
				next_ray = ray(cur_hit.position, cur_hit.reflection);
			}
			if(cur_hit.mat.diffuse > 0.0){
				sample_color = 0.1*cur_hit.mat.color;	
				//sample_color = vec3(0.0);
				return sample_color;

				//attenuation *= cur_hit.mat.color;	
				//next_ray = ray(cur_hit.position, random_on_hemisphere(cur_hit.normal, vec2(g_seed) + cur_ray.dir.xy));



			}
			if(cur_hit.mat.refractive > 0.0){
				float air_ior = 1.0;
				float sphere_ior = 1.5;
				float eta = air_ior/sphere_ior;

				vec3 incidence = -cur_ray.dir;
				
				float cos_theta = min(1.0, dot(incidence, cur_hit.normal));
				float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

				bool cannot_refract = (eta * sin_theta > 1.0);

				vec3 out_dir = cur_hit.reflection;

				if(!cannot_refract || reflectance(cos_theta, eta) < random(vec2(g_seed))){
					vec3 out_perp_component = eta * (incidence + cos_theta * cur_hit.normal);
					vec3 out_para_component = -sqrt(abs(1.0 - length(out_perp_component)*length(out_perp_component))) *
													cur_hit.normal;

					out_dir = out_perp_component + out_para_component;
					
				}

				next_ray = ray(cur_hit.position, out_dir);
			}

			cur_ray = next_ray;


		// if we hit nothing
		} else{
			sample_color = attenuation*get_background_gradient(cur_ray);
			return sample_color;
		}

	}

	return sample_color;

}

vec3 sample_light(hit start_hit, hittable objs[NUM_OBJS], hittable light_obj, int sample_num){
	
	vec3 light_dir = light_obj.center - start_hit.position;

	float rand_influence = 2.0;

	light_dir += rand_influence * random_on_hemisphere(light_dir, vec2(g_seed) + vec2(float(sample_num)));

	if(dot(start_hit.normal, light_dir) > 0.0){
		vec3 color = sample_trace_path(ray(start_hit.position, light_dir), objs, light_obj);

		if(color == light_obj.mat.color){
			return color;
		}
	}

	return vec3(0.0);

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
	
	hittable objs[NUM_OBJS];

	#define WALL_SIZE 30.0

	material white_mat = material(
				vec3(0.9, 0.9, 0.9),
				1.0,
				0.0,
				0.0
			);
	material a_mat = material(
				vec3(0.953, 0.259, 0.075),
				0.95,
				0.05,
				0.0
			);
	material b_mat = material(
				vec3(0.498, 0.471, 0.906),
				1.0,
				0.0,
				0.0
			);
	material t_mat = material(
				vec3(1.0), // if mat.color == vec3(1.0), do texture stuff
				1.0,
				0.0,
				0.0
			);
	material light_mat = material(
				vec3(1.0, 0.992, 0.31),
				1.0,
				0.0,
				0.0
			);
	material mirror_mat = material(
				vec3(0.9, 0.9, 0.9),
				0.0,
				1.0,
				0.0
			);
	material glass_mat = material(
				vec3(0.9, 0.9, 0.9),
				0.0,
				0.0,
				1.0
			);

	hittable light_sphere = hittable(
				2,
				//vec3(sin(u_time)*8.0, 3.0, cos(u_time)*8.0),
				vec3(8.0, 3.0, 8.0),
				vec3(0.0, 0.0, 0.0),
				2.0,
				light_mat
			);
	objs[0] = hittable(					// floor
				1,						// type: disk
				vec3(0.0, -2.0, 0.0),	// center
				vec3(0.0, 1.0, 0.0),	// normal
				WALL_SIZE,				// radius
				white_mat				// color
			);
	objs[1] = hittable(					// mirror ball
				0,
				vec3(4.0, 1.0, -4.0),
				vec3(0.0, 0.0, 0.0),
				3.0,
				mirror_mat
			);
	objs[2] = hittable(					// blue ball
				0,
				vec3(-8.0, 1.0, -4.0),
				vec3(0.0, 0.0, 0.0),
				3.0,
				b_mat
			);
	objs[3] = hittable(					// orange ball
				0,
				vec3(-5.0, 4.0, 6.0),
				vec3(0.0, 0.0, 0.0),
				2.0,
				a_mat
			);
	objs[4] = hittable(					// glass ball
				0,
				vec3(-10.0, 1.0, 4.0),
				vec3(0.0, 0.0, 0.0),
				3.0,
				glass_mat
			);

	//
    // Handle intersections
	//

	vec3 total_color = vec3(0.1);

	
	// get first intersection
	hit first_hit = intersect_world(pixel_ray, objs, light_sphere);

	if(first_hit.dist > 0.0){
	
		// if light
		if(first_hit.type == 2){
			total_color = first_hit.mat.color;
		} else{

			// if diffuse
			if(first_hit.mat.diffuse > 0.0){
				
				vec3 diffuse_color = vec3(0.0);
				vec3 light_dir = light_sphere.center - first_hit.position;

				for(int sample_num = 0; sample_num<NUM_SAMPLES; sample_num++){
					//vec3 sample_color = first_hit.mat.color;
					vec3 sample_color = vec3(0.0);
					vec3 sample_dir = vec3(0.0);
					
					sample_dir = (0.1*first_hit.normal) +
								 random_on_hemisphere(first_hit.normal, vec2(g_seed) + vec2(float(sample_num)));
					
					// if( dot(normalize(light_dir), normalize(first_hit.normal))
					// 		> (random(vec2(g_seed) + vec2(float(sample_num)))) ){
					// 	sample_dir += (0.4 * light_dir);	
					// }

					sample_color += first_hit.mat.color *
									(sample_trace_path(ray(first_hit.position, sample_dir), objs, light_sphere));

					diffuse_color += sample_color;
				}
				diffuse_color /= float(NUM_SAMPLES);

				total_color += (diffuse_color * first_hit.mat.diffuse);
			
			}
			// if reflective
			if(first_hit.mat.reflective > 0.0){

				vec3 reflective_color = vec3(0.0);

				float mirror_roughness = 0.0;
				
				if(mirror_roughness > 0.0){
					for(int sample_num = 0; sample_num<NUM_SAMPLES; sample_num++){
						vec3 sample_color = vec3(0.0);

						vec3 sample_dir = ((1.0 - mirror_roughness)*first_hit.reflection) +
										  (mirror_roughness *
										   random_on_hemisphere(first_hit.normal, vec2(g_seed) + vec2(sample_num)));
						sample_color = sample_trace_path(ray(first_hit.position, sample_dir), objs, light_sphere);
						reflective_color += sample_color;
					}
					reflective_color /= float(NUM_SAMPLES);

				} else{
					vec3 sample_dir = (first_hit.reflection);
					
					reflective_color = sample_trace_path(ray(first_hit.position, sample_dir), objs, light_sphere);
	
				}


				total_color += (reflective_color * first_hit.mat.reflective);


			}
			// if refractive
			if(first_hit.mat.refractive > 0.0){
				
				float air_ior = 1.0;
				float sphere_ior = 1.5;
				float eta = air_ior/sphere_ior;

				vec3 incidence = -pixel_ray.dir;
				
				float cos_theta = min(1.0, dot(incidence, first_hit.normal));
				float sin_theta = sqrt(1.0 - cos_theta*cos_theta);

				bool cannot_refract = (eta * sin_theta > 1.0);

				vec3 out_dir = first_hit.reflection;

				if(!cannot_refract || reflectance(cos_theta, eta) < random(vec2(g_seed))){
					vec3 out_perp_component = eta * (incidence + cos_theta * first_hit.normal);
					vec3 out_para_component = -sqrt(abs(1.0 - length(out_perp_component)*length(out_perp_component))) *
													first_hit.normal;

					out_dir = out_perp_component + out_para_component;
					
				}
				
				total_color = sample_trace_path(ray(first_hit.position, out_dir), objs, light_sphere);

			}

		}

	} else{
		total_color = get_background_gradient(pixel_ray);
	}


	frag_color = vec4(total_color, 1.0);
}

void main() {
    mainImage(gl_FragColor, gl_FragCoord.xy);
}
