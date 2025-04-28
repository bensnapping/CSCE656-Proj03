
precision highp float;

uniform vec2 u_resolution;
uniform vec2 u_mouse;

const int MAX_BOUNCES = 3;
const float EPSILON = 0.01;
const float IOR = 1.5;

struct Ray {
    vec3 origin;
    vec3 dir;
};

struct Hit {
    float t;
    vec3 point;
    vec3 normal;
    vec3 color;
    bool hit;
    bool isPlane;
    bool isRefractive;
};

vec3 lightPos;

bool intersectSphere(Ray ray, vec3 center, float radius, out Hit hit, vec3 color, bool refractFlag) {
    vec3 oc = ray.origin - center;
    float b = dot(oc, ray.dir);
    float c = dot(oc, oc) - radius * radius;
    float h = b * b - c;
    if (h < 0.0) return false;
    h = sqrt(h);
    float t = -b - h;
    if (t < EPSILON) t = -b + h;
    if (t < EPSILON) return false;

    hit.t = t;
    hit.point = ray.origin + ray.dir * t;
    hit.normal = normalize(hit.point - center);
    hit.color = color;
    hit.hit = true;
    hit.isPlane = false;
    hit.isRefractive = refractFlag;
    return true;
}

bool intersectPlane(Ray ray, out Hit hit, vec3 color) {
    float denom = dot(ray.dir, vec3(0, 1, 0));
    if (abs(denom) < 0.001) return false;
    float t = -(ray.origin.y + 1.0) / denom;
    if (t < EPSILON) return false;

    hit.t = t;
    hit.point = ray.origin + ray.dir * t;
    hit.normal = vec3(0, 1, 0);
    hit.color = color;
    hit.hit = true;
    hit.isPlane = true;
    hit.isRefractive = false;
    return true;
}

bool inShadow(vec3 point) {
    vec3 toLight = normalize(lightPos - point);
    Ray shadowRay = Ray(point + toLight * EPSILON, toLight);
    Hit temp;
    vec3 colors[3];
    colors[0] = vec3(1, 0, 0);
    colors[1] = vec3(0, 1, 0);
    colors[2] = vec3(0.2, 0.5, 1.0);
    vec3 centers[3];
    centers[0] = vec3(-1.5, -0.5, 0.0);
    centers[1] = vec3(0.8, -0.3, -1.0);
    centers[2] = vec3(0.0, -0.4, 1.0);

    for (int i = 0; i < 3; i++) {
        if (intersectSphere(shadowRay, centers[i], 0.5, temp, colors[i], false)) return true;
    }

    if (intersectPlane(shadowRay, temp, colors[2])) return true;
    return false;
}

vec3 trace(Ray ray) {
    vec3 finalColor = vec3(0.0);
    vec3 attenuation = vec3(1.0);
    float currentIOR = 1.0;

    for (int bounce = 0; bounce < MAX_BOUNCES; bounce++) {
        Hit closest;
        closest.t = 1e6;
        closest.hit = false;

        vec3 colors[3];
        colors[0] = vec3(1, 0.2, 0.2);
        colors[1] = vec3(0.2, 1, 0.3);
        colors[2] = vec3(0.9, 0.9, 1.0);
        vec3 centers[3];
        centers[0] = vec3(-1.5, -0.5, 0.0);
        centers[1] = vec3(0.8, -0.3, -1.0);
        centers[2] = vec3(0.0, -0.4, 1.0);

        Hit hit;
        for (int i = 0; i < 3; i++) {
            bool isRefractive = (i == 2);
            if (intersectSphere(ray, centers[i], 0.5, hit, colors[i], isRefractive) && hit.t < closest.t) {
                closest = hit;
            }
        }

        if (intersectPlane(ray, hit, vec3(0.2, 0.5, 1.0)) && hit.t < closest.t) {
            closest = hit;
        }

        if (!closest.hit) break;

        vec3 toLight = normalize(lightPos - closest.point);
        float diff = max(dot(closest.normal, toLight), 0.0);
        float shadow = inShadow(closest.point) ? 0.1 : 1.0;
        vec3 localColor = closest.color * diff * shadow;

        if (closest.isRefractive) {
            vec3 N = closest.normal;
            float eta = currentIOR / IOR;
            float cosi = clamp(dot(-ray.dir, N), -1.0, 1.0);
            if (cosi < 0.0) {
                cosi = -cosi;
                N = -N;
                eta = 1.0 / eta;
            }

            float k = 1.0 - eta * eta * (1.0 - cosi * cosi);
            vec3 refractedDir = (k < 0.0) ? reflect(ray.dir, N) : normalize(eta * ray.dir + (eta * cosi - sqrt(k)) * N);
            ray.origin = closest.point - N * EPSILON;
            ray.dir = refractedDir;
            attenuation *= 0.9;
            currentIOR = (currentIOR == 1.0) ? IOR : 1.0;
            continue;
        }

        finalColor += localColor * attenuation;
        ray.origin = closest.point + closest.normal * EPSILON;
        ray.dir = reflect(ray.dir, closest.normal);
        attenuation *= 0.5;
    }

    return finalColor;
}

void main() {
    vec2 uv = (gl_FragCoord.xy / u_resolution) * 2.0 - 1.0;
    uv.x *= u_resolution.x / u_resolution.y;

    Ray ray;
    ray.origin = vec3(0.0, 0.0, 4.0);
    ray.dir = normalize(vec3(uv, -1.5));

    lightPos = vec3((u_mouse / u_resolution - 0.5) * vec2(2.0, -2.0), 2.0);

    vec3 col = trace(ray);
    gl_FragColor = vec4(col, 1.0);
}

