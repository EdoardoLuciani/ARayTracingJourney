struct Light {
    vec3 pos;
    uint type;
    vec3 dir;
    uint casts_shadows;
    vec3 color;
    float falloff_distance;
    vec3 area_pos2;
    float penumbra_angle;
    vec3 area_pos3;
    float umbra_angle;
};

bool is_point(Light light) {
    return light.type == 0;
}

bool is_spot(Light light) {
    return light.type == 1;
}

bool is_directional(Light light) {
    return light.type == 2;
}

bool is_area(Light light) {
    return light.type == 3;
}

bool casts_shadows(Light light) {
    return bool(light.casts_shadows);
}

vec3 get_light_radiance(Light light, vec3 pos, vec3 L_vec) {
    vec3 radiance = light.color;
    if (is_spot(light) || is_area(light)) {
        float theta_s = acos(dot(light.dir, -L_vec));
        float t = clamp((theta_s - light.umbra_angle) / (light.penumbra_angle - light.umbra_angle), 0.0, 1.0);
        radiance *= pow(t, 2.0);
    }

    if (light.falloff_distance > 0.0) {
        float dist = length(light.pos - pos);
        radiance *= pow(max(1-pow(dist/light.falloff_distance, 2.0f), 0.0f), 2.0f);
    }

    return radiance;
}

vec3 compute_barycentric(vec3 a, vec3 b, vec3 c, vec3 p) {
    vec3 v0 = b - a;
    vec3 v1 = c - a;
    vec3 v2 = p - a;

    float d00 = dot(v0, v0);
    float d01 = dot(v0, v1);
    float d11 = dot(v1, v1);
    float d20 = dot(v2, v0);
    float d21 = dot(v2, v1);
    float denom = d00 * d11 - d01 * d01;

    vec3 bary;
    bary.x = (d11 * d20 - d01 * d21) / denom;
    bary.y = (d00 * d21 - d01 * d20) / denom;
    bary.z = 1 - bary.x - bary.y;
    return bary;
}

vec3 closest_point_to_segment(vec3 pos0, vec3 pos1, vec3 p) {
    vec3 v01 = pos1 - pos0;
    float t = dot(p - pos0, v01)/dot(v01, v01);
    t = clamp(t, 0.0, 1.0);
    return pos0 + t * v01;
}

vec3 closest_point_to_triangle(vec3 pos0, vec3 pos1, vec3 pos2, vec3 point) {
    vec3 cp_plane_barycentrics = compute_barycentric(pos0, pos1, pos2, point);

    vec3 closest_clamped_point;
    if (cp_plane_barycentrics.x < 0) {
        closest_clamped_point = closest_point_to_segment(pos2, pos0, point);
    }
    else if (cp_plane_barycentrics.z < 0) {
        closest_clamped_point = closest_point_to_segment(pos1, pos2, point);
    }
    else {
        closest_clamped_point = point;
    }

    return closest_clamped_point;
}

vec3 get_unnormalized_L_vec(Light light, vec3 pos) {
    if(is_point(light) || is_spot(light)) {
        return light.pos - pos;
    }
    else if (is_directional(light)) {
        return -light.dir * 10.0;
    }
    else if (is_area(light)) {
        float distance = dot(light.dir, light.area_pos2) - dot(light.dir, pos);

        vec3 cp_on_plane = pos + (distance * light.dir);
        vec3 cp_plane_barycentrics = compute_barycentric(light.pos, light.area_pos2, light.area_pos3, cp_on_plane);

        vec3 closest_clamped_point;
        if (cp_plane_barycentrics.x < 0) {
            vec3 pos4 = light.pos - light.area_pos2 + light.area_pos3;
            closest_clamped_point = closest_point_to_triangle(light.pos, light.area_pos3, pos4, cp_on_plane);
        }
        else if (cp_plane_barycentrics.y < 0) {
            closest_clamped_point = closest_point_to_segment(light.pos, light.area_pos2, cp_on_plane);
        }
        else if (cp_plane_barycentrics.z < 0) {
            closest_clamped_point = closest_point_to_segment(light.area_pos2, light.area_pos3, cp_on_plane);
        }
        else {
            closest_clamped_point = cp_on_plane;
        }

        return closest_clamped_point - pos;
    }
    return vec3(1.0);
}

