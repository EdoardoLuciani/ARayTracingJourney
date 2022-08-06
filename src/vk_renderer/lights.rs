use nalgebra::*;

#[derive(Default)]
pub struct Lights {
    point_lights: Vec<PointLight>,
    spot_lights: Vec<SpotLight>,
    directional_lights: Vec<DirectionalLight>,
    area_lights: Vec<AreaLight>,
}

impl Lights {
    pub fn get_point_lights(&self) -> &[PointLight] {
        &self.point_lights
    }
    pub fn get_spot_lights(&self) -> &[SpotLight] {
        &self.spot_lights
    }
    pub fn get_directional_lights(&self) -> &[DirectionalLight] {
        &self.directional_lights
    }
    pub fn get_area_lights(&self) -> &[AreaLight] {
        &self.area_lights
    }
    pub fn copy_lights_shader_data(&self, lights_shader_data: &mut [LightShaderData]) {
        assert_eq!(lights_shader_data.len(), self.get_lights_count());

        let mut destination_idx = 0;

        self.point_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.point_lights.len();

        self.spot_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.spot_lights.len();

        self.directional_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
        destination_idx += self.directional_lights.len();

        self.area_lights
            .iter()
            .for_each(|light| lights_shader_data[destination_idx] = light.get_light_shader_data());
    }
    pub fn get_lights_count(&self) -> usize {
        self.point_lights.len()
            + self.spot_lights.len()
            + self.directional_lights.len()
            + self.area_lights.len()
    }

    pub fn get_point_lights_mut(&mut self) -> &mut Vec<PointLight> {
        &mut self.point_lights
    }
    pub fn get_spot_lights_mut(&mut self) -> &mut Vec<SpotLight> {
        &mut self.spot_lights
    }
    pub fn get_directional_lights_mut(&mut self) -> &mut Vec<DirectionalLight> {
        &mut self.directional_lights
    }
    pub fn get_area_lights_mut(&mut self) -> &mut Vec<AreaLight> {
        &mut self.area_lights
    }
}

#[derive(Copy, Clone)]
#[repr(C, packed)]
pub struct LightShaderData {
    pos: Vector3<f32>,
    light_type: u32,
    dir: Vector3<f32>,
    casts_shadows: u32,
    color: Vector3<f32>,
    falloff_distance: f32,
    area_pos2: Vector3<f32>,
    penumbra_angle: f32,
    area_pos3: Vector3<f32>,
    umbra_angle: f32,
}

trait LightShaderSerializable {
    fn get_light_shader_data(&self) -> LightShaderData;
}

enum LightType {
    Point = 0,
    Spot = 1,
    Directional = 2,
    Area = 3,
}

pub struct PointLight {
    pos: Vector3<f32>,
    color: Vector3<f32>,
    falloff_distance: f32,
    casts_shadows: bool,
}

impl PointLight {
    pub fn new(
        pos: Vector3<f32>,
        color: Vector3<f32>,
        falloff_distance: f32,
        casts_shadows: bool,
    ) -> Self {
        Self {
            pos,
            color,
            falloff_distance,
            casts_shadows,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }
    pub fn casts_shadows(&self) -> bool {
        self.casts_shadows
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
    pub fn set_casts_shadows(&mut self, casts_shadows: bool) {
        self.casts_shadows = casts_shadows;
    }
}

impl LightShaderSerializable for PointLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: self.pos,
            light_type: LightType::Point as u32,
            dir: Vector3::zeros(),
            casts_shadows: self.casts_shadows as u32,
            color: self.color,
            falloff_distance: self.falloff_distance,
            area_pos2: Vector3::zeros(),
            penumbra_angle: 0.0f32,
            area_pos3: Vector3::zeros(),
            umbra_angle: 0.0f32,
        }
    }
}

pub struct SpotLight {
    pos: Vector3<f32>,
    dir: Vector3<f32>,
    color: Vector3<f32>,
    falloff_distance: f32,
    penumbra_umbra_angles: Vector2<f32>,
    casts_shadows: bool,
}

impl SpotLight {
    pub fn new(
        pos: Vector3<f32>,
        dir: Vector3<f32>,
        color: Vector3<f32>,
        falloff_distance: f32,
        penumbra_umbra_angles: Vector2<f32>,
        casts_shadows: bool,
    ) -> Self {
        Self {
            pos,
            dir,
            color,
            falloff_distance,
            penumbra_umbra_angles,
            casts_shadows,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn dir(&self) -> Vector3<f32> {
        self.dir
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }
    pub fn penumbra_umbra_angles(&self) -> Vector2<f32> {
        self.penumbra_umbra_angles
    }
    pub fn casts_shadows(&self) -> bool {
        self.casts_shadows
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_dir(&mut self, dir: Vector3<f32>) {
        self.dir = dir;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
    pub fn set_penumbra_umbra_angles(&mut self, penumbra_umbra_angles: Vector2<f32>) {
        self.penumbra_umbra_angles = penumbra_umbra_angles;
    }
    pub fn set_casts_shadows(&mut self, casts_shadows: bool) {
        self.casts_shadows = casts_shadows;
    }
}

impl LightShaderSerializable for SpotLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: self.pos,
            light_type: LightType::Spot as u32,
            dir: self.dir,
            casts_shadows: self.casts_shadows as u32,
            color: self.color,
            falloff_distance: self.falloff_distance,
            area_pos2: Vector3::zeros(),
            penumbra_angle: self.penumbra_umbra_angles.x,
            area_pos3: Vector3::zeros(),
            umbra_angle: self.penumbra_umbra_angles.y,
        }
    }
}

pub struct DirectionalLight {
    dir: Vector3<f32>,
    color: Vector3<f32>,
    casts_shadows: bool,
}

impl DirectionalLight {
    pub fn new(dir: Vector3<f32>, color: Vector3<f32>, casts_shadows: bool) -> Self {
        Self {
            dir,
            color,
            casts_shadows,
        }
    }

    pub fn dir(&self) -> Vector3<f32> {
        self.dir
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn casts_shadows(&self) -> bool {
        self.casts_shadows
    }

    pub fn set_dir(&mut self, dir: Vector3<f32>) {
        self.dir = dir;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_casts_shadows(&mut self, casts_shadows: bool) {
        self.casts_shadows = casts_shadows;
    }
}

impl LightShaderSerializable for DirectionalLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        LightShaderData {
            pos: Vector3::zeros(),
            light_type: LightType::Directional as u32,
            dir: self.dir,
            casts_shadows: self.casts_shadows as u32,
            color: self.color,
            falloff_distance: 0.0f32,
            area_pos2: Vector3::zeros(),
            penumbra_angle: 0.0f32,
            area_pos3: Vector3::zeros(),
            umbra_angle: 0.0f32,
        }
    }
}

pub struct AreaLight {
    pos: Vector3<f32>,
    pos2: Vector3<f32>,
    pos3: Vector3<f32>,
    invert_normal: bool,
    color: Vector3<f32>,
    falloff_distance: f32,
    penumbra_umbra_angles: Vector2<f32>,
    casts_shadows: bool,
}

impl AreaLight {
    pub fn new(
        pos: Vector3<f32>,
        pos2: Vector3<f32>,
        pos3: Vector3<f32>,
        invert_normal: bool,
        color: Vector3<f32>,
        falloff_distance: f32,
        penumbra_umbra_angles: Vector2<f32>,
        casts_shadows: bool,
    ) -> Self {
        Self {
            pos,
            pos2,
            pos3,
            invert_normal,
            color,
            falloff_distance,
            penumbra_umbra_angles,
            casts_shadows,
        }
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }
    pub fn pos2(&self) -> Vector3<f32> {
        self.pos2
    }
    pub fn pos3(&self) -> Vector3<f32> {
        self.pos2
    }
    pub fn invert_normal(&self) -> bool {
        self.invert_normal
    }
    pub fn color(&self) -> Vector3<f32> {
        self.color
    }
    pub fn falloff_distance(&self) -> f32 {
        self.falloff_distance
    }
    pub fn penumbra_umbra_angles(&self) -> Vector2<f32> {
        self.penumbra_umbra_angles
    }
    pub fn casts_shadows(&self) -> bool {
        self.casts_shadows
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }
    pub fn set_pos2(&mut self, pos2: Vector3<f32>) {
        self.pos2 = pos2;
    }
    pub fn set_pos3(&mut self, pos3: Vector3<f32>) {
        self.pos3 = pos3;
    }
    pub fn set_invert_normal(&mut self, invert_normal: bool) {
        self.invert_normal = invert_normal;
    }
    pub fn set_color(&mut self, color: Vector3<f32>) {
        self.color = color;
    }
    pub fn set_falloff_distance(&mut self, falloff_distance: f32) {
        self.falloff_distance = falloff_distance;
    }
    pub fn set_penumbra_umbra_angles(&mut self, penumbra_umbra_angles: Vector2<f32>) {
        self.penumbra_umbra_angles = penumbra_umbra_angles;
    }
    pub fn set_casts_shadows(&mut self, casts_shadows: bool) {
        self.casts_shadows = casts_shadows;
    }
}

impl LightShaderSerializable for AreaLight {
    fn get_light_shader_data(&self) -> LightShaderData {
        let mut plane_normal = (self.pos - self.pos2).cross(&(self.pos3 - self.pos2));
        if self.invert_normal {
            plane_normal = -plane_normal
        }
        plane_normal.normalize_mut();
        LightShaderData {
            pos: self.pos,
            light_type: LightType::Area as u32,
            dir: plane_normal,
            casts_shadows: self.casts_shadows as u32,
            color: self.color,
            falloff_distance: self.falloff_distance,
            area_pos2: self.pos2,
            penumbra_angle: self.penumbra_umbra_angles.x,
            area_pos3: self.pos3,
            umbra_angle: self.penumbra_umbra_angles.y,
        }
    }
}
