use super::model_reader::model_reader::ModelReader;
use super::vk_boot::vk_base::VkBase;
use nalgebra::*;

struct VkModel {}

impl VkModel {
    pub fn new(bvk: &VkBase, model: &impl ModelReader, model_matrix: Matrix4<f32>) {}
}
