use super::model_reader::model_reader::ModelReader;
use crate::vk_renderer::vk_allocator::VkAllocator;
use crate::{MeshAttributeType, TextureType};
use nalgebra::*;
use std::cell::RefCell;
use std::rc::Rc;

struct VkModel {}

impl VkModel {
    pub fn new(
        allocator: Rc<RefCell<VkAllocator>>,
        model: &impl ModelReader,
        model_matrix: Matrix4<f32>,
    ) {
        let copy_info = model.copy_model_data_to_ptr(
            MeshAttributeType::VERTICES
                | MeshAttributeType::TEX_COORDS
                | MeshAttributeType::NORMALS
                | MeshAttributeType::TANGENTS
                | MeshAttributeType::INDICES,
            TextureType::ALBEDO | TextureType::ORM | TextureType::NORMAL,
            std::ptr::null_mut(),
        );
    }
}
