use super::model_reader::model_reader::ModelReader;
use crate::vk_renderer::vk_allocator::VkAllocator;
use nalgebra::*;

struct VkModel {}

impl VkModel {
    pub fn new(allocator: &mut VkAllocator, model: &impl ModelReader, model_matrix: Matrix4<f32>) {
        //RefCell::borrow_mut(allocator.borrow_mut()).allocator.
    }
}
