mod vk_renderer;
mod window_manager;

use crate::vk_renderer::model_reader::model_reader::{MeshAttributeType, TextureType};
use crate::vk_renderer::renderer::VulkanTempleRayTracedRenderer;
use crate::window_manager::WindowManager;
use vk_renderer::model_reader::gltf_model_reader::GltfModelReader;

fn main() {
    let window_size = (800u32, 800u32);

    let mut window = WindowManager::new(window_size, None);
    let mut renderer = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());
}
