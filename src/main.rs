mod vk_renderer;
mod window_manager;

use vk_renderer::model_reader::gltf_model_reader::GltfModelReader;
use vk_renderer::model_reader::model_reader::ModelReader;
use vk_renderer::renderer::*;
use window_manager::WindowManager;

fn main() {
    let window_size = (800u32, 800u32);

    let mut window = WindowManager::new(window_size, None);
    let mut renderer = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());

    let sponza = GltfModelReader::open("assets/models/WaterBottle.glb".as_ref(), true, None);
    println!("Hello");
}
