mod vk_renderer;
mod window_manager;

use std::time::Instant;
use vk_renderer::model_reader::gltf_model_reader::GltfModelReader;
use vk_renderer::model_reader::model_reader::ModelReader;
use vk_renderer::renderer::*;
use window_manager::WindowManager;

fn main() {
    let window_size = (800u32, 800u32);

    //let mut window = WindowManager::new(window_size, None);
    //let mut renderer = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());

    let starting_time = Instant::now();
    let sponza = GltfModelReader::open(
        "assets/models/WaterBottle.glb".as_ref(),
        true,
        Some(ash::vk::Format::B8G8R8A8_UNORM),
    );
    println!("{}", starting_time.elapsed().as_millis())
}
