mod vk_renderer;
mod window_manager;

use crate::vk_renderer::model_reader::model_reader::{MeshAttributeType, TextureType};
use std::time::Instant;
use vk_renderer::model_reader::gltf_model_reader::GltfModelReader;
use vk_renderer::model_reader::model_reader::ModelReader;

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
    let res = sponza.copy_model_data_to_ptr(
        MeshAttributeType::all(),
        TextureType::all(),
        std::ptr::null_mut(),
    );
    let model_size = res.compute_total_required_size();
    println!("Model is {} bytes", model_size);

    let mut vec_data = vec![0u8; model_size];
    let res = sponza.copy_model_data_to_ptr(
        MeshAttributeType::all(),
        TextureType::all(),
        vec_data.as_mut_ptr(),
    );

    println!("Time elapsed {}", starting_time.elapsed().as_millis())
}
