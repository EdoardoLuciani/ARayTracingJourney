mod vk_renderer;
mod window_manager;

use vk_renderer::renderer::*;
use window_manager::WindowManager;

fn main() {
    let window_size = (800u32, 800u32);

    let mut window = WindowManager::new(window_size, None);
    let mut vtrt = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());
}
