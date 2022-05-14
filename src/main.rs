mod vk_renderer;
mod window_manager;

use crate::vk_renderer::renderer::VulkanTempleRayTracedRenderer;
use crate::window_manager::WindowManager;
use nalgebra::*;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

fn main() {
    let window_size = (800u32, 800u32);

    let mut window = WindowManager::new(window_size, None);
    let mut renderer = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());
    renderer.add_model(
        std::path::Path::new("assets/shaders-spirv"),
        Matrix3x4::identity(),
    );
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                _ => {}
            },
            Event::MainEventsCleared => renderer.render_frame(&window.window),
            _ => {}
        }
    });
}
