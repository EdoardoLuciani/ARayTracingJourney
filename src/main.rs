mod vk_renderer;
mod window_manager;

use crate::window_manager::WindowManager;
use nalgebra::*;
use vk_renderer::renderer::VulkanTempleRayTracedRenderer;
use vk_renderer::vk_camera::VkCamera;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

fn main() {
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");

    let window_size = (800u32, 800u32);
    let mut window = WindowManager::new(window_size, None);
    let mut renderer = VulkanTempleRayTracedRenderer::new(window_size, window.get_window_handle());
    renderer.add_model(
        std::path::Path::new("assets/models/WaterBottle.glb"),
        Matrix3x4::identity(),
    );
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    dbg!("Moved");
                    let mut camera_pos_diff = Vector3::from_element(0.0f32);
                    const SPEED: f32 = 0.003f32;
                    match input.virtual_keycode.unwrap() {
                        winit::event::VirtualKeyCode::W => camera_pos_diff[2] = SPEED,
                        winit::event::VirtualKeyCode::S => camera_pos_diff[2] = -SPEED,
                        winit::event::VirtualKeyCode::D => camera_pos_diff[0] = SPEED,
                        winit::event::VirtualKeyCode::A => camera_pos_diff[0] = -SPEED,
                        winit::event::VirtualKeyCode::LControl => camera_pos_diff[1] = SPEED,
                        winit::event::VirtualKeyCode::LShift => camera_pos_diff[1] = -SPEED,
                        _ => {}
                    }
                    let new_pos = renderer.camera_mut().pos()
                        + renderer
                            .camera_mut()
                            .view_matrix()
                            .fixed_slice::<3, 3>(0, 0)
                            * camera_pos_diff;
                    renderer.camera_mut().set_pos(new_pos);
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                const SENSITIVITY: f32 = 0.01f32;
                let delta_pos_polar = Vector2::new(delta.0 as f32, delta.1 as f32) * SENSITIVITY;
                let mut delta_pos_euclidean = Vector3::new(
                    delta_pos_polar.x.cos() * delta_pos_polar.y.sin(),
                    delta_pos_polar.x.sin(),
                    delta_pos_polar.x.cos() * delta_pos_polar.y.cos(),
                );
                delta_pos_euclidean.normalize();
                renderer.camera_mut().set_dir(delta_pos_euclidean);
            }
            Event::MainEventsCleared => renderer.render_frame(&window.window),
            _ => {}
        }
    });
}
