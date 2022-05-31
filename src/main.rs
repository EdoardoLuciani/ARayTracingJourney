mod vk_renderer;
mod window_manager;

use crate::window_manager::WindowManager;
use ash::vk;
use nalgebra::*;
use vk_renderer::renderer::VulkanTempleRayTracedRenderer;
use winit::event::{Event, WindowEvent};
use winit::event_loop::ControlFlow;
use winit::platform::run_return::EventLoopExtRunReturn;

fn main() {
    std::env::set_var("WINIT_UNIX_BACKEND", "x11");

    let window_size = (800u32, 800u32);
    let mut window = WindowManager::new(window_size, None);
    window.window.set_cursor_visible(false);
    //window.window.set_cursor_grab(true).unwrap();

    let mut renderer = VulkanTempleRayTracedRenderer::new(
        vk::Extent2D {
            width: window_size.0,
            height: window_size.1,
        },
        window.get_window_handle(),
    );
    renderer.add_model(
        std::path::Path::new("assets/models/Sponzav2.glb"),
        Matrix3x4::identity(),
    );
    renderer.prepare_first_frame();

    let mut camera_virtual_pos = Vector2::<f32>::new(0.0f32, 0.0f32);
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = winit::event_loop::ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::KeyboardInput { input, .. } => {
                    let mut camera_pos_diff = Vector3::from_element(0.0f32);
                    const SPEED: f32 = 0.1f32;
                    match input.virtual_keycode {
                        Some(winit::event::VirtualKeyCode::W) => camera_pos_diff[2] = -SPEED,
                        Some(winit::event::VirtualKeyCode::S) => camera_pos_diff[2] = SPEED,
                        Some(winit::event::VirtualKeyCode::D) => camera_pos_diff[0] = SPEED,
                        Some(winit::event::VirtualKeyCode::A) => camera_pos_diff[0] = -SPEED,
                        Some(winit::event::VirtualKeyCode::LControl) => camera_pos_diff[1] = SPEED,
                        Some(winit::event::VirtualKeyCode::LShift) => camera_pos_diff[1] = -SPEED,
                        Some(winit::event::VirtualKeyCode::Escape) => {
                            *control_flow = winit::event_loop::ControlFlow::Exit
                        }
                        _ => {}
                    }
                    let new_pos = renderer.camera_mut().pos()
                        + renderer
                            .camera_mut()
                            .view_matrix()
                            .transpose()
                            .fixed_slice::<3, 3>(0, 0)
                            * camera_pos_diff;
                    renderer.camera_mut().set_pos(new_pos);
                }
                WindowEvent::Resized(physical_size) => {
                    renderer
                        .camera_mut()
                        .set_aspect(physical_size.width as f32 / physical_size.height as f32);
                }
                _ => {}
            },
            Event::DeviceEvent {
                event: winit::event::DeviceEvent::MouseMotion { delta },
                ..
            } => {
                const SENSITIVITY: f32 = 0.001f32;
                camera_virtual_pos += Vector2::new(-delta.1 as f32, delta.0 as f32) * SENSITIVITY;

                let mut euclidian_dir = Vector3::new(
                    camera_virtual_pos.x.cos() * camera_virtual_pos.y.sin(),
                    camera_virtual_pos.x.sin(),
                    camera_virtual_pos.x.cos() * camera_virtual_pos.y.cos(),
                );
                euclidian_dir.normalize_mut();

                renderer.camera_mut().set_dir(euclidian_dir);
            }
            Event::MainEventsCleared => renderer.render_frame(&window.window),
            _ => {}
        }
    });
}
