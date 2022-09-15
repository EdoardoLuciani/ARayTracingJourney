mod frame_timer;
mod vk_renderer;
mod window_manager;

use crate::frame_timer::FrameTimer;
use crate::window_manager::WindowManager;
use ash::vk;
use nalgebra::*;
use vk_renderer::lights::*;
use vk_renderer::renderer::*;
use winit::event::{DeviceEvent, ElementState, Event, WindowEvent};
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
        (window.get_window_handle(), window.get_display_handle()),
        Settings {
            upscaling_quality: UpscalingQuality::QUALITY,
        },
    );
    renderer.add_model(
        std::path::Path::new("assets/models/Sponza.glb"),
        Similarity3::from_scaling(2.0f32).to_homogeneous(),
    );

    renderer
        .lights_mut()
        .lights_mut()
        .get_spot_lights_mut()
        .push(SpotLight::new(
            Vector3::new(0.0f32, 1.5f32, 0.0f32),
            Vector3::new(0.0f32, -1.0f32, 0.0f32).normalize(),
            Vector3::new(1.36f32, 0.16f32, 2.22f32) * 10.0f32,
            3.0f32,
            Vector2::new(30f32.to_radians(), 45f32.to_radians()),
            true,
        ));

    renderer
        .lights_mut()
        .lights_mut()
        .get_area_lights_mut()
        .push(AreaLight::new(
            Vector3::new(-0.70f32, 0.77f32, 0.08f32),
            Vector3::new(-0.70f32, 0.77f32, -0.16f32),
            Vector3::new(-0.70f32, 0.90f32, -0.16f32),
            false,
            Vector3::new(1.96f32, 0.06f32, 0.41f32) * 3.0f32,
            3.0f32,
            Vector2::from_element(90.0f32.to_radians()),
            true,
        ));

    renderer.prepare_first_frame();

    let mut frame_timer = FrameTimer::new();
    let mut camera_virtual_pos = Vector2::<f32>::new(0.0f32, 0.0f32);
    let mut clock = std::time::Instant::now();
    window.event_loop.run_return(|event, _, control_flow| {
        *control_flow = ControlFlow::Poll;
        match event {
            Event::WindowEvent { event, .. } => match event {
                WindowEvent::CloseRequested => {
                    *control_flow = ControlFlow::Exit;
                }
                WindowEvent::Resized(physical_size) => {
                    renderer
                        .camera_mut()
                        .set_aspect(physical_size.width as f32 / physical_size.height as f32);
                }
                _ => {}
            },
            Event::DeviceEvent { event, .. } => match event {
                DeviceEvent::MouseMotion { delta, .. } => {
                    const SENSITIVITY: f32 = 0.002f32;
                    camera_virtual_pos +=
                        Vector2::new(-delta.1 as f32, delta.0 as f32) * SENSITIVITY;

                    let mut euclidian_dir = Vector3::new(
                        camera_virtual_pos.x.cos() * camera_virtual_pos.y.sin(),
                        camera_virtual_pos.x.sin(),
                        camera_virtual_pos.x.cos() * camera_virtual_pos.y.cos(),
                    );
                    euclidian_dir.normalize_mut();
                    renderer.camera_mut().set_dir(euclidian_dir);
                }
                DeviceEvent::Key(keyboard_input) => {
                    let mut camera_pos_diff = Vector3::from_element(0.0f32);
                    const SPEED: f32 = 0.002f32;

                    if keyboard_input.state == ElementState::Pressed {
                        match keyboard_input.scancode {
                            0x11 => camera_pos_diff[2] = -SPEED, // W
                            0x1f => camera_pos_diff[2] = SPEED,  // S
                            0x20 => camera_pos_diff[0] = SPEED,  // D
                            0x1e => camera_pos_diff[0] = -SPEED, // A
                            0x1d => camera_pos_diff[1] = SPEED,  // LControl
                            0x2a => camera_pos_diff[1] = -SPEED, // LShift
                            0x01 => {
                                // Esc
                                *control_flow = ControlFlow::Exit
                            }
                            _ => {}
                        }
                    }
                    let new_pos = renderer.camera_mut().pos()
                        + renderer
                            .camera_mut()
                            .view_matrix()
                            .to_homogeneous()
                            .transpose()
                            .fixed_slice::<3, 3>(0, 0)
                            * camera_pos_diff
                            * clock.elapsed().as_millis() as f32;
                    renderer.camera_mut().set_pos(new_pos);
                }
                _ => {}
            },
            Event::MainEventsCleared => {
                clock = std::time::Instant::now();
                renderer.render_frame(&window.window);
                frame_timer.frame_end();
            }
            _ => {}
        }
    });
}
