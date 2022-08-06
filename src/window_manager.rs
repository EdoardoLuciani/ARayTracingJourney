use raw_window_handle::{
    HasRawDisplayHandle, HasRawWindowHandle, RawDisplayHandle, RawWindowHandle,
};
use winit::dpi::LogicalSize;
use winit::event_loop::EventLoop;
use winit::window::Fullscreen;
use winit::*;

pub struct WindowManager {
    pub event_loop: EventLoop<()>,
    pub window: window::Window,
}

impl WindowManager {
    pub fn new(resolution: (u32, u32), fullscreen: Option<Fullscreen>) -> Self {
        let event_loop = event_loop::EventLoop::new();
        let window = window::WindowBuilder::new()
            .with_fullscreen(fullscreen)
            .with_min_inner_size(LogicalSize {
                width: resolution.0,
                height: resolution.1,
            })
            .build(&event_loop)
            .unwrap();
        WindowManager { event_loop, window }
    }

    pub fn get_window_handle(&self) -> RawWindowHandle {
        self.window.raw_window_handle()
    }

    pub fn get_display_handle(&self) -> RawDisplayHandle {
        self.window.raw_display_handle()
    }
}
