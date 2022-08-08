use std::time::Instant;

pub struct FrameTimer {
    time_start: Instant,
    rendered_frames: u64,
}

impl FrameTimer {
    pub fn new() -> Self {
        FrameTimer {
            time_start: Instant::now(),
            rendered_frames: 0,
        }
    }

    pub fn frame_end(&mut self) {
        self.rendered_frames += 1;
        let delta_time = self.time_start.elapsed();
        if delta_time.as_millis() > 1000 {
            println!(
                "Msec/frame: {:.3}, FPS: {}",
                delta_time.as_secs_f32() * 1000f32 / self.rendered_frames as f32,
                self.rendered_frames
            );
            self.time_start = Instant::now();
            self.rendered_frames = 0;
        }
    }
}
