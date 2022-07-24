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
        let delta_time = self.time_start.elapsed().as_millis();
        if delta_time > 1000 {
            println!(
                "Msec/frame: {}, FPS: {}",
                delta_time / self.rendered_frames as u128,
                self.rendered_frames
            );
            self.time_start = Instant::now();
            self.rendered_frames = 0;
        }
    }
}
