use super::vk_boot::*;
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use nalgebra::*;
use raw_window_handle::RawWindowHandle;
use std::ffi::CStr;
use std::mem::size_of;

struct FrameData {
    after_exec_fence: vk::Fence,
    main_command: base_vk::CommandRecordInfo,
}

pub struct VulkanTempleRayTracedRenderer {
    bvk: base_vk::Base,
}

impl VulkanTempleRayTracedRenderer {
    pub fn new(window_size: (u32, u32), window_handle: raw_window_handle::RawWindowHandle) -> Self {
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder().build();

        let bvk = base_vk::Base::new(
            "VulkanTempleRayTracedRenderer",
            &[],
            &[
                "VK_KHR_ray_tracing_pipeline",
                "VK_KHR_acceleration_structure",
                "VK_KHR_deferred_host_operations",
            ],
            &physical_device_features2,
            std::slice::from_ref(&(vk::QueueFlags::GRAPHICS, 1.0f32)),
            Some(window_handle),
        );
        VulkanTempleRayTracedRenderer { bvk }
    }
}
