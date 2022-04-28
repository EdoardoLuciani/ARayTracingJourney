use super::vk_allocator::VkAllocator;
use super::vk_boot::vk_base;
use crate::vk_renderer::model_reader::model_reader::ModelReader;
use crate::{GltfModelReader, MeshAttributeType, TextureType};
use ash::{extensions::*, vk};
use gpu_allocator::MemoryLocation;
use nalgebra::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

struct FrameData {
    after_exec_fence: vk::Fence,
    main_command: vk_base::CommandRecordInfo,
}

pub struct VulkanTempleRayTracedRenderer {
    bvk: vk_base::VkBase,
}

impl VulkanTempleRayTracedRenderer {
    pub fn new(window_size: (u32, u32), window_handle: raw_window_handle::RawWindowHandle) -> Self {
        let mut physical_device_ray_tracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
        let mut physical_device_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true);
        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);
        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true);
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut physical_device_acceleration_structure)
            .push_next(&mut physical_device_ray_tracing_pipeline)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let bvk = vk_base::VkBase::new(
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
        let acceleration_structure_fp =
            khr::AccelerationStructure::new(bvk.instance(), bvk.device());

        let allocator = Rc::new(RefCell::new(VkAllocator::new(
            bvk.instance().clone(),
            bvk.device(),
            bvk.physical_device().clone(),
        )));

        VulkanTempleRayTracedRenderer { bvk }
    }

    pub fn add_model(file_path: &std::path::Path, model_matrix: Matrix4<f32>) {}
}
