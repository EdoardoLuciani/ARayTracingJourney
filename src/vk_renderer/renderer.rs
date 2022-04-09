use super::model_reader::model_reader::ModelReader;
use super::vk_boot::*;
use ash::{extensions::*, vk};
use nalgebra::*;

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
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut physical_device_acceleration_structure)
            .push_next(&mut physical_device_ray_tracing_pipeline);

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

        VulkanTempleRayTracedRenderer { bvk }
    }

    pub fn add_model(file_path: &std::path::Path, model_matrix: Matrix4<f32>) {
        let acceleration_structure_geometry_triangles_data =
            vk::AccelerationStructureGeometryTrianglesDataKHR::default();
        let acceleration_structure_geometry_data = vk::AccelerationStructureGeometryDataKHR {
            triangles: acceleration_structure_geometry_triangles_data,
        };
        let acceleration_structure_geometry = vk::AccelerationStructureGeometryKHR::builder()
            .geometry_type(vk::GeometryTypeKHR::TRIANGLES)
            .geometry(acceleration_structure_geometry_data)
            .flags(vk::GeometryFlagsKHR::OPAQUE);
    }
}
