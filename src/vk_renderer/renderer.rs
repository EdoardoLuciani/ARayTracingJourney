use super::vk_allocator::VkAllocator;
use super::vk_boot::vk_base;
use super::vk_model::VkModel;
use super::vk_rendering_layers::vk_rt_lightning_shadows::VkRTLightningShadows;
use super::vk_tlas_builder::VkTlasBuilder;
use ash::{extensions::*, vk};
use itertools::all;
use nalgebra::*;
use std::cell::RefCell;
use std::rc::Rc;

struct FrameData {
    device: Rc<ash::Device>,
    semaphores: Vec<vk::Semaphore>,
    after_exec_fence: vk::Fence,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl FrameData {
    fn new(device: Rc<ash::Device>, queue_family_index: u32) -> Self {
        let semaphores = (0..2)
            .map(|_| {
                let semaphore_ci = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_ci, None).unwrap() }
            })
            .collect::<Vec<vk::Semaphore>>();

        let after_exec_fence = unsafe {
            let fence_ci = vk::FenceCreateInfo::default();
            device.create_fence(&fence_ci, None).unwrap()
        };

        let command_pool = unsafe {
            let command_pool_ci =
                vk::CommandPoolCreateInfo::builder().queue_family_index(queue_family_index);
            device.create_command_pool(&command_pool_ci, None).unwrap()
        };

        let command_buffers = unsafe {
            let command_buffer_ai = vk::CommandBufferAllocateInfo::builder()
                .command_pool(command_pool)
                .level(vk::CommandBufferLevel::PRIMARY)
                .command_buffer_count(1);
            device.allocate_command_buffers(&command_buffer_ai).unwrap()
        };

        FrameData {
            device,
            semaphores,
            after_exec_fence,
            command_pool,
            command_buffers,
        }
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            for semaphore in self.semaphores.iter() {
                self.device.destroy_semaphore(*semaphore, None);
            }
            self.device.destroy_fence(self.after_exec_fence, None);
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

pub struct VulkanTempleRayTracedRenderer {
    bvk: vk_base::VkBase,
    device: Rc<ash::Device>,
    acceleration_structure_fp: Rc<khr::AccelerationStructure>,
    ray_tracing_pipeline_fp: Rc<khr::RayTracingPipeline>,
    allocator: Rc<RefCell<VkAllocator>>,
    models: Vec<VkModel>,
    tlas_builder: VkTlasBuilder,
    rendering_layer: VkRTLightningShadows,
    frame_data: FrameData,
    rendered_frames: u64,
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

        let device = Rc::new(bvk.get_device().clone());
        let acceleration_structure_fp = Rc::new(khr::AccelerationStructure::new(
            bvk.get_instance(),
            bvk.get_device(),
        ));
        let ray_tracing_pipeline_fp = Rc::new(khr::RayTracingPipeline::new(
            bvk.get_instance(),
            bvk.get_device(),
        ));

        let allocator = Rc::new(RefCell::new(VkAllocator::new(
            bvk.get_instance().clone(),
            device.clone(),
            bvk.get_physical_device().clone(),
        )));

        let tlas_builder = VkTlasBuilder::new(
            device.clone(),
            acceleration_structure_fp.clone(),
            allocator.clone(),
        );

        let mut ray_tracing_pipeline_properties =
            vk::PhysicalDeviceRayTracingPipelinePropertiesKHR::default();
        let mut physical_device_properties = vk::PhysicalDeviceProperties2::builder()
            .push_next(&mut ray_tracing_pipeline_properties);
        unsafe {
            bvk.get_instance().get_physical_device_properties2(
                bvk.get_physical_device(),
                &mut physical_device_properties,
            );
        }

        let rendering_layer = VkRTLightningShadows::new(
            device.clone(),
            ray_tracing_pipeline_fp.clone(),
            &ray_tracing_pipeline_properties,
            allocator.clone(),
            vk::Extent2D {
                width: window_size.0,
                height: window_size.1,
            },
            std::path::Path::new("assets//shaders-spirv"),
        );

        let frame_data = FrameData::new(device.clone(), bvk.get_queue_family_index());

        VulkanTempleRayTracedRenderer {
            bvk,
            device,
            acceleration_structure_fp,
            ray_tracing_pipeline_fp,
            allocator,
            models: Vec::default(),
            tlas_builder,
            rendering_layer,
            frame_data,
            rendered_frames: 0,
        }
    }

    pub fn add_model(&mut self, file_path: &std::path::Path, model_matrix: Matrix3x4<f32>) {
        self.models.push(VkModel::new(
            self.device.clone(),
            Some(self.acceleration_structure_fp.clone()),
            self.allocator.clone(),
            file_path.to_path_buf(),
            model_matrix,
        ));
    }
}
