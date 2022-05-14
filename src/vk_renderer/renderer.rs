use super::vk_allocator::VkAllocator;
use super::vk_boot::vk_base;
use super::vk_model::VkModel;
use super::vk_rendering_layers::vk_rt_lightning_shadows::VkRTLightningShadows;
use super::vk_tlas_builder::VkTlasBuilder;
use ash::{extensions::*, vk};
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
    fn new(
        device: Rc<ash::Device>,
        queue_family_index: u32,
        semaphores_count: u32,
        command_buffers_count: u32,
    ) -> Self {
        let semaphores = (0..semaphores_count)
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
                .command_buffer_count(command_buffers_count);
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
    frames_data: [FrameData; 3],
    rendered_frames: u64,
}

impl VulkanTempleRayTracedRenderer {
    pub fn new(window_size: (u32, u32), window_handle: raw_window_handle::RawWindowHandle) -> Self {
        let mut physical_device_ray_tracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
        let mut physical_device_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true)
                .descriptor_binding_acceleration_structure_update_after_bind(true);
        let mut vulkan_12_features =
            vk::PhysicalDeviceVulkan12Features::builder().buffer_device_address(true);
        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true);
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .push_next(&mut physical_device_acceleration_structure)
            .push_next(&mut physical_device_ray_tracing_pipeline)
            .push_next(&mut vulkan_12_features)
            .push_next(&mut vulkan_13_features);

        let mut bvk = vk_base::VkBase::new(
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
        bvk.recreate_swapchain(
            vk::PresentModeKHR::FIFO,
            vk::Extent2D {
                width: window_size.0,
                height: window_size.1,
            },
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            vk::SurfaceFormatKHR {
                format: vk::Format::R8G8B8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
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
            vk::Format::R8G8B8A8_UNORM,
        );

        let frames_data: [FrameData; 3] = (0..3)
            .map(|_| {
                FrameData::new(
                    device.clone(),
                    bvk.get_queue_family_index(),
                    2,
                    bvk.get_swapchain_image_views().len() as u32,
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .ok()
            .unwrap();

        VulkanTempleRayTracedRenderer {
            bvk,
            device,
            acceleration_structure_fp,
            ray_tracing_pipeline_fp,
            allocator,
            models: Vec::default(),
            tlas_builder,
            rendering_layer,
            frames_data,
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

    fn record_frame_data(&mut self, frame_data_idx: usize) {
        let frame_data = &self.frames_data[frame_data_idx];
        unsafe {
            self.device
                .reset_command_pool(frame_data.command_pool, vk::CommandPoolResetFlags::empty());
        }

        for (i, cb) in frame_data.command_buffers.iter().copied().enumerate() {
            unsafe {
                let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
                self.device
                    .begin_command_buffer(cb, &command_buffer_begin_info);
            }
            for model in self.models.iter_mut() {
                model.update_model_status(&Vector3::from_element(0.0f32), cb);
                if let Some(tlas_buffer) = model.get_blas_buffer() {
                    let buffer_memory_barrier2 = vk::BufferMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                        .buffer(tlas_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE);
                    let dependency_info = vk::DependencyInfo::builder()
                        .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier2));
                    unsafe {
                        self.device.cmd_pipeline_barrier2(cb, &dependency_info);
                    }
                }
            }

            let tlases = self
                .models
                .iter()
                .filter_map(|m| m.get_acceleration_structure_instance())
                .collect::<Vec<_>>();
            let tlas = self.tlas_builder.recreate_tlas(cb, &tlases);

            let buffer_memory_barrier2 = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
                .buffer(self.tlas_builder.get_tlas_buffer().unwrap())
                .offset(0)
                .size(vk::WHOLE_SIZE);
            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier2));
            unsafe {
                self.device.cmd_pipeline_barrier2(cb, &dependency_info);
            }

            self.rendering_layer.set_tlas(tlas);
            self.rendering_layer.trace_rays(cb);

            let image_memory_barrier2 = vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                .dst_access_mask(vk::AccessFlags2::TRANSFER_READ)
                .old_layout(vk::ImageLayout::GENERAL)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.rendering_layer.get_output_image())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            let dependency_info = vk::DependencyInfo::builder()
                .image_memory_barriers(std::slice::from_ref(&image_memory_barrier2));
            unsafe {
                self.device.cmd_pipeline_barrier2(cb, &dependency_info);
            }

            unsafe {
                let region = vk::ImageBlit2::builder()
                    .src_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .src_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: self.bvk.get_swapchain_create_info().image_extent.width as i32,
                            y: self.bvk.get_swapchain_create_info().image_extent.height as i32,
                            z: 1,
                        },
                    ])
                    .dst_subresource(vk::ImageSubresourceLayers {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        mip_level: 0,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .dst_offsets([
                        vk::Offset3D { x: 0, y: 0, z: 0 },
                        vk::Offset3D {
                            x: self.bvk.get_swapchain_create_info().image_extent.width as i32,
                            y: self.bvk.get_swapchain_create_info().image_extent.height as i32,
                            z: 1,
                        },
                    ]);

                let image_blit_info = vk::BlitImageInfo2::builder()
                    .src_image(self.rendering_layer.get_output_image())
                    .src_image_layout(vk::ImageLayout::GENERAL)
                    .dst_image(self.bvk.get_swapchain_images()[i])
                    .dst_image_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                    .regions(std::slice::from_ref(&region))
                    .filter(vk::Filter::LINEAR);
                self.device.cmd_blit_image2(cb, &image_blit_info);
                self.device.end_command_buffer(cb);
            }
        }
    }

    pub fn render_frame(&mut self, window: &winit::window::Window) {
        let current_data_idx = self.rendered_frames as usize % self.frames_data.len();
        self.record_frame_data(current_data_idx);

        let current_frame_data = &self.frames_data[current_data_idx];

        let swapchain_image_idx = unsafe {
            let res = self.bvk.get_swapchain_fn().acquire_next_image(
                self.bvk.get_swapchain().unwrap(),
                u64::MAX,
                current_frame_data.semaphores[0],
                vk::Fence::null(),
            );
            if res.is_err() || res.unwrap().1 {
                panic!("Fucking hell");
            }
            self.device.wait_for_fences(
                std::slice::from_ref(&current_frame_data.after_exec_fence),
                false,
                u64::MAX,
            );
            self.device
                .reset_fences(std::slice::from_ref(&current_frame_data.after_exec_fence));
            res.unwrap().0
        };

        let wait_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
            .semaphore(current_frame_data.semaphores[0])
            .stage_mask(vk::PipelineStageFlags2KHR::BLIT)
            .device_index(0);
        let command_submit_info = vk::CommandBufferSubmitInfoKHR::builder()
            .command_buffer(current_frame_data.command_buffers[swapchain_image_idx as usize])
            .device_mask(0);
        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
            .semaphore(current_frame_data.semaphores[1])
            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
            .device_index(0);
        let submit_info = vk::SubmitInfo2KHR::builder()
            .wait_semaphore_infos(std::slice::from_ref(&wait_semaphore_submit_info))
            .command_buffer_infos(std::slice::from_ref(&command_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
            .build();
        unsafe {
            self.device
                .queue_submit2(
                    self.bvk.get_queues()[0],
                    std::slice::from_ref(&submit_info),
                    current_frame_data.after_exec_fence,
                )
                .expect("Error submitting queue");
        }
        let swapchain = self.bvk.get_swapchain().unwrap();
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(&current_frame_data.semaphores[1]))
            .swapchains(std::slice::from_ref(&swapchain))
            .image_indices(std::slice::from_ref(&swapchain_image_idx));

        unsafe {
            self.bvk
                .get_swapchain_fn()
                .queue_present(self.bvk.get_queues()[0], &present_info)
                .expect("Queue present failed");
        }
        self.rendered_frames += 1;
    }
}
