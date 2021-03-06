use super::vk_allocator::VkAllocator;
use super::vk_boot::vk_base;
use super::vk_model::VkModel;
use super::vk_rendering_layers::vk_rt_lightning_shadows::VkRTLightningShadows;
use super::vk_rt_descriptor_set::VkRTDescriptorSet;
use super::vk_tlas_builder::VkTlasBuilder;
use crate::vk_renderer::vk_blas_builder::VkBlasBuilder;
use crate::vk_renderer::vk_camera::VkCamera;
use crate::vk_renderer::vk_rt_descriptor_set::DescriptorSetPrimitiveInfo;
use ash::{extensions::*, vk};
use nalgebra::*;
use std::cell::RefCell;
use std::mem::ManuallyDrop;
use std::rc::Rc;

struct CommandRecordInfo {
    device: Rc<ash::Device>,
    command_pool: vk::CommandPool,
    command_buffers: Vec<vk::CommandBuffer>,
}

impl CommandRecordInfo {
    fn new(device: Rc<ash::Device>, queue_family_index: u32, command_buffers_count: u32) -> Self {
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

        CommandRecordInfo {
            device,
            command_pool,
            command_buffers,
        }
    }
}

impl Drop for CommandRecordInfo {
    fn drop(&mut self) {
        unsafe {
            self.device.destroy_command_pool(self.command_pool, None);
        }
    }
}

struct FrameData {
    device: Rc<ash::Device>,
    tlas_builder: VkTlasBuilder,
    semaphores: Vec<vk::Semaphore>,
    after_exec_fence: vk::Fence,
    main_cri: CommandRecordInfo,
    presentation_cri: CommandRecordInfo,
}

impl FrameData {
    fn new(
        device: Rc<ash::Device>,
        tlas_builder: VkTlasBuilder,
        queue_family_index: u32,
        semaphores_count: u32,
        presentation_command_buffers_count: u32,
    ) -> Self {
        let semaphores = (0..semaphores_count)
            .map(|_| {
                let semaphore_ci = vk::SemaphoreCreateInfo::default();
                unsafe { device.create_semaphore(&semaphore_ci, None).unwrap() }
            })
            .collect::<Vec<vk::Semaphore>>();

        let mut fd = FrameData {
            device: device.clone(),
            tlas_builder,
            semaphores,
            after_exec_fence: vk::Fence::null(),
            main_cri: CommandRecordInfo::new(device.clone(), queue_family_index, 1),
            presentation_cri: CommandRecordInfo::new(
                device,
                queue_family_index,
                presentation_command_buffers_count,
            ),
        };
        fd.recreate_fence();
        fd
    }

    fn recreate_fence(&mut self) {
        unsafe {
            self.device.destroy_fence(self.after_exec_fence, None);
        }

        self.after_exec_fence = unsafe {
            let fence_ci = vk::FenceCreateInfo::builder().flags(vk::FenceCreateFlags::SIGNALED);
            self.device.create_fence(&fence_ci, None).unwrap()
        };
    }
}

impl Drop for FrameData {
    fn drop(&mut self) {
        unsafe {
            for semaphore in self.semaphores.iter() {
                self.device.destroy_semaphore(*semaphore, None);
            }
            self.device.destroy_fence(self.after_exec_fence, None);
        }
    }
}

pub struct VulkanTempleRayTracedRenderer {
    device: Rc<ash::Device>,
    acceleration_structure_fp: Rc<khr::AccelerationStructure>,
    ray_tracing_pipeline_fp: Rc<khr::RayTracingPipeline>,
    allocator: Rc<RefCell<VkAllocator>>,
    tlas_builder: Rc<VkBlasBuilder>,
    models: Vec<VkModel>,
    camera: VkCamera,
    rt_descriptor_set: VkRTDescriptorSet,
    rendered_frames: u64,
    rendering_layer: VkRTLightningShadows,
    frames_data: [FrameData; 3],
    bvk: vk_base::VkBase,
}

impl VulkanTempleRayTracedRenderer {
    pub fn new(
        window_size: vk::Extent2D,
        window_handle: raw_window_handle::RawWindowHandle,
    ) -> Self {
        let mut physical_device_ray_tracing_pipeline =
            vk::PhysicalDeviceRayTracingPipelineFeaturesKHR::builder().ray_tracing_pipeline(true);
        let mut physical_device_acceleration_structure =
            vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true)
                .descriptor_binding_acceleration_structure_update_after_bind(true);
        let vulkan_features = vk::PhysicalDeviceFeatures::builder()
            .sampler_anisotropy(true)
            .shader_int64(true)
            .shader_int16(true)
            .build();
        let mut vulkan_11_features =
            vk::PhysicalDeviceVulkan11Features::builder().storage_buffer16_bit_access(true);
        let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::builder()
            .buffer_device_address(true)
            .descriptor_binding_storage_buffer_update_after_bind(true)
            .descriptor_binding_sampled_image_update_after_bind(true)
            .descriptor_binding_partially_bound(true)
            .runtime_descriptor_array(true)
            .shader_sampled_image_array_non_uniform_indexing(true);
        let mut vulkan_13_features =
            vk::PhysicalDeviceVulkan13Features::builder().synchronization2(true);
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::builder()
            .features(vulkan_features)
            .push_next(&mut physical_device_acceleration_structure)
            .push_next(&mut physical_device_ray_tracing_pipeline)
            .push_next(&mut vulkan_11_features)
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
            window_size,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
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

        let tlas_builder = Rc::new(VkBlasBuilder::new(
            device.clone(),
            acceleration_structure_fp.clone(),
            allocator.clone(),
        ));

        let camera = VkCamera::new(
            device.clone(),
            allocator.clone(),
            Vector3::from_element(0.0f32),
            Vector3::new(0.0f32, 0.0f32, 1.0f32),
            window_size.width as f32 / window_size.height as f32,
            nalgebra::RealField::frac_pi_2(),
            0.1f32,
            1000f32,
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

        let rt_descriptor_set = VkRTDescriptorSet::new(device.clone(), allocator.clone());

        let init_cri = CommandRecordInfo::new(device.clone(), bvk.get_queue_family_index(), 1);
        let init_cb = init_cri.command_buffers[0];
        unsafe {
            let begin_info = vk::CommandBufferBeginInfo::builder()
                .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);
            device.begin_command_buffer(init_cb, &begin_info).unwrap();
        }

        let rendering_layer = VkRTLightningShadows::new(
            device.clone(),
            ray_tracing_pipeline_fp.clone(),
            &ray_tracing_pipeline_properties,
            allocator.clone(),
            std::path::Path::new("assets//shaders-spirv"),
            &[
                rt_descriptor_set.descriptor_set_layout(),
                camera.descriptor_set_layout(),
            ],
            window_size,
            vk::Format::R8G8B8A8_UNORM,
            init_cb,
        );

        let frames_data: [FrameData; 3] = (0..3)
            .map(|_| {
                let tlas_builder = VkTlasBuilder::new(
                    device.clone(),
                    acceleration_structure_fp.clone(),
                    allocator.clone(),
                );
                FrameData::new(
                    device.clone(),
                    tlas_builder,
                    bvk.get_queue_family_index(),
                    3,
                    bvk.get_swapchain_image_views().len() as u32,
                )
            })
            .collect::<Vec<_>>()
            .try_into()
            .ok()
            .unwrap();

        let mut rtr = VulkanTempleRayTracedRenderer {
            bvk,
            device,
            acceleration_structure_fp,
            ray_tracing_pipeline_fp,
            tlas_builder,
            allocator,
            models: Vec::default(),
            camera,
            rt_descriptor_set,
            rendering_layer,
            frames_data,
            rendered_frames: 0,
        };

        unsafe {
            rtr.device.end_command_buffer(init_cb).unwrap();
        }
        rtr.submit_and_wait(init_cb, rtr.bvk.get_queues()[0]);

        for i in 0..rtr.frames_data.len() {
            rtr.record_static_command_buffers(i);
        }

        rtr
    }

    pub fn add_model(&mut self, file_path: &std::path::Path, model_matrix: Matrix3x4<f32>) {
        self.models.push(VkModel::new(
            self.device.clone(),
            self.allocator.clone(),
            Some(self.tlas_builder.clone()),
            file_path.to_path_buf(),
            model_matrix,
        ));
    }

    pub fn prepare_first_frame(&mut self) {
        let frame_data_idx = self.rendered_frames as usize % self.frames_data.len();
        unsafe {
            self.device.reset_fences(std::slice::from_ref(
                &self.frames_data[frame_data_idx].after_exec_fence,
            ));
            self.record_main_command(frame_data_idx);
        }
    }

    pub fn render_frame(&mut self, window: &winit::window::Window) {
        let current_frame_idx = self.rendered_frames as usize % self.frames_data.len();

        self.camera.update_host_buffer();

        let swapchain_image_idx = unsafe {
            let res = self.bvk.get_swapchain_fn().acquire_next_image(
                self.bvk.get_swapchain().unwrap(),
                u64::MAX,
                self.frames_data[current_frame_idx].semaphores[0],
                vk::Fence::null(),
            );
            if res.is_err() || res.unwrap().1 {
                self.resize(vk::Extent2D {
                    width: window.inner_size().width,
                    height: window.inner_size().height,
                });
                return;
            }
            res.unwrap().0
        };

        let command_submit_info = vk::CommandBufferSubmitInfoKHR::builder()
            .command_buffer(self.frames_data[current_frame_idx].main_cri.command_buffers[0])
            .device_mask(0);
        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
            .semaphore(self.frames_data[current_frame_idx].semaphores[1])
            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
            .device_index(0);
        let submit_info0 = vk::SubmitInfo2KHR::builder()
            .wait_semaphore_infos(&[])
            .command_buffer_infos(std::slice::from_ref(&command_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
            .build();

        let wait_semaphore_submit_infos = [
            vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(self.frames_data[current_frame_idx].semaphores[0])
                .stage_mask(vk::PipelineStageFlags2KHR::BLIT)
                .device_index(0)
                .build(),
            vk::SemaphoreSubmitInfoKHR::builder()
                .semaphore(self.frames_data[current_frame_idx].semaphores[1])
                .stage_mask(vk::PipelineStageFlags2KHR::BLIT)
                .device_index(0)
                .build(),
        ];
        let command_submit_info = vk::CommandBufferSubmitInfoKHR::builder()
            .command_buffer(
                self.frames_data[current_frame_idx]
                    .presentation_cri
                    .command_buffers[swapchain_image_idx as usize],
            )
            .device_mask(0);
        let signal_semaphore_submit_info = vk::SemaphoreSubmitInfoKHR::builder()
            .semaphore(self.frames_data[current_frame_idx].semaphores[2])
            .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS)
            .device_index(0);
        let submit_info1 = vk::SubmitInfo2KHR::builder()
            .wait_semaphore_infos(&wait_semaphore_submit_infos)
            .command_buffer_infos(std::slice::from_ref(&command_submit_info))
            .signal_semaphore_infos(std::slice::from_ref(&signal_semaphore_submit_info))
            .build();
        unsafe {
            self.device
                .queue_submit2(
                    self.bvk.get_queues()[0],
                    &[submit_info0, submit_info1],
                    self.frames_data[current_frame_idx].after_exec_fence,
                )
                .expect("Error submitting queue");
        }

        // start of next frame recording
        let next_frame_idx = (self.rendered_frames as usize + 1) % self.frames_data.len();
        unsafe {
            self.device
                .wait_for_fences(
                    std::slice::from_ref(&self.frames_data[next_frame_idx].after_exec_fence),
                    false,
                    u64::MAX,
                )
                .unwrap();
            self.device
                .reset_fences(std::slice::from_ref(
                    &self.frames_data[next_frame_idx].after_exec_fence,
                ))
                .unwrap();
        }
        self.record_main_command(next_frame_idx);

        let swapchain = self.bvk.get_swapchain().unwrap();
        let present_info = vk::PresentInfoKHR::builder()
            .wait_semaphores(std::slice::from_ref(
                &self.frames_data[current_frame_idx].semaphores[2],
            ))
            .swapchains(std::slice::from_ref(&swapchain))
            .image_indices(std::slice::from_ref(&swapchain_image_idx));
        unsafe {
            let res = self
                .bvk
                .get_swapchain_fn()
                .queue_present(self.bvk.get_queues()[0], &present_info);
            self.rendered_frames += 1;
            if res.is_err() || res.unwrap() {
                self.resize(vk::Extent2D {
                    width: window.inner_size().width,
                    height: window.inner_size().height,
                });
            }
        }

        if self
            .models
            .iter()
            .any(|m| m.needs_command_buffer_submission())
        {
            unsafe {
                self.device
                    .wait_for_fences(
                        std::slice::from_ref(&self.frames_data[current_frame_idx].after_exec_fence),
                        false,
                        u64::MAX,
                    )
                    .unwrap();
            }
        }
        self.models
            .iter_mut()
            .for_each(|m| m.reset_command_buffer_submission_status());
    }

    pub fn camera_mut(&mut self) -> &mut VkCamera {
        &mut self.camera
    }

    pub fn models_mut(&mut self) -> &mut [VkModel] {
        &mut self.models
    }

    fn resize(&mut self, window_resolution: vk::Extent2D) {
        let rendering_resolution = window_resolution;
        let presentation_resolution = rendering_resolution;

        unsafe {
            self.device.device_wait_idle();
        }
        self.bvk.recreate_swapchain(
            vk::PresentModeKHR::FIFO,
            presentation_resolution,
            vk::ImageUsageFlags::TRANSFER_DST | vk::ImageUsageFlags::STORAGE,
            vk::SurfaceFormatKHR {
                format: vk::Format::B8G8R8A8_UNORM,
                color_space: vk::ColorSpaceKHR::SRGB_NONLINEAR,
            },
        );
        self.rendering_layer
            .resize(rendering_resolution, vk::Format::R8G8B8A8_UNORM);

        for i in 0..self.frames_data.len() {
            self.frames_data[i].recreate_fence();
            self.record_static_command_buffers(i);
        }
        self.prepare_first_frame();
    }

    fn record_static_command_buffers(&self, frame_data_idx: usize) {
        let frame_data = &self.frames_data[frame_data_idx];
        unsafe {
            self.device.reset_command_pool(
                frame_data.presentation_cri.command_pool,
                vk::CommandPoolResetFlags::empty(),
            );
        }
        for (i, cb) in frame_data
            .presentation_cri
            .command_buffers
            .iter()
            .copied()
            .enumerate()
        {
            unsafe {
                let command_buffer_bi = vk::CommandBufferBeginInfo::default();
                self.device.begin_command_buffer(cb, &command_buffer_bi);
            }
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
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
                    })
                    .build(),
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::NONE)
                    .src_access_mask(vk::AccessFlags2::NONE)
                    .dst_stage_mask(vk::PipelineStageFlags2::BLIT)
                    .dst_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                    .old_layout(vk::ImageLayout::UNDEFINED)
                    .new_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                    .image(self.bvk.get_swapchain_images()[i])
                    .subresource_range(vk::ImageSubresourceRange {
                        aspect_mask: vk::ImageAspectFlags::COLOR,
                        base_mip_level: 0,
                        level_count: 1,
                        base_array_layer: 0,
                        layer_count: 1,
                    })
                    .build(),
            ];
            let dependency_info =
                vk::DependencyInfo::builder().image_memory_barriers(&image_memory_barriers);
            unsafe {
                self.device.cmd_pipeline_barrier2(cb, &dependency_info);
            }

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
                .dst_image_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .regions(std::slice::from_ref(&region))
                .filter(vk::Filter::LINEAR);
            unsafe {
                self.device.cmd_blit_image2(cb, &image_blit_info);
            }
            let image_memory_barrier = vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::BLIT)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::NONE)
                .dst_access_mask(vk::AccessFlags2::NONE)
                .old_layout(vk::ImageLayout::TRANSFER_DST_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .image(self.bvk.get_swapchain_images()[i])
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build();
            let dependency_info = vk::DependencyInfo::builder()
                .image_memory_barriers(std::slice::from_ref(&image_memory_barrier));
            unsafe {
                self.device.cmd_pipeline_barrier2(cb, &dependency_info);
                self.device.end_command_buffer(cb).unwrap();
            }
        }
    }

    fn record_main_command(&mut self, frame_data_idx: usize) {
        let frame_data = &mut self.frames_data[frame_data_idx];

        unsafe {
            self.device
                .reset_command_pool(
                    frame_data.main_cri.command_pool,
                    vk::CommandPoolResetFlags::empty(),
                )
                .unwrap();
        }
        let cb = frame_data.main_cri.command_buffers[0];

        unsafe {
            let command_buffer_begin_info = vk::CommandBufferBeginInfo::default();
            self.device
                .begin_command_buffer(cb, &command_buffer_begin_info)
                .unwrap();
        }

        let mut blas_buffer_memory_barriers = Vec::<vk::BufferMemoryBarrier2>::new();
        for model in self.models.iter_mut() {
            model.update_model_status(&self.camera.pos(), cb);
            if let Some(blas_buffer) = model.get_blas_buffer() {
                blas_buffer_memory_barriers.push(
                    vk::BufferMemoryBarrier2::builder()
                        .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
                        .dst_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
                        .dst_access_mask(vk::AccessFlags2::SHADER_READ)
                        .buffer(blas_buffer)
                        .offset(0)
                        .size(vk::WHOLE_SIZE)
                        .build(),
                );
            }
        }
        let dependency_info =
            vk::DependencyInfo::builder().buffer_memory_barriers(&blas_buffer_memory_barriers);
        unsafe {
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }

        let mut instance_custom_index = 0;
        let as_instances = self
            .models
            .iter()
            .filter_map(|m| {
                let res = m.get_acceleration_structure_instance(instance_custom_index);
                instance_custom_index += m.get_device_primitives_count().unwrap_or(0);
                res
            })
            .collect::<Vec<_>>();
        frame_data.tlas_builder.recreate_tlas(cb, &as_instances);

        self.rt_descriptor_set
            .set_tlas(frame_data.tlas_builder.get_tlas().unwrap());

        let buffer_memory_barrier2 = vk::BufferMemoryBarrier2::builder()
            .src_stage_mask(vk::PipelineStageFlags2::ACCELERATION_STRUCTURE_BUILD_KHR)
            .src_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_WRITE_KHR)
            .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
            .dst_access_mask(vk::AccessFlags2::ACCELERATION_STRUCTURE_READ_KHR)
            .buffer(frame_data.tlas_builder.get_tlas_buffer().unwrap())
            .offset(0)
            .size(vk::WHOLE_SIZE);
        let dependency_info = vk::DependencyInfo::builder()
            .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barrier2));
        unsafe {
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }

        let mut primitives_info = Vec::new();
        for model in self.models.iter() {
            if let (Some(e1), Some(e2), Some(e3), Some(e4)) = (
                model.get_vertices_addresses(),
                model.get_indices_addresses(),
                model.get_image_views(),
                model.get_indices_size(),
            ) {
                itertools::izip!(e1, e2, e3, e4).for_each(|(e1, e2, e3, e4)| {
                    primitives_info.push(DescriptorSetPrimitiveInfo {
                        vertices_device_address: e1,
                        indices_device_address: e2,
                        single_index_size: e4,
                        image_view: e3,
                    });
                });
            }
        }
        self.rt_descriptor_set
            .set_primitives_info(&primitives_info, cb);

        self.rendering_layer.trace_rays(
            cb,
            &[
                self.rt_descriptor_set.descriptor_set(),
                self.camera.descriptor_set(),
            ],
        );

        unsafe {
            self.device.end_command_buffer(cb).unwrap();
        }
    }

    fn submit_and_wait(&self, cb: vk::CommandBuffer, queue: vk::Queue) {
        let command_buffers = vk::CommandBufferSubmitInfo::builder()
            .command_buffer(cb)
            .device_mask(0);
        let submit_info =
            vk::SubmitInfo2::builder().command_buffer_infos(std::slice::from_ref(&command_buffers));
        unsafe {
            self.device
                .queue_submit2(queue, std::slice::from_ref(&submit_info), vk::Fence::null())
                .unwrap();
            self.device.queue_wait_idle(queue).unwrap();
        }
    }
}

impl Drop for VulkanTempleRayTracedRenderer {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
        }
    }
}
