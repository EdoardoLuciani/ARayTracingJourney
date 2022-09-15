mod amd_fsr2_api_vk_bindings;

use super::super::vk_allocator::vk_memory_resource_allocator::*;
use super::super::vk_allocator::VkAllocator;
use amd_fsr2_api_vk_bindings::*;
use ash::{extensions::*, vk, Entry, Instance};
use gpu_allocator::MemoryLocation;
use nalgebra::*;
use std::cell::RefCell;
use std::ffi::c_void;
use std::rc::Rc;
use super::VkImagePrevState;

pub struct AmdFsr2 {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    physical_device: vk::PhysicalDevice,
    scratch_buffer: Vec<u8>,
    context_description: FfxFsr2ContextDescription,
    context: FfxFsr2Context,
    frame_index: u32,
    proj_jitter: Vector2<f32>,
    last_frame_time: std::time::Instant,
    input_color_image: vk::Image,
    input_color_image_view: vk::ImageView,
    input_depth_image: vk::Image,
    input_depth_image_view: vk::ImageView,
    input_motion_vector_image: vk::Image,
    input_motion_vector_image_view: vk::ImageView,
    output_image: ImageAllocation,
    output_image_view: vk::ImageView,
}

impl AmdFsr2 {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        physical_device: vk::PhysicalDevice,
        entry: &Entry,
        instance: &Instance,
        input_res: vk::Extent2D,
        output_res: vk::Extent2D,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_depth_image: vk::Image,
        input_depth_image_view: vk::ImageView,
        input_motion_vector_image: vk::Image,
        input_motion_vector_image_view: vk::ImageView,
    ) -> Self {
        let mut context_description = unsafe {
            FfxFsr2ContextDescription {
                flags: (FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_HIGH_DYNAMIC_RANGE
                    | FfxFsr2InitializationFlagBits_FFX_FSR2_ENABLE_AUTO_EXPOSURE)
                    as u32,
                maxRenderSize: FfxDimensions2D::default(),
                displaySize: FfxDimensions2D::default(),
                callbacks: std::mem::zeroed(),
                device: ffxGetDeviceVK(device.handle()),
            }
        };

        let scratch_buffer_size = unsafe {
            let pd = std::mem::transmute::<_, _>(physical_device);
            ffxFsr2GetScratchMemorySizeVK(
                pd,
                std::mem::transmute::<_, _>(
                    instance.fp_v1_0().enumerate_device_extension_properties,
                ),
            ) as usize
        };

        let mut scratch_buffer = vec![0u8; scratch_buffer_size];

        unsafe {
            ffxFsr2GetInterfaceVK(
                &mut context_description.callbacks as *mut FfxFsr2Interface,
                scratch_buffer.as_mut_ptr() as *mut c_void,
                scratch_buffer.len() as u64,
                std::mem::transmute::<_, _>(instance.handle()),
                std::mem::transmute::<_, _>(physical_device),
                std::mem::transmute::<_, _>(entry.static_fn().get_instance_proc_addr),
            );
        };

        let mut ret = unsafe {
            Self {
                device,
                allocator,
                physical_device,
                context_description,
                scratch_buffer,
                context: std::mem::zeroed(),
                frame_index: 0,
                proj_jitter: Vector2::zeros(),
                last_frame_time: std::time::Instant::now(),
                input_color_image,
                input_color_image_view,
                input_depth_image,
                input_depth_image_view,
                input_motion_vector_image,
                input_motion_vector_image_view,
                output_image: std::mem::zeroed(),
                output_image_view: vk::ImageView::null(),
            }
        };

        ret.resize(
            input_res,
            output_res,
            input_color_image,
            input_color_image_view,
            input_depth_image,
            input_depth_image_view,
            input_motion_vector_image,
            input_motion_vector_image_view,
        );

        ret
    }

    pub fn resize(
        &mut self,
        input_res: vk::Extent2D,
        output_res: vk::Extent2D,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_depth_image: vk::Image,
        input_depth_image_view: vk::ImageView,
        input_motion_vector_image: vk::Image,
        input_motion_vector_image_view: vk::ImageView,
    ) {
        unsafe {
            if self.context_description.maxRenderSize != FfxDimensions2D::default() {
                ffxFsr2ContextDestroy(&mut self.context);
            }

            self.context_description.maxRenderSize = FfxDimensions2D {
                width: input_res.width,
                height: input_res.height,
            };
            self.context_description.displaySize = FfxDimensions2D {
                width: output_res.width,
                height: output_res.height,
            };

            ffxFsr2ContextCreate(&mut self.context, &self.context_description);
        }
        take_mut::take(&mut self.output_image, |image| {
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(image);

            let image_ci = vk::ImageCreateInfo::builder()
                .image_type(vk::ImageType::TYPE_2D)
                .format(vk::Format::B10G11R11_UFLOAT_PACK32)
                .extent(vk::Extent3D {
                    width: output_res.width,
                    height: output_res.height,
                    depth: 1,
                })
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::STORAGE)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .initial_layout(vk::ImageLayout::UNDEFINED);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .allocate_image(&image_ci, MemoryLocation::GpuOnly)
        });

        unsafe {
            self.device.destroy_image_view(self.output_image_view, None);
            let image_view_ci = vk::ImageViewCreateInfo::builder()
                .image(self.output_image.get_image())
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(vk::Format::B10G11R11_UFLOAT_PACK32)
                .components(vk::ComponentMapping::default())
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                });
            self.output_image_view = self.device.create_image_view(&image_view_ci, None).unwrap();
        }

        self.input_color_image = input_color_image;
        self.input_color_image_view = input_color_image_view;
        self.input_depth_image = input_depth_image;
        self.input_depth_image_view = input_depth_image_view;
        self.input_motion_vector_image = input_motion_vector_image;
        self.input_motion_vector_image_view = input_motion_vector_image_view;
    }

    pub fn update_proj_jitter(&mut self) -> Vector2<f32> {
        unsafe {
            let phase_count = ffxFsr2GetJitterPhaseCount(
                self.context_description.maxRenderSize.width as i32,
                self.context_description.displaySize.width as i32,
            );
            ffxFsr2GetJitterOffset(
                &mut self.proj_jitter.x,
                &mut self.proj_jitter.y,
                self.frame_index as i32,
                phase_count,
            );
        }
        self.frame_index += 1;
        self.proj_jitter
    }

    pub fn get_output_image(&self) -> vk::Image {
        self.output_image.get_image()
    }

    pub fn get_output_image_view(&self) -> vk::ImageView {
        self.output_image_view
    }

    pub fn upscale(
        &mut self,
        cb: vk::CommandBuffer,
        camera_near: f32,
        camera_far: f32,
        camera_fovy: f32,
        reset: bool,
        input_color_prev_state: VkImagePrevState,
        input_motion_vector_prev_state: VkImagePrevState,
    ) {
        let image_memory_barriers = [
            vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(input_color_prev_state.src_stage)
                .src_access_mask(input_color_prev_state.src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(input_color_prev_state.src_layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.input_color_image)
                .subresource_range(vk::ImageSubresourceRange {
                    aspect_mask: vk::ImageAspectFlags::COLOR,
                    base_mip_level: 0,
                    level_count: 1,
                    base_array_layer: 0,
                    layer_count: 1,
                })
                .build(),
            vk::ImageMemoryBarrier2::builder()
                .src_stage_mask(input_motion_vector_prev_state.src_stage)
                .src_access_mask(input_motion_vector_prev_state.src_access)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_SAMPLED_READ)
                .old_layout(input_motion_vector_prev_state.src_layout)
                .new_layout(vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL)
                .image(self.input_motion_vector_image)
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
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::GENERAL)
                .image(self.output_image.get_image())
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
        unsafe { self.device.cmd_pipeline_barrier2(cb, &dependency_info) }

        unsafe {
            let dispatch_description = FfxFsr2DispatchDescription {
                commandList: ffxGetCommandListVK(std::mem::transmute(cb)),
                color: ffxGetTextureResourceVK(
                    &mut self.context,
                    self.input_color_image,
                    self.input_color_image_view,
                    self.context_description.maxRenderSize.width,
                    self.context_description.maxRenderSize.height,
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                depth: ffxGetTextureResourceVK(
                    &mut self.context,
                    self.input_depth_image,
                    self.input_depth_image_view,
                    self.context_description.maxRenderSize.width,
                    self.context_description.maxRenderSize.height,
                    vk::Format::R16_SFLOAT,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                motionVectors: ffxGetTextureResourceVK(
                    &mut self.context,
                    self.input_motion_vector_image,
                    self.input_motion_vector_image_view,
                    self.context_description.maxRenderSize.width,
                    self.context_description.maxRenderSize.height,
                    vk::Format::R16G16_SFLOAT,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                exposure: ffxGetTextureResourceVK(
                    &mut self.context,
                    vk::Image::null(),
                    vk::ImageView::null(),
                    1,
                    1,
                    vk::Format::UNDEFINED,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                reactive: ffxGetTextureResourceVK(
                    &mut self.context,
                    vk::Image::null(),
                    vk::ImageView::null(),
                    1,
                    1,
                    vk::Format::UNDEFINED,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                transparencyAndComposition: ffxGetTextureResourceVK(
                    &mut self.context,
                    vk::Image::null(),
                    vk::ImageView::null(),
                    1,
                    1,
                    vk::Format::UNDEFINED,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_COMPUTE_READ,
                ),
                output: ffxGetTextureResourceVK(
                    &mut self.context,
                    self.output_image.get_image(),
                    self.output_image_view,
                    self.context_description.displaySize.width,
                    self.context_description.displaySize.height,
                    vk::Format::B10G11R11_UFLOAT_PACK32,
                    std::ptr::null_mut(),
                    FfxResourceStates_FFX_RESOURCE_STATE_UNORDERED_ACCESS,
                ),
                jitterOffset: FfxFloatCoords2D {
                    x: self.proj_jitter.x,
                    y: self.proj_jitter.y,
                },
                motionVectorScale: FfxFloatCoords2D {
                    x: self.context_description.maxRenderSize.width as f32,
                    y: self.context_description.maxRenderSize.height as f32,
                },
                renderSize: self.context_description.maxRenderSize,
                enableSharpening: false,
                sharpness: 1.0f32,
                frameTimeDelta: self.last_frame_time.elapsed().as_secs_f32() * 1000.0f32,
                preExposure: 1.0f32,
                reset,
                cameraNear: camera_near,
                cameraFar: camera_far,
                cameraFovAngleVertical: camera_fovy,
            };
            ffxFsr2ContextDispatch(&mut self.context, &dispatch_description);
            self.last_frame_time = std::time::Instant::now();
        }
    }
}

impl Drop for AmdFsr2 {
    fn drop(&mut self) {
        unsafe {
            ffxFsr2ContextDestroy(&mut self.context);
            self.device.destroy_image_view(self.output_image_view, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_allocator_mut()
                .destroy_image(std::mem::replace(
                    &mut self.output_image,
                    std::mem::zeroed(),
                ))
        };
    }
}
