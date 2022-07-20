use crate::vk_renderer::vk_allocator::vk_buffers_suballocator::{
    SubAllocationData, VkBuffersSubAllocator,
};
use crate::vk_renderer::vk_allocator::vk_descriptor_sets_allocator::DescriptorSetAllocation;
use crate::vk_renderer::vk_allocator::VkAllocator;
use ash::vk;
use std::cell::RefCell;
use std::ops::DerefMut;
use std::rc::Rc;

pub struct VkRTDescriptorSet {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    image_sampler: vk::Sampler,
    model_info_host_allocation: SubAllocationData,
    model_info_device_allocation: SubAllocationData,
    model_info_bytes_occupied: u64,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
}

const DESCRIPTOR_SET_TLAS_BINDING: u32 = 0;
const DESCRIPTOR_SET_PRIMITIVE_INFO_BUFFER_BINDING: u32 = 1;
const DESCRIPTOR_SET_PRIMITIVE_INFO_IMAGES_BINDING: u32 = 2;

pub struct DescriptorSetPrimitiveInfo {
    pub vertices_device_address: u64,
    pub indices_device_address: u64,
    pub single_index_size: u32,
    pub image_view: vk::ImageView,
}

#[derive(Clone, Copy)]
#[repr(C, packed)]
struct ShaderPrimitiveInfo {
    vertices_device_address: u64,
    indices_device_address: u64,
    texture_offset: u32,
    single_index_size: u32,
}

impl VkRTDescriptorSet {
    pub fn new(device: Rc<ash::Device>, allocator: Rc<RefCell<VkAllocator>>) -> Self {
        let image_sampler_ci = vk::SamplerCreateInfo::builder()
            .mag_filter(vk::Filter::LINEAR)
            .min_filter(vk::Filter::LINEAR)
            .mipmap_mode(vk::SamplerMipmapMode::LINEAR)
            .address_mode_u(vk::SamplerAddressMode::REPEAT)
            .address_mode_v(vk::SamplerAddressMode::REPEAT)
            .address_mode_w(vk::SamplerAddressMode::REPEAT)
            .mip_lod_bias(0.0f32)
            .anisotropy_enable(true)
            .max_anisotropy(16.0f32)
            .compare_enable(false)
            .min_lod(0.0f32)
            .max_lod(vk::LOD_CLAMP_NONE)
            .unnormalized_coordinates(false);
        let image_sampler = unsafe { device.create_sampler(&image_sampler_ci, None).unwrap() };

        let descriptor_set_layout = unsafe {
            let immutable_samplers = vec![image_sampler; 256];
            let descriptor_set_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_TLAS_BINDING)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_PRIMITIVE_INFO_BUFFER_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_PRIMITIVE_INFO_IMAGES_BINDING)
                    .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                    .descriptor_count(256)
                    .stage_flags(vk::ShaderStageFlags::CLOSEST_HIT_KHR)
                    .immutable_samplers(&immutable_samplers)
                    .build(),
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND
                    | vk::DescriptorBindingFlags::PARTIALLY_BOUND,
            ];
            let mut descriptor_set_layout_binding_flags_ci =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder()
                    .binding_flags(&binding_flags);
            let descriptor_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .push_next(&mut descriptor_set_layout_binding_flags_ci)
                .flags(vk::DescriptorSetLayoutCreateFlags::UPDATE_AFTER_BIND_POOL)
                .bindings(&descriptor_set_bindings);
            device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .unwrap()
        };
        let descriptor_set_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(&[descriptor_set_layout]);

        let model_info_host_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<ShaderPrimitiveInfo>() * 16, 1);
        let model_info_device_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_device_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<ShaderPrimitiveInfo>() * 16, 1);

        VkRTDescriptorSet {
            device,
            allocator,
            image_sampler,
            model_info_host_allocation,
            model_info_device_allocation,
            model_info_bytes_occupied: 0,
            descriptor_set_layout,
            descriptor_set_allocation,
        }
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.descriptor_set_allocation.get_descriptor_sets()[0]
    }

    pub fn set_tlas(&self, tlas: vk::AccelerationStructureKHR) {
        let mut write_descriptor_set_acceleration_structure =
            vk::WriteDescriptorSetAccelerationStructureKHR::builder()
                .acceleration_structures(std::slice::from_ref(&tlas));
        let mut descriptor_set_write = vk::WriteDescriptorSet::builder()
            .push_next(&mut write_descriptor_set_acceleration_structure)
            .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
            .dst_binding(DESCRIPTOR_SET_TLAS_BINDING)
            .dst_array_element(0)
            .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
            .build();
        descriptor_set_write.descriptor_count = 1;
        unsafe {
            self.device
                .update_descriptor_sets(&[descriptor_set_write], &[]);
        }
    }

    pub fn set_primitives_info(
        &mut self,
        primitive_infos: &[DescriptorSetPrimitiveInfo],
        cb: vk::CommandBuffer,
    ) {
        let shader_primitive_infos = primitive_infos
            .iter()
            .enumerate()
            .map(|(i, elem)| ShaderPrimitiveInfo {
                vertices_device_address: elem.vertices_device_address,
                indices_device_address: elem.indices_device_address,
                single_index_size: elem.single_index_size,
                texture_offset: i as u32,
            })
            .collect::<Vec<_>>();

        let required_buffer_size =
            shader_primitive_infos.len() * std::mem::size_of::<ShaderPrimitiveInfo>();
        if required_buffer_size > self.model_info_host_allocation.get_size() {
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_host_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.model_info_host_allocation,
                    unsafe { std::mem::zeroed() },
                ));
            self.model_info_host_allocation = self
                .allocator
                .as_ref()
                .borrow_mut()
                .get_host_uniform_sub_allocator_mut()
                .allocate(required_buffer_size, 1);

            self.allocator
                .as_ref()
                .borrow_mut()
                .get_device_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.model_info_device_allocation,
                    unsafe { std::mem::zeroed() },
                ));
            self.model_info_device_allocation = self
                .allocator
                .as_ref()
                .borrow_mut()
                .get_device_uniform_sub_allocator_mut()
                .allocate(required_buffer_size, 1);
        }

        let dst_slice = unsafe {
            std::slice::from_raw_parts_mut(
                self.model_info_host_allocation
                    .get_host_ptr()
                    .unwrap()
                    .as_ptr() as *mut ShaderPrimitiveInfo,
                shader_primitive_infos.len(),
            )
        };
        dst_slice.copy_from_slice(shader_primitive_infos.as_slice());

        self.update_primitives_info_descriptor_set(&primitive_infos);

        unsafe {
            let buffer_region = vk::BufferCopy2::builder()
                .src_offset(self.model_info_host_allocation.get_buffer_offset() as u64)
                .dst_offset(self.model_info_device_allocation.get_buffer_offset() as u64)
                .size(self.model_info_host_allocation.get_size() as u64);
            let copy_buffer_info2 = vk::CopyBufferInfo2::builder()
                .src_buffer(self.model_info_host_allocation.get_buffer())
                .dst_buffer(self.model_info_device_allocation.get_buffer())
                .regions(std::slice::from_ref(&buffer_region));
            self.device.cmd_copy_buffer2(cb, &copy_buffer_info2);

            let buffer_memory_barriers = vk::BufferMemoryBarrier2::builder()
                .src_stage_mask(vk::PipelineStageFlags2::COPY)
                .src_access_mask(vk::AccessFlags2::TRANSFER_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ)
                .buffer(self.model_info_device_allocation.get_buffer())
                .offset(self.model_info_device_allocation.get_buffer_offset() as u64)
                .size(self.model_info_device_allocation.get_size() as u64);
            let dependency_info = vk::DependencyInfo::builder()
                .buffer_memory_barriers(std::slice::from_ref(&buffer_memory_barriers));
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);
        }
    }

    fn update_primitives_info_descriptor_set(
        &self,
        primitives_info: &[DescriptorSetPrimitiveInfo],
    ) {
        let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
            .buffer(self.model_info_device_allocation.get_buffer())
            .offset(self.model_info_device_allocation.get_buffer_offset() as u64)
            .range((primitives_info.len() * std::mem::size_of::<ShaderPrimitiveInfo>()) as u64);

        let descriptor_image_infos = primitives_info
            .iter()
            .map(|elem| vk::DescriptorImageInfo {
                sampler: vk::Sampler::null(),
                image_view: elem.image_view,
                image_layout: vk::ImageLayout::SHADER_READ_ONLY_OPTIMAL,
            })
            .collect::<Vec<_>>();
        let descriptor_writes = [
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set())
                .dst_binding(DESCRIPTOR_SET_PRIMITIVE_INFO_BUFFER_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                .buffer_info(std::slice::from_ref(&descriptor_buffer_info))
                .build(),
            vk::WriteDescriptorSet::builder()
                .dst_set(self.descriptor_set())
                .dst_binding(DESCRIPTOR_SET_PRIMITIVE_INFO_IMAGES_BINDING)
                .dst_array_element(0)
                .descriptor_type(vk::DescriptorType::COMBINED_IMAGE_SAMPLER)
                .image_info(&descriptor_image_infos)
                .build(),
        ];
        unsafe {
            self.device.update_descriptor_sets(&descriptor_writes, &[]);
        }
    }
}

impl Drop for VkRTDescriptorSet {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.descriptor_set_allocation,
                    DescriptorSetAllocation::null(),
                ));
            self.device.destroy_sampler(self.image_sampler, None);
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_host_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.model_info_host_allocation,
                    std::mem::zeroed(),
                ));
            self.allocator
                .as_ref()
                .borrow_mut()
                .get_device_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.model_info_device_allocation,
                    std::mem::zeroed(),
                ));
        }
    }
}
