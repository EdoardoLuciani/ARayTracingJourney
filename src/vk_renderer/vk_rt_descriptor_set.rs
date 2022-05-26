use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::rc::Rc;
use ash::vk;
use itertools::all;
use crate::vk_renderer::vk_allocator::vk_buffers_suballocator::SubAllocationData;
use crate::vk_renderer::vk_allocator::vk_descriptor_sets_allocator::DescriptorSetAllocation;
use crate::vk_renderer::vk_allocator::VkAllocator;

pub struct VkRTDescriptorSet {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    model_info_host_allocation: SubAllocationData,
    model_info_device_allocation: SubAllocationData,
    model_info_bytes_occupied: u64,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,

}

const DESCRIPTOR_SET_TLAS_BINDING: u32 = 0;
const DESCRIPTOR_SET_MODEL_INFO_BUFFER_BINDING: u32 = 1;

struct ModelInfo {
    vertices_device_address: u64,
    indices_device_address: u64,
}

impl VkRTDescriptorSet {
    pub fn new(device: Rc<ash::Device>, allocator: Rc<RefCell<VkAllocator>>) -> Self {
        let descriptor_set_layout = unsafe {
            let descriptor_set_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_TLAS_BINDING)
                    .descriptor_type(vk::DescriptorType::ACCELERATION_STRUCTURE_KHR)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(DESCRIPTOR_SET_MODEL_INFO_BUFFER_BINDING)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::RAYGEN_KHR)
                    .build(),
            ];
            let binding_flags = [
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
                vk::DescriptorBindingFlags::UPDATE_AFTER_BIND,
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
        let descriptor_set_allocation = allocator.as_ref().borrow_mut().get_descriptor_set_allocator_mut().allocate_descriptor_sets(&[descriptor_set_layout]);

        let model_info_host_allocation = allocator.as_ref().borrow_mut().get_host_uniform_sub_allocator_mut().allocate(std::mem::size_of::<ModelInfo>() * 16, 1);
        let model_info_device_allocation = allocator.as_ref().borrow_mut().get_device_uniform_sub_allocator_mut().allocate(std::mem::size_of::<ModelInfo>() * 16, 1);

        VkRTDescriptorSet {
            device,
            allocator,
            model_info_host_allocation,
            model_info_device_allocation,
            model_info_bytes_occupied: 0,
            descriptor_set_layout,
            descriptor_set_allocation
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

    pub fn add_model_data(&mut self) {
        todo!();
    }
}