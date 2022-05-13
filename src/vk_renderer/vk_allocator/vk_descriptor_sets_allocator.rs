use ash::vk;
use std::ops::Deref;

pub struct VkDescriptorSetsAllocator {
    device: std::rc::Rc<ash::Device>,
    descriptor_pool_flags: vk::DescriptorPoolCreateFlags,
    descriptor_pool_max_sets: u32,
    descriptor_pool_sizes: Vec<vk::DescriptorPoolSize>,
    pools: Vec<vk::DescriptorPool>,
}

pub struct DescriptorSetAllocation {
    descriptor_sets: Vec<vk::DescriptorSet>,
    descriptor_pool: vk::DescriptorPool,
}

impl DescriptorSetAllocation {
    pub fn get_descriptor_sets(&self) -> &[vk::DescriptorSet] {
        &self.descriptor_sets
    }
}

impl VkDescriptorSetsAllocator {
    pub fn new(
        device: std::rc::Rc<ash::Device>,
        descriptor_pool_flags: vk::DescriptorPoolCreateFlags,
        descriptor_pool_max_sets: u32,
        descriptor_pool_sizes: Vec<vk::DescriptorPoolSize>,
    ) -> VkDescriptorSetsAllocator {
        VkDescriptorSetsAllocator {
            device,
            descriptor_pool_flags: descriptor_pool_flags
                | vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET,
            descriptor_pool_max_sets,
            descriptor_pool_sizes,
            pools: Vec::default(),
        }
    }

    pub fn allocate_descriptor_sets(
        &mut self,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> DescriptorSetAllocation {
        let mut descriptor_set_allocate_info =
            vk::DescriptorSetAllocateInfo::builder().set_layouts(set_layouts);
        let mut alloc = |device: &ash::Device, pool: vk::DescriptorPool| {
            descriptor_set_allocate_info.descriptor_pool = pool;
            unsafe { device.allocate_descriptor_sets(&descriptor_set_allocate_info) }
        };

        // iter in the existing pools to search for a viable pool
        for pool in self.pools.iter().copied() {
            if let Ok(sets) = alloc(self.device.deref(), pool) {
                return DescriptorSetAllocation {
                    descriptor_sets: sets,
                    descriptor_pool: pool,
                };
            }
        }

        // if all pools are exhausted create a new one
        let pool = self.create_pool();
        DescriptorSetAllocation {
            descriptor_sets: alloc(self.device.deref(), pool).expect("Empty created pool is not viable for the requested sets layout, consider changing constructor flags"),
            descriptor_pool: pool,
        }
    }

    pub fn free_descriptor_sets(&mut self, descriptor_set_allocation: DescriptorSetAllocation) {
        unsafe {
            self.device
                .free_descriptor_sets(
                    descriptor_set_allocation.descriptor_pool,
                    &descriptor_set_allocation.descriptor_sets,
                )
                .unwrap();
        }
    }

    fn create_pool(&mut self) -> vk::DescriptorPool {
        let descriptor_pool_create_info = vk::DescriptorPoolCreateInfo::builder()
            .flags(self.descriptor_pool_flags)
            .max_sets(self.descriptor_pool_max_sets)
            .pool_sizes(&self.descriptor_pool_sizes);
        let descriptor_pool = unsafe {
            self.device
                .create_descriptor_pool(&descriptor_pool_create_info, None)
                .unwrap()
        };
        self.pools.push(descriptor_pool);
        descriptor_pool
    }
}

impl Drop for VkDescriptorSetsAllocator {
    fn drop(&mut self) {
        for pool in self.pools.iter().copied() {
            unsafe {
                self.device.destroy_descriptor_pool(pool, None);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vk_renderer::vk_boot::vk_base::VkBase;

    #[test]
    fn vk_descriptors_allocator_overload_max_sets() {
        let physical_device_features2 = vk::PhysicalDeviceFeatures2::default();
        let bvk = VkBase::new(
            "",
            &[],
            &[],
            &physical_device_features2,
            &[(vk::QueueFlags::GRAPHICS, 1.0f32)],
            None,
        );
        let device = std::rc::Rc::new(bvk.get_device().clone());

        let descriptor_pool_sizes = [
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::UNIFORM_BUFFER,
                descriptor_count: 100,
            },
            vk::DescriptorPoolSize {
                ty: vk::DescriptorType::COMBINED_IMAGE_SAMPLER,
                descriptor_count: 100,
            },
        ];
        let mut descriptor_set_allocator = VkDescriptorSetsAllocator::new(
            device.clone(),
            vk::DescriptorPoolCreateFlags::empty(),
            2,
            descriptor_pool_sizes.to_vec(),
        );

        let descriptor_set_layout = unsafe {
            let descriptor_set_layout_binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(vk::ShaderStageFlags::ALL_GRAPHICS);
            let descriptor_set_layout_create_info = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(std::slice::from_ref(&descriptor_set_layout_binding));
            bvk.get_device()
                .create_descriptor_set_layout(&descriptor_set_layout_create_info, None)
                .unwrap()
        };

        let allocation0 = descriptor_set_allocator
            .allocate_descriptor_sets(&[descriptor_set_layout, descriptor_set_layout]);
        let allocation1 = descriptor_set_allocator
            .allocate_descriptor_sets(&[descriptor_set_layout, descriptor_set_layout]);

        assert_ne!(allocation0.descriptor_pool, allocation1.descriptor_pool);
        assert_eq!(descriptor_set_allocator.pools.len(), 2);

        descriptor_set_allocator.free_descriptor_sets(allocation0);
        descriptor_set_allocator.free_descriptor_sets(allocation1);

        assert_eq!(descriptor_set_allocator.pools.len(), 2);
    }
}
