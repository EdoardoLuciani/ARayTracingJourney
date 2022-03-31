use super::vk_allocator::*;
use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::*;
use std::ops::Bound::*;
use std::rc::Rc;

struct UsedBlock {
    size: u64,
    po2_alignment_increment: u64,
}

struct BufferUnitData {
    allocation: BufferAllocation,
    // red-black binary tree to keep size|offset
    free_blocks: BTreeMap<u64, u64>,
    // hashmap to keep address|used_block
    used_blocks: HashMap<u64, UsedBlock>,
}

pub struct VkBuffersSubAllocator {
    allocator: Rc<RefCell<VkMemoryResourceAllocator>>,
    buffers_usage: vk::BufferUsageFlags,
    buffers_location: MemoryLocation,
    blocks_initial_size: usize,
    min_allocation_size: usize,
    buffer_units: HashMap<vk::Buffer, BufferUnitData>,
}

impl VkBuffersSubAllocator {
    pub fn new(
        allocator: Rc<RefCell<VkMemoryResourceAllocator>>,
        buffers_usage: vk::BufferUsageFlags,
        buffers_location: MemoryLocation,
        blocks_initial_size: usize,
        min_allocation_size: usize,
    ) -> Self {
        VkBuffersSubAllocator {
            allocator,
            buffers_usage,
            buffers_location,
            blocks_initial_size: blocks_initial_size.next_power_of_two(),
            min_allocation_size: min_allocation_size.next_power_of_two(),
            buffer_units: Default::default(),
        }
    }

    // the block that is passed (old_block_size, old_block_address) is assumed to be already deleted from the map
    fn split_block_recursive(
        buffer_free_blocks: &mut BTreeMap<u64, u64>,
        old_block_size: u64,
        old_block_address: u64,
        requested_block_size: u64,
    ) -> (u64, u64) {
        // create the right block
        let new_block_size = old_block_size / 2;
        buffer_free_blocks.insert(new_block_size, old_block_address + new_block_size);
        if new_block_size != requested_block_size {
            // continuing to subdivide the left block without actually creating it
            return Self::split_block_recursive(
                buffer_free_blocks,
                new_block_size,
                old_block_address,
                requested_block_size,
            );
        }
        // on the last step we return the data of the left block, but we do not create it, since it is going to be removed shortly after
        (new_block_size, old_block_address)
    }

    fn merge_block_recursive(
        buffer_free_blocks: &mut BTreeMap<u64, u64>,
        block_address: u64,
        block_size: u64,
    ) {
        let left_block_address = (block_address - block_size) as i64;
        let right_block_address = (block_address + block_size) as i64;

        //for candidate_block in buffer_free_blocks.range(Included(block_size), Excluded(block_size));
        /*
        auto its = buffer_free_blocks.equal_range(address_size_source_block.second);
        for (auto it = its.first; it != its.second; it++) {
            if (it->second == left_adjacent_block_address) {
                buffer_free_blocks.erase(it);
                // The new block has the left adjacent block as the address
                return merge_blocks_recursive(buffer_free_blocks, {left_adjacent_block_address, address_size_source_block.second*2});
            }
            else if (it->second == right_adjacent_block_address) {
                buffer_free_blocks.erase(it);
                // The new block has the source block address as the address
                return merge_blocks_recursive(buffer_free_blocks, {address_size_source_block.first, address_size_source_block.second*2});
            }
        }
        // When there are no other blocks which can be joined, insert the merged block into the free ones
        buffer_free_blocks.emplace(address_size_source_block.second, address_size_source_block.first);
        return;
        */
    }

    fn request_next_buffer(&mut self, buffer_size: usize) {
        // All blocks needs to be have size to a power of 2
        let buffer_size =
            (std::cmp::max(buffer_size, self.min_allocation_size)).next_power_of_two();

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size as u64)
            .usage(self.buffers_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer_allocation = RefCell::borrow_mut(self.allocator.borrow_mut())
            .allocate_buffer(&buffer_create_info, self.buffers_location);

        let buffer = buffer_allocation.buffer;
        let buffer_unit_data = BufferUnitData {
            allocation: buffer_allocation,
            free_blocks: BTreeMap::from([(buffer_size as u64, 0)]),
            used_blocks: Default::default(),
        };
        self.buffer_units.insert(buffer, buffer_unit_data);
    }
}
