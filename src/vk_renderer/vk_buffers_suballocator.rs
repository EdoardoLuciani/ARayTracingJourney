use super::vk_allocator::*;
use ash::vk;
use gpu_allocator::{vulkan as vkalloc, MemoryLocation};
use num::Integer;
use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::collections::btree_map::Entry::Occupied;
use std::collections::*;
use std::ops::Bound::*;
use std::rc::Rc;

struct UsedBlock {
    size: u64,
    po2_alignment_increment: u64,
}

struct BufferUnitData {
    allocation: BufferAllocation,
    // tree to keep size|offset, the offset is a hashset to accommodate duplicate size values
    free_blocks: BTreeMap<usize, HashSet<usize>>,
    // hashmap to keep offset|used_block
    used_blocks: HashMap<usize, UsedBlock>,
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

    pub fn allocate(&mut self, size: usize, alignment: usize) {
        let floored_alignment = 1 << (usize::BITS - (alignment >> 1).leading_zeros());
        let predicted_alignment_increment = alignment - floored_alignment;
        let size = std::cmp::max(
            self.min_allocation_size,
            size + predicted_alignment_increment,
        )
        .next_power_of_two();

        let mut bu_iter = self.buffer_units.iter_mut();
        loop {
            let (buffer, buffer_unit_data) = match bu_iter.next() {
                Some(v) => (*v.0, v.1),
                None => {
                    let b = self.request_next_buffer(size);
                    (b, self.buffer_units.get_mut(&b).unwrap())
                }
            };

            for (blocks_size, blocks_addresses) in buffer_unit_data
                .free_blocks
                .range_mut((Included(size), Unbounded))
            {
                let aligned_block_address = blocks_addresses
                    .iter()
                    .find(|addr| **addr % floored_alignment == 0)
                    .copied();
                // If the block has the same size as requested and the correct alignment then it is selected right away
                if *blocks_size == size && aligned_block_address.is_some() {
                    blocks_addresses.remove(&aligned_block_address.unwrap());
                    if blocks_addresses.is_empty() {
                        buffer_unit_data.free_blocks.remove(blocks_size);
                    }
                    return;
                }
                // If the block has correct alignment but not the right size it is split up and then selected
                else if let Some(aligned_block_address) = aligned_block_address {
                    blocks_addresses.remove(&aligned_block_address);
                    if blocks_addresses.is_empty() {
                        buffer_unit_data.free_blocks.remove(blocks_size);
                    }
                    Self::split_block_recursive(
                        &mut buffer_unit_data.free_blocks,
                        *blocks_size,
                        aligned_block_address,
                        size,
                    );
                    return;
                }
            }
        }
    }

    // function that splits a block until its children are of the same size of requested_block_size
    // the block that is passed (old_block_size, old_block_address) is assumed to be already deleted from the map
    fn split_block_recursive(
        buffer_free_blocks: &mut BTreeMap<usize, HashSet<usize>>,
        old_block_size: usize,
        old_block_address: usize,
        requested_block_size: usize,
    ) -> (usize, usize) {
        // create the right block
        let new_block_size = old_block_size / 2;

        buffer_free_blocks
            .entry(new_block_size)
            .or_default()
            .insert(old_block_address + new_block_size);
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

    // function that given a block to free, tries to find children to merge, defragmenting the memory,
    // then reinserts the block back
    fn merge_block_recursive(
        buffer_free_blocks: &mut BTreeMap<usize, HashSet<usize>>,
        block_address: usize,
        block_size: usize,
    ) {
        if let Occupied(mut entry) = buffer_free_blocks.entry(block_size) {
            let left_block_address = (block_address - block_size) as i64;
            let right_block_address = (block_address + block_size) as i64;

            let addresses = entry.get_mut();

            if addresses.remove(&(left_block_address as usize)) {
                if addresses.is_empty() {
                    entry.remove_entry();
                }
                // The new block has the left adjacent block as the address
                return Self::merge_block_recursive(
                    buffer_free_blocks,
                    left_block_address as usize,
                    block_size * 2,
                );
            } else if addresses.remove(&(right_block_address as usize)) {
                if addresses.is_empty() {
                    entry.remove_entry();
                }
                // The new block has the source block address as the address
                return Self::merge_block_recursive(
                    buffer_free_blocks,
                    block_address,
                    block_size * 2,
                );
            }
        }
        buffer_free_blocks
            .entry(block_size)
            .or_default()
            .insert(block_address);
    }

    fn request_next_buffer(&mut self, buffer_size: usize) -> vk::Buffer {
        // All blocks needs to be have size to a power of 2
        let buffer_size =
            (std::cmp::max(buffer_size, self.blocks_initial_size)).next_power_of_two();

        let buffer_create_info = vk::BufferCreateInfo::builder()
            .size(buffer_size as u64)
            .usage(self.buffers_usage)
            .sharing_mode(vk::SharingMode::EXCLUSIVE);

        let buffer_allocation = RefCell::borrow_mut(self.allocator.borrow_mut())
            .allocate_buffer(&buffer_create_info, self.buffers_location);

        let buffer = buffer_allocation.buffer;
        let buffer_unit_data = BufferUnitData {
            allocation: buffer_allocation,
            free_blocks: BTreeMap::from([(buffer_size, HashSet::from([0]))]),
            used_blocks: Default::default(),
        };
        self.buffer_units.insert(buffer, buffer_unit_data);
        buffer
    }
}
