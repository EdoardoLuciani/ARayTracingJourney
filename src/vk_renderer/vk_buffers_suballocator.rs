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

pub struct VkBuffersSubAllocator {
    allocator: Rc<RefCell<VkMemoryResourceAllocator>>,
    buffers_usage: vk::BufferUsageFlags,
    buffers_location: MemoryLocation,
    blocks_initial_size: usize,
    min_allocation_size: usize,
    buffer_units: HashMap<vk::Buffer, BufferUnitData>,
}

struct BufferUnitData {
    allocation: BufferAllocation,
    // tree to keep size|offset, the offset is a hashset to accommodate duplicate size values
    free_blocks: BTreeMap<usize, HashSet<usize>>,
    // hashmap to keep offset|used_block
    used_blocks: HashMap<usize, UsedBlock>,
}

struct UsedBlock {
    size: usize,
    po2_alignment_increment: usize,
}

pub struct SubAllocationData {
    buffer: vk::Buffer,
    buffer_offset: usize,
    host_ptr: Option<std::ptr::NonNull<std::ffi::c_void>>,
    device_ptr: Option<vk::DeviceAddress>,
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

    pub fn allocate(&mut self, size: usize, alignment: usize) -> SubAllocationData {
        // The alignment is split in two parts:
        // - the first is a power of 2 which it is used to look for suitable blocks,
        // - the second is a predicted residual part that becomes part of the allocation size, note however that the real
        // residual is calculated at the moment of block allocation, this prediction is the worst case scenario and guarantees
        // that the block has enough size to accommodate data + alignment

        // this is just a bit_floor, i.e. rounding down to the previous power of 2
        let po2_rounded_alignment = 1 << (usize::BITS - (alignment >> 1).leading_zeros());
        let predicted_alignment_increment = alignment - po2_rounded_alignment;
        let size = std::cmp::max(
            self.min_allocation_size,
            size + predicted_alignment_increment,
        )
        .next_power_of_two();

        let (buffer, mut block_size, mut block_address) =
            self.pick_best_fit_block(size, po2_rounded_alignment);

        let free_blocks = &mut self.buffer_units.get_mut(&buffer).unwrap().free_blocks;
        // remove the selected block
        if free_blocks[&block_size].len() == 1 {
            // there is only one block with that size so remove the whole BTree entry
            free_blocks.remove(&block_size);
        } else {
            // remove only the address in the BTree entry value
            free_blocks
                .get_mut(&block_size)
                .unwrap()
                .remove(&block_address);
        }

        // if the selected block is too big, split it
        if block_size != size {
            block_address =
                Self::split_block_recursive(free_blocks, block_size, block_address, size);
        }

        // correct the predicted alignment and insert it to used_blocks
        let corrected_alignment_increment = predicted_alignment_increment
            * f32::ceil(block_address as f32 / predicted_alignment_increment as f32) as usize
            - block_address;
        let used_block = UsedBlock {
            size: block_size,
            po2_alignment_increment: corrected_alignment_increment,
        };
        self.buffer_units
            .get_mut(&buffer)
            .unwrap()
            .used_blocks
            .insert(block_address + corrected_alignment_increment, used_block);

        let buffer_offset = block_address + corrected_alignment_increment;
        SubAllocationData {
            buffer: buffer,
            buffer_offset,
            host_ptr: self.buffer_units[&buffer]
                .allocation
                .allocation
                .mapped_ptr(),
            device_ptr: Some(
                self.buffer_units[&buffer].allocation.device_address + buffer_offset as u64,
            ),
        }
    }

    pub fn free() {
        todo!();
    }

    // picks the best available block given the input. Return type is block_size, block_address
    fn pick_best_fit_block(&mut self, size: usize, alignment: usize) -> (vk::Buffer, usize, usize) {
        let buffer_handles = self
            .buffer_units
            .keys()
            .copied()
            .collect::<Vec<vk::Buffer>>();
        let mut buffer_iter = buffer_handles.iter();
        loop {
            let (buffer, buffer_unit_data) = match buffer_iter.next() {
                Some(v) => (*v, self.buffer_units.get(v).unwrap()),
                None => {
                    let b = self.request_next_buffer(size);
                    (b, self.buffer_units.get(&b).unwrap())
                }
            };

            for (blocks_size, blocks_addresses) in buffer_unit_data
                .free_blocks
                .range((Included(size), Unbounded))
            {
                let aligned_block_address = blocks_addresses
                    .iter()
                    .find(|addr| **addr % alignment == 0)
                    .copied();
                // Select the smallest block with the correct alignment
                if let Some(aligned_block_address) = aligned_block_address {
                    return (buffer, *blocks_size, aligned_block_address);
                }
            }
        }
    }

    // function that splits a block until its children are of the same size of requested_block_size
    // the block that is passed (block_size, block_address) is assumed to be already deleted from the map
    fn split_block_recursive(
        buffer_free_blocks: &mut BTreeMap<usize, HashSet<usize>>,
        block_size: usize,
        block_address: usize,
        requested_block_size: usize,
    ) -> usize {
        // create the right block
        let new_block_size = block_size / 2;

        buffer_free_blocks
            .entry(new_block_size)
            .or_default()
            .insert(block_address + new_block_size);
        if new_block_size != requested_block_size {
            // continuing to subdivide the left block without actually creating it
            return Self::split_block_recursive(
                buffer_free_blocks,
                new_block_size,
                block_address,
                requested_block_size,
            );
        }
        // on the last step we return the data of the left block, but we do not create it, since it is going to be removed shortly after
        block_address
    }

    // function that given a block to free, tries to find children to merge, then reinserts the block back
    fn merge_block_recursive(
        buffer_free_blocks: &mut BTreeMap<usize, HashSet<usize>>,
        block_address: usize,
        block_size: usize,
    ) {
        if let Occupied(mut entry) = buffer_free_blocks.entry(block_size) {
            let left_block_address = block_address as i64 - block_size as i64;
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

#[cfg(test)]
mod tests {
    use crate::vk_renderer::vk_buffers_suballocator::VkBuffersSubAllocator;
    use std::collections::{BTreeMap, HashMap, HashSet};

    #[test]
    fn split_block_test() {
        let mut free_blocks = BTreeMap::from([(256usize, HashSet::<usize>::from([0, 256]))]);
        let res_block =
            VkBuffersSubAllocator::split_block_recursive(&mut free_blocks, 1024, 512, 128);
        assert_eq!(res_block, 512);
        assert_eq!(free_blocks.len(), 3);
        assert_eq!(free_blocks[&128], HashSet::<usize>::from([640]));
        assert_eq!(free_blocks[&256], HashSet::<usize>::from([0, 256, 768]));
        assert_eq!(free_blocks[&512], HashSet::<usize>::from([1024]));
    }

    #[test]
    fn merge_block_test() {
        let mut free_blocks = BTreeMap::from([
            (128usize, HashSet::<usize>::from([640])),
            (256usize, HashSet::<usize>::from([0, 256, 768])),
            (512usize, HashSet::<usize>::from([1024])),
        ]);
        VkBuffersSubAllocator::merge_block_recursive(&mut free_blocks, 512, 128);
        // there is also another way to merge the blocks! this is not the only result possible
        assert_eq!(free_blocks.len(), 2);
        assert_eq!(free_blocks[&256], HashSet::<usize>::from([0, 768]));
        assert_eq!(free_blocks[&512], HashSet::<usize>::from([256, 1024]));
    }
}
