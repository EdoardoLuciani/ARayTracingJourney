use super::vk_allocator::vk_buffers_suballocator::SubAllocationData;
use super::vk_allocator::vk_descriptor_sets_allocator::DescriptorSetAllocation;
use super::vk_allocator::VkAllocator;
use ash::vk;
use nalgebra::*;
use std::cell::RefCell;
use std::rc::Rc;

#[repr(C, packed)]
struct Uniform {
    view: Matrix4<f32>,
    view_inv: Matrix4<f32>,
    prev_view: Matrix4<f32>,
    proj: Matrix4<f32>,
    proj_inv: Matrix4<f32>,
    prev_proj: Matrix4<f32>,
    vp: Matrix4<f32>,
    prev_vp: Matrix4<f32>,
    camera_pos: Vector3<f32>,
    pad: f32,
    jitter: Vector2<f32>,
    prev_jitter: Vector2<f32>
}

pub struct VkCamera {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    pos: Vector3<f32>,
    dir: Vector3<f32>,
    aspect: f32,
    fovy: f32,
    znear: f32,
    zfar: f32,
    jitter: Vector2<f32>,
    host_uniform_suballocation: SubAllocationData,
    uniform_descriptor_set_layout: vk::DescriptorSetLayout,
    uniform_descriptor_set_allocation: DescriptorSetAllocation,
}

impl VkCamera {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        pos: Vector3<f32>,
        dir: Vector3<f32>,
        aspect: f32,
        fovy: f32,
        znear: f32,
        zfar: f32,
    ) -> Self {
        let host_uniform_suballocation = allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<Uniform>(), 1);

        let uniform_descriptor_set_layout = unsafe {
            let binding = vk::DescriptorSetLayoutBinding::builder()
                .binding(0)
                .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                .descriptor_count(1)
                .stage_flags(
                    vk::ShaderStageFlags::RAYGEN_KHR | vk::ShaderStageFlags::CLOSEST_HIT_KHR,
                );
            let descriptor_set_layout_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .bindings(std::slice::from_ref(&binding));
            device
                .create_descriptor_set_layout(&descriptor_set_layout_ci, None)
                .unwrap()
        };

        let uniform_descriptor_set_allocation = {
            let descriptor_set_allocation = allocator
                .as_ref()
                .borrow_mut()
                .get_descriptor_set_allocator_mut()
                .allocate_descriptor_sets(&[uniform_descriptor_set_layout]);

            unsafe {
                let descriptor_buffer_info = vk::DescriptorBufferInfo::builder()
                    .buffer(host_uniform_suballocation.get_buffer())
                    .offset(host_uniform_suballocation.get_buffer_offset() as u64)
                    .range(std::mem::size_of::<Uniform>() as u64);
                let write_descriptor_set = vk::WriteDescriptorSet::builder()
                    .dst_set(descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&descriptor_buffer_info));
                device.update_descriptor_sets(std::slice::from_ref(&write_descriptor_set), &[]);
            }
            descriptor_set_allocation
        };

        let mut ret = VkCamera {
            device,
            allocator,
            pos,
            dir,
            aspect,
            fovy,
            znear,
            zfar,
            host_uniform_suballocation,
            uniform_descriptor_set_layout,
            uniform_descriptor_set_allocation,
            jitter: Vector2::<f32>::zeros(),
        };
        ret
    }

    pub fn update_host_buffer(&mut self) {
        let view = self.view_matrix();
        let view_mat = view.to_homogeneous();

        let proj_mat = self.perspective_matrix();

        let uniform_host_ptr = self
            .host_uniform_suballocation
            .get_host_ptr()
            .unwrap()
            .as_ptr() as *mut Uniform;
        unsafe {
            (*uniform_host_ptr).prev_view = (*uniform_host_ptr).view;
            (*uniform_host_ptr).prev_proj = (*uniform_host_ptr).proj;
            (*uniform_host_ptr).prev_vp = (*uniform_host_ptr).vp;
            (*uniform_host_ptr).prev_jitter = (*uniform_host_ptr).jitter;

            (*uniform_host_ptr).view = view_mat;
            (*uniform_host_ptr).view_inv = view.inverse().to_homogeneous();

            (*uniform_host_ptr).proj = proj_mat;
            (*uniform_host_ptr).proj_inv = proj_mat.try_inverse().unwrap();

            (*uniform_host_ptr).vp = proj_mat * view_mat;

            (*uniform_host_ptr).camera_pos = self.pos;

            (*uniform_host_ptr).jitter = self.jitter;
        }
    }

    pub fn set_pos(&mut self, pos: Vector3<f32>) {
        self.pos = pos;
    }

    pub fn set_dir(&mut self, dir: Vector3<f32>) {
        self.dir = dir.normalize();
    }

    pub fn set_aspect(&mut self, aspect: f32) {
        self.aspect = aspect;
    }

    pub fn set_fovy(&mut self, fovy: f32) {
        self.fovy = fovy;
    }

    pub fn set_znear(&mut self, znear: f32) {
        self.znear = znear;
    }

    pub fn set_zfar(&mut self, zfar: f32) {
        self.zfar = zfar;
    }

    pub fn set_pixel_jitter(&mut self, jitter: Vector2<f32>) {
        self.jitter = jitter;
    }

    pub fn pos(&self) -> Vector3<f32> {
        self.pos
    }

    pub fn dir(&self) -> Vector3<f32> {
        self.dir
    }

    pub fn aspect(&self) -> f32 {
        self.aspect
    }

    pub fn fovy(&self) -> f32 {
        self.fovy
    }

    pub fn znear(&self) -> f32 {
        self.znear
    }

    pub fn zfar(&self) -> f32 {
        self.zfar
    }

    pub fn view_matrix(&self) -> Isometry3<f32> {
        Isometry3::look_at_rh(
            &Point3::from(self.pos),
            &Point3::from(self.pos + self.dir),
            &Vector3::new(0.0f32, -1.0f32, 0.0f32),
        )
    }

    pub fn perspective_matrix(&self) -> Matrix4<f32> {
        Perspective3::new(self.aspect, self.fovy, self.znear, self.zfar).to_homogeneous()
    }

    pub fn descriptor_set_layout(&self) -> vk::DescriptorSetLayout {
        self.uniform_descriptor_set_layout
    }

    pub fn descriptor_set(&self) -> vk::DescriptorSet {
        self.uniform_descriptor_set_allocation.get_descriptor_sets()[0]
    }
}

impl Drop for VkCamera {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_descriptor_set_layout(self.uniform_descriptor_set_layout, None);
        }

        let mut al = self.allocator.as_ref().borrow_mut();

        take_mut::take(&mut self.host_uniform_suballocation, |suballocation| {
            al.get_host_uniform_sub_allocator_mut().free(suballocation);
            unsafe { std::mem::zeroed() }
        });

        take_mut::take(
            &mut self.uniform_descriptor_set_allocation,
            |ds_allocation| {
                al.get_descriptor_set_allocator_mut()
                    .free_descriptor_sets(ds_allocation);
                unsafe { DescriptorSetAllocation::null() }
            },
        );
    }
}
