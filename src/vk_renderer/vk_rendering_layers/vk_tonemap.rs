#![allow(non_upper_case_globals)]
#![allow(non_snake_case)]

use super::super::vk_allocator::vk_descriptor_sets_allocator::*;
use super::super::vk_allocator::VkAllocator;
use super::super::vk_boot::helper::vk_create_shader_stage;
use crate::vk_renderer::vk_allocator::vk_buffers_suballocator::SubAllocationData;
use ash::vk;
use itertools::Itertools;
use nalgebra::*;
use std::cell::RefCell;
use std::path::Path;
use std::rc::Rc;

fn LpmColXyToZ(s: Vector2<f32>) -> Vector3<f32> {
    Vector3::new(s[0], s[1], 1.0 - s[0] + s[1])
}

fn LpmColRgbToXyz(
    r: Vector2<f32>,
    g: Vector2<f32>,
    b: Vector2<f32>,
    w: Vector2<f32>,
) -> Matrix3<f32> {
    // Expand from xy to xyz.
    let rz = LpmColXyToZ(r);
    let gz = LpmColXyToZ(g);
    let bz = LpmColXyToZ(b);

    let rgb3 = Matrix3::from_columns(&[rz, gz, bz]);

    // Convert white xyz to XYZ.
    let mut w3 = LpmColXyToZ(w);
    w3 *= (w[1]).recip();

    // Compute xyz to XYZ scalars for primaries.
    let rgbv = rgb3.try_inverse().unwrap();

    let s = Vector3::new(
        rgbv.row(0).transpose().dot(&w3),
        rgbv.row(1).transpose().dot(&w3),
        rgbv.row(2).transpose().dot(&w3),
    );
    // Scale.
    Matrix3::from_rows(&[
        rgb3.row(0).transpose().component_mul(&s).transpose(),
        rgb3.row(1).transpose().component_mul(&s).transpose(),
        rgb3.row(2).transpose().component_mul(&s).transpose(),
    ])
}

#[repr(C, packed)]
struct LpmData {
    ctl: [Vector4<u32>; 24],
}

impl LpmData {
    pub fn new(
        shoulder: bool,
        softGap: f32,
        hdrMax: f32,
        lpmExposure: f32,
        contrast: f32,
        shoulderContrast: f32,
        saturation: Vector3<f32>,
        crosstalk: Vector3<f32>,
    ) -> Self {
        // only mode is SDR with REC709 colorspace (aka sRGB)
        let [con, soft, con2, clip, scale_only] = Self::get_lpm_config();
        let (colors, scale) = Self::get_lpm_colors();

        Self {
            ctl: Self::get_control_block(
                shoulder,
                con,
                soft,
                con2,
                clip,
                scale_only,
                colors[0],
                colors[1],
                colors[2],
                colors[3],
                colors[4],
                colors[5],
                colors[6],
                colors[7],
                colors[8],
                colors[9],
                colors[10],
                colors[11],
                scale,
                softGap,
                hdrMax,
                lpmExposure,
                contrast,
                shoulderContrast,
                saturation,
                crosstalk,
            ),
        }
    }

    fn get_lpm_config() -> [bool; 5] {
        // LPM_CONFIG_709_709
        return [false, false, false, false, false];
    }

    fn get_lpm_colors() -> ([Vector2<f32>; 12], f32) {
        const lpmCol709R: Vector2<f32> = Vector2::new(0.64, 0.33);
        const lpmCol709G: Vector2<f32> = Vector2::new(0.30, 0.60);
        const lpmCol709B: Vector2<f32> = Vector2::new(0.15, 0.06);
        const lpmColD65: Vector2<f32> = Vector2::new(0.3127, 0.3290);

        // LPM_COLORS_709_709
        (
            [
                lpmCol709R, lpmCol709G, lpmCol709B, lpmColD65, lpmCol709R, lpmCol709G, lpmCol709B,
                lpmColD65, lpmCol709R, lpmCol709G, lpmCol709B, lpmColD65,
            ],
            1.0f32,
        )
    }

    fn get_control_block(
        // Path control.
        shoulder: bool, // Use optional extra shoulderContrast tuning (set to false if shoulderContrast is 1.0).
        // Prefab start, "LPM_CONFIG_".
        con: bool, // Use first RGB conversion matrix, if 'soft' then 'con' must be true also.
        soft: bool, // Use soft gamut mapping.
        con2: bool, // Use last RGB conversion matrix.
        clip: bool, // Use clipping in last conversion matrix.
        scaleOnly: bool, // Scale only for last conversion matrix (used for 709 HDR to scRGB).
        // Gamut control, "LPM_COLORS_".
        xyRedW: Vector2<f32>,
        xyGreenW: Vector2<f32>,
        xyBlueW: Vector2<f32>,
        xyWhiteW: Vector2<f32>, // Chroma coordinates for working color space.
        xyRedO: Vector2<f32>,
        xyGreenO: Vector2<f32>,
        xyBlueO: Vector2<f32>,
        xyWhiteO: Vector2<f32>, // For the output color space.
        xyRedC: Vector2<f32>,
        xyGreenC: Vector2<f32>,
        xyBlueC: Vector2<f32>,
        xyWhiteC: Vector2<f32>,
        scaleC: f32, // For the output container color space (if con2).
        // Prefab end.
        mut softGap: f32, // Range of 0 to a little over zero, controls how much feather region in out-of-gamut mapping, 0=clip.
        // Tonemapping control.
        hdrMax: f32,                  // Maximum input value.
        exposure: f32, // Number of stops between 'hdrMax' and 18% mid-level on input.
        mut contrast: f32, // Input range {0.0 (no extra contrast) to 1.0 (maximum contrast)}.
        shoulderContrast: f32, // Shoulder shaping, 1.0 = no change (fast path).
        mut saturation: Vector3<f32>, // A per channel adjustment, use <0 decrease, 0=no change, >0 increase.
        crosstalk: Vector3<f32>,
    ) -> [Vector4<u32>; 24] {
        contrast += 1.0f32;
        saturation.add_scalar_mut(contrast);
        softGap = softGap.max(1.0f32 / 1024.032);

        let midIn = hdrMax * 0.18f32 * (-exposure).exp2();
        let midOut = 0.18f32;

        let mut toneScaleBias = Vector2::<f32>::zeros();

        let cs = contrast * shoulderContrast;
        let z0 = -midIn.powf(contrast);
        let z1 = hdrMax.powf(cs) * midIn.powf(contrast);
        let z2 = hdrMax.powf(contrast) * midIn.powf(cs) * midOut;
        let z3 = hdrMax.powf(cs) * midOut;
        let z4 = midIn.powf(cs) * midOut;
        toneScaleBias[0] = -((z0 + (midOut * (z1 - z2)) * (z3 - z4).recip()) * z4.recip());

        let w0 = hdrMax.powf(cs) * midIn.powf(contrast);
        let w1 = hdrMax.powf(contrast) * midIn.powf(cs) * midOut;
        let w2 = hdrMax.powf(cs) * midOut;
        let w3 = midIn.powf(cs) * midOut;
        toneScaleBias[1] = (w0 - w1) * (w2 - w3).recip();

        let rgbToXyzW = LpmColRgbToXyz(xyRedW, xyGreenW, xyBlueW, xyWhiteW);
        // Use the Y vector of the matrix for the associated luma coef.
        // For safety, make sure the vector sums to 1.0.
        let lumaW = rgbToXyzW.row(1) * (rgbToXyzW.m21 + rgbToXyzW.m22 + rgbToXyzW.m23).recip();

        let rgbToXyz0 = LpmColRgbToXyz(xyRedO, xyGreenO, xyBlueO, xyWhiteO);
        let mut lumaT = if soft {
            rgbToXyz0.row(1).clone_owned()
        } else {
            rgbToXyzW.row(1).clone_owned()
        };
        lumaT *= (lumaT[0] + lumaT[1] + lumaT[2]).recip();
        let rcpLumaT = Vector3::from_element(1.0f32).component_div(&lumaT.transpose());

        let softGap2 = if soft {
            Vector2::new(
                softGap,
                (1.0f32 - softGap) * (softGap * std::f32::consts::LN_2).recip(),
            )
        } else {
            Vector2::zeros()
        };

        let con_m = if con {
            let xyzToRgb0 = rgbToXyz0.try_inverse().unwrap();
            xyzToRgb0 * rgbToXyzW
        } else {
            Matrix3::zeros()
        };

        let mut con2_m = if con2 {
            let xyzToRgbC = LpmColRgbToXyz(xyRedC, xyGreenC, xyBlueC, xyWhiteC)
                .try_inverse()
                .unwrap();
            let mut con2 = xyzToRgbC * rgbToXyz0;
            con2 *= scaleC;
            con2
        } else {
            Matrix3::zeros()
        };

        if scaleOnly {
            con2_m.m11 = scaleC;
        }

        // Write to control block
        let mut ctl: [Vector4<u32>; 24] = [Vector4::zeros(); 24];

        ctl[0] = Vector4::new(
            saturation[0].to_bits(),
            saturation[1].to_bits(),
            saturation[2].to_bits(),
            contrast.to_bits(),
        );
        ctl[1] = Vector4::new(
            toneScaleBias[0].to_bits(),
            toneScaleBias[1].to_bits(),
            lumaT[0].to_bits(),
            lumaT[1].to_bits(),
        );
        ctl[2] = Vector4::new(
            lumaT[2].to_bits(),
            crosstalk[0].to_bits(),
            crosstalk[1].to_bits(),
            crosstalk[2].to_bits(),
        );
        ctl[3] = Vector4::new(
            rcpLumaT[0].to_bits(),
            rcpLumaT[1].to_bits(),
            rcpLumaT[2].to_bits(),
            con2_m.m11.to_bits(),
        );
        ctl[4] = Vector4::new(
            con2_m.m12.to_bits(),
            con2_m.m13.to_bits(),
            con2_m.m21.to_bits(),
            con2_m.m22.to_bits(),
        );
        ctl[5] = Vector4::new(
            con2_m.m23.to_bits(),
            con2_m.m31.to_bits(),
            con2_m.m32.to_bits(),
            con2_m.m33.to_bits(),
        );
        ctl[6] = Vector4::new(
            shoulderContrast.to_bits(),
            lumaW[0].to_bits(),
            lumaW[1].to_bits(),
            lumaW[2].to_bits(),
        );
        ctl[7] = Vector4::new(
            softGap2[0].to_bits(),
            softGap2[1].to_bits(),
            con_m.m11.to_bits(),
            con_m.m12.to_bits(),
        );
        ctl[8] = Vector4::new(
            con_m.m13.to_bits(),
            con_m.m21.to_bits(),
            con_m.m22.to_bits(),
            con_m.m23.to_bits(),
        );
        ctl[9] = Vector4::new(
            con_m.m31.to_bits(),
            con_m.m32.to_bits(),
            con_m.m33.to_bits(),
            0,
        );

        fn pack_2f16_to_u32(f1: f32, f2: f32) -> u32 {
            (half::f16::from_f32(f1).to_bits() as u32) << 16
                | (half::f16::from_f32(f2).to_bits() as u32)
        }

        // Packed 16-bit part of control block.
        ctl[16] = Vector4::new(
            pack_2f16_to_u32(saturation[0], saturation[1]),
            pack_2f16_to_u32(saturation[2], contrast),
            pack_2f16_to_u32(toneScaleBias[0], toneScaleBias[1]),
            pack_2f16_to_u32(lumaT[0], lumaT[1]),
        );
        ctl[17] = Vector4::new(
            pack_2f16_to_u32(lumaT[2], crosstalk[0]),
            pack_2f16_to_u32(crosstalk[1], crosstalk[2]),
            pack_2f16_to_u32(rcpLumaT[0], rcpLumaT[1]),
            pack_2f16_to_u32(rcpLumaT[2], con2_m.m11),
        );
        ctl[18] = Vector4::new(
            pack_2f16_to_u32(con2_m.m12, con2_m.m13),
            pack_2f16_to_u32(con2_m.m21, con2_m.m22),
            pack_2f16_to_u32(con2_m.m23, con2_m.m31),
            pack_2f16_to_u32(con2_m.m32, con2_m.m33),
        );
        ctl[19] = Vector4::new(
            pack_2f16_to_u32(shoulderContrast, lumaW[0]),
            pack_2f16_to_u32(lumaW[1], lumaW[2]),
            pack_2f16_to_u32(softGap2[0], softGap2[1]),
            pack_2f16_to_u32(con_m.m11, con_m.m12),
        );
        ctl[20] = Vector4::new(
            pack_2f16_to_u32(con_m.m13, con_m.m21),
            pack_2f16_to_u32(con_m.m22, con_m.m23),
            pack_2f16_to_u32(con_m.m31, con_m.m32),
            pack_2f16_to_u32(con_m.m33, 0.0f32),
        );
        ctl
    }
}

pub struct VkTonemap {
    device: Rc<ash::Device>,
    allocator: Rc<RefCell<VkAllocator>>,
    presentation_resolution: vk::Extent2D,
    input_color_image: vk::Image,
    input_color_image_view: vk::ImageView,
    input_ao_image: vk::Image,
    input_ao_image_view: vk::ImageView,
    lpm_constant_buffer: SubAllocationData,
    output_images: Vec<vk::Image>,
    output_image_views: Vec<vk::ImageView>,
    descriptor_set_layout: vk::DescriptorSetLayout,
    descriptor_set_allocation: DescriptorSetAllocation,
    pipeline_layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
}

impl VkTonemap {
    pub fn new(
        device: Rc<ash::Device>,
        allocator: Rc<RefCell<VkAllocator>>,
        presentation_resolution: vk::Extent2D,
        shader_spirv_location: &Path,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
        output_images: Vec<vk::Image>,
        output_image_views: Vec<vk::ImageView>,
    ) -> Self {
        assert_eq!(output_images.len(), output_image_views.len());

        let descriptor_set_layout = unsafe {
            let descriptor_set_bindings = [
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(1)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(2)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .descriptor_count(output_images.len() as u32)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
                vk::DescriptorSetLayoutBinding::builder()
                    .binding(3)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
                    .build(),
            ];
            let flags = [
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::empty(),
                vk::DescriptorBindingFlags::PARTIALLY_BOUND,
                vk::DescriptorBindingFlags::empty(),
            ];
            let mut descriptor_flags =
                vk::DescriptorSetLayoutBindingFlagsCreateInfo::builder().binding_flags(&flags);
            let descriptor_set_ci = vk::DescriptorSetLayoutCreateInfo::builder()
                .push_next(&mut descriptor_flags)
                .bindings(&descriptor_set_bindings);
            device
                .create_descriptor_set_layout(&descriptor_set_ci, None)
                .unwrap()
        };
        let descriptor_set_allocation = allocator
            .as_ref()
            .borrow_mut()
            .get_descriptor_set_allocator_mut()
            .allocate_descriptor_sets(std::slice::from_ref(&descriptor_set_layout));

        let pipeline_info =
            Self::create_pipeline(&device, shader_spirv_location, &[descriptor_set_layout]);

        let lpm_constant_buffer = allocator
            .as_ref()
            .borrow_mut()
            .get_host_uniform_sub_allocator_mut()
            .allocate(std::mem::size_of::<LpmData>(), 128);
        let lpm_data = lpm_constant_buffer.get_host_ptr().unwrap().as_ptr() as *mut LpmData;
        unsafe {
            *lpm_data = LpmData::new(
                false,
                0.0,
                256.0,
                8.0,
                0.25,
                1.0,
                Vector3::zeros(),
                Vector3::new(1.0, 1.0 / 2.0, 1.0 / 32.0),
            );
        }

        let ret = Self {
            device,
            allocator,
            presentation_resolution,
            input_color_image,
            input_color_image_view,
            input_ao_image,
            input_ao_image_view,
            lpm_constant_buffer,
            output_images,
            output_image_views,
            descriptor_set_layout,
            descriptor_set_allocation,
            pipeline_layout: pipeline_info.0,
            pipeline: pipeline_info.1,
        };
        ret.update_descriptor_set();
        ret
    }

    pub fn resize(
        &mut self,
        presentation_resolution: vk::Extent2D,
        input_color_image: vk::Image,
        input_color_image_view: vk::ImageView,
        input_ao_image: vk::Image,
        input_ao_image_view: vk::ImageView,
        output_images: Vec<vk::Image>,
        output_image_views: Vec<vk::ImageView>,
    ) {
        self.presentation_resolution = presentation_resolution;
        self.input_color_image = input_color_image;
        self.input_color_image_view = input_color_image_view;
        self.input_ao_image = input_ao_image;
        self.input_ao_image_view = input_ao_image_view;
        self.output_images = output_images;
        self.output_image_views = output_image_views;
        self.update_descriptor_set();
    }

    pub fn present(&self, cb: vk::CommandBuffer, dst_image_idx: u32) {
        unsafe {
            let image_memory_barriers = [
                vk::ImageMemoryBarrier2::builder()
                    .src_stage_mask(vk::PipelineStageFlags2::RAY_TRACING_SHADER_KHR)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
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
                    .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .src_access_mask(vk::AccessFlags2::SHADER_STORAGE_WRITE)
                    .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                    .dst_access_mask(vk::AccessFlags2::SHADER_STORAGE_READ_KHR)
                    .old_layout(vk::ImageLayout::GENERAL)
                    .new_layout(vk::ImageLayout::GENERAL)
                    .image(self.input_ao_image)
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
                    .image(self.output_images[dst_image_idx as usize])
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
            self.device.cmd_pipeline_barrier2(cb, &dependency_info);

            self.device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, self.pipeline);

            self.device.cmd_push_constants(
                cb,
                self.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                &dst_image_idx.to_ne_bytes(),
            );

            self.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                self.pipeline_layout,
                0,
                std::slice::from_ref(&self.descriptor_set_allocation.get_descriptor_sets()[0]),
                &[],
            );

            self.device.cmd_dispatch(
                cb,
                self.presentation_resolution.width / 8,
                self.presentation_resolution.height / 8,
                1,
            );
        }
    }

    fn update_descriptor_set(&self) {
        unsafe {
            let input_color = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_color_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let input_ao = vk::DescriptorImageInfo::builder()
                .sampler(vk::Sampler::null())
                .image_view(self.input_ao_image_view)
                .image_layout(vk::ImageLayout::GENERAL);

            let output_images = self
                .output_image_views
                .iter()
                .copied()
                .map(|image_view| {
                    vk::DescriptorImageInfo::builder()
                        .sampler(vk::Sampler::null())
                        .image_view(image_view)
                        .image_layout(vk::ImageLayout::GENERAL)
                        .build()
                })
                .collect_vec();

            let lpm_buffer_info = vk::DescriptorBufferInfo::builder()
                .buffer(self.lpm_constant_buffer.get_buffer())
                .offset(self.lpm_constant_buffer.get_buffer_offset() as u64)
                .range(std::mem::size_of::<LpmData>() as u64);

            let descriptor_set_writes = [
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(0)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&input_color))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(1)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(std::slice::from_ref(&input_ao))
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(2)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::STORAGE_IMAGE)
                    .image_info(&output_images)
                    .build(),
                vk::WriteDescriptorSet::builder()
                    .dst_set(self.descriptor_set_allocation.get_descriptor_sets()[0])
                    .dst_binding(3)
                    .dst_array_element(0)
                    .descriptor_type(vk::DescriptorType::UNIFORM_BUFFER)
                    .buffer_info(std::slice::from_ref(&lpm_buffer_info))
                    .build(),
            ];
            self.device
                .update_descriptor_sets(&descriptor_set_writes, &[]);
        }
    }

    fn create_pipeline(
        device: &ash::Device,
        path: &Path,
        set_layouts: &[vk::DescriptorSetLayout],
    ) -> (vk::PipelineLayout, vk::Pipeline) {
        let shader_stage = vk_create_shader_stage(
            format!("{}//{}", path.to_str().unwrap(), "tonemap.comp.spirv"),
            device,
        );

        let pipeline_layout = unsafe {
            let push_constant = vk::PushConstantRange::builder()
                .stage_flags(vk::ShaderStageFlags::COMPUTE)
                .offset(0)
                .size(4);
            let pipeline_layout_ci = vk::PipelineLayoutCreateInfo::builder()
                .set_layouts(set_layouts)
                .push_constant_ranges(std::slice::from_ref(&push_constant));
            device
                .create_pipeline_layout(&pipeline_layout_ci, None)
                .unwrap()
        };

        let pipeline_ci = vk::ComputePipelineCreateInfo::builder()
            .stage(shader_stage)
            .layout(pipeline_layout);

        let pipeline = unsafe {
            device
                .create_compute_pipelines(
                    vk::PipelineCache::null(),
                    std::slice::from_ref(&pipeline_ci),
                    None,
                )
                .unwrap()[0]
        };

        unsafe {
            device.destroy_shader_module(shader_stage.module, None);
        }

        (pipeline_layout, pipeline)
    }
}

impl Drop for VkTonemap {
    fn drop(&mut self) {
        unsafe {
            self.device
                .destroy_pipeline_layout(self.pipeline_layout, None);
            self.device.destroy_pipeline(self.pipeline, None);

            self.device
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);

            let mut al = self.allocator.as_ref().borrow_mut();
            al.get_descriptor_set_allocator_mut()
                .free_descriptor_sets(std::mem::replace(
                    &mut self.descriptor_set_allocation,
                    DescriptorSetAllocation::null(),
                ));
            al.get_host_uniform_sub_allocator_mut()
                .free(std::mem::replace(
                    &mut self.lpm_constant_buffer,
                    std::mem::zeroed(),
                ));
        }
    }
}
