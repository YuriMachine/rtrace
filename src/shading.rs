use crate::scene_components::MaterialType;
use crate::{one3, utils::*, vec_comp_div, vec_comp_mul, zero3};
use glm::{dot, normalize, vec2, vec3};
use glm::{Vec2, Vec3};
use nalgebra_glm::{epsilon, is_null};
use std::collections::VecDeque;
use std::f32::consts::PI;

pub struct MaterialPoint {
    pub m_type: MaterialType,
    pub emission: Vec3,
    pub color: Vec3,
    pub roughness: f32,
    pub metallic: f32,
    pub ior: f32,
    pub density: Vec3,
    pub scattering: Vec3,
    pub scanisotropy: f32,
    pub trdepth: f32,
    pub opacity: f32,
}

impl Default for MaterialPoint {
    fn default() -> Self {
        MaterialPoint {
            m_type: MaterialType::Matte,
            emission: zero3!(),
            color: zero3!(),
            roughness: 0.0,
            metallic: 0.0,
            ior: 1.0,
            density: zero3!(),
            scattering: zero3!(),
            scanisotropy: 0.0,
            trdepth: 0.01,
            opacity: 1.0,
        }
    }
}

impl MaterialPoint {
    pub fn eval_emission(&self, normal: &Vec3, outgoing: &Vec3) -> Vec3 {
        if dot(normal, outgoing) >= 0.0 {
            self.emission
        } else {
            zero3!()
        }
    }

    pub fn sample_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        match self.m_type {
            MaterialType::Matte => self.sample_matte(normal, outgoing, rn),
            MaterialType::Glossy => self.sample_glossy(normal, outgoing, rnl, rn),
            MaterialType::Reflective => self.sample_reflective(normal, outgoing, rn),
            MaterialType::Transparent => self.sample_transparent(normal, outgoing, rnl, rn),
            MaterialType::Refractive => self.sample_refractive(normal, outgoing, rnl, rn),
            MaterialType::Subsurface => self.sample_refractive(normal, outgoing, rnl, rn),
            MaterialType::Gltfpbr => self.sample_gltfpbr(normal, outgoing, rnl, rn),
            _ => zero3!(),
        }
    }

    pub fn eval_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness == 0.0 {
            return zero3!();
        }
        match self.m_type {
            MaterialType::Matte => self.eval_matte(normal, outgoing, incoming),
            MaterialType::Glossy => self.eval_glossy(normal, outgoing, incoming),
            MaterialType::Reflective => self.eval_reflective(normal, outgoing, incoming),
            MaterialType::Transparent => self.eval_transparent(normal, outgoing, incoming),
            MaterialType::Refractive => self.eval_refractive(normal, outgoing, incoming),
            MaterialType::Subsurface => self.eval_refractive(normal, outgoing, incoming),
            MaterialType::Gltfpbr => self.eval_gltfpbr(normal, outgoing, incoming),
            _ => zero3!(),
        }
    }

    pub fn sample_bsdfcos_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if self.roughness == 0.0 {
            return 0.0;
        }
        match self.m_type {
            MaterialType::Matte => self.sample_matte_pdf(normal, outgoing, incoming),
            MaterialType::Glossy => self.sample_glossy_pdf(normal, outgoing, incoming),
            MaterialType::Reflective => self.sample_reflective_pdf(normal, outgoing, incoming),
            MaterialType::Transparent => self.sample_transparent_pdf(normal, outgoing, incoming),
            MaterialType::Refractive => self.sample_refractive_pdf(normal, outgoing, incoming),
            MaterialType::Subsurface => self.sample_refractive_pdf(normal, outgoing, incoming),
            MaterialType::Gltfpbr => self.sample_gltfpbr_pdf(normal, outgoing, incoming),
            _ => 0.0,
        }
    }

    pub fn sample_delta(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32) -> Vec3 {
        if self.roughness != 0.0 {
            return zero3!();
        }
        match self.m_type {
            MaterialType::Reflective => self.sample_reflective_delta(normal, outgoing),
            MaterialType::Transparent => self.sample_transparent_delta(normal, outgoing, rnl),
            MaterialType::Refractive => self.sample_refractive_delta(normal, outgoing, rnl),
            _ => zero3!(),
        }
    }

    pub fn eval_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness != 0.0 {
            return zero3!();
        }
        match self.m_type {
            MaterialType::Reflective => self.eval_reflective_delta(normal, outgoing, incoming),
            MaterialType::Transparent => self.eval_transparent_delta(normal, outgoing, incoming),
            MaterialType::Refractive => self.eval_refractive_delta(normal, outgoing, incoming),
            _ => zero3!(),
        }
    }

    pub fn sample_delta_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if self.roughness != 0.0 {
            return 0.0;
        }
        match self.m_type {
            MaterialType::Reflective => {
                self.sample_reflective_pdf_delta(normal, outgoing, incoming)
            }
            MaterialType::Transparent => {
                self.sample_transparent_pdf_delta(normal, outgoing, incoming)
            }
            MaterialType::Refractive => {
                self.sample_refractive_pdf_delta(normal, outgoing, incoming)
            }
            _ => 0.0,
        }
    }

    fn sample_matte(&self, normal: &Vec3, outgoing: &Vec3, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        sample_hemisphere_cos(&up_normal, rn)
    }

    fn eval_matte(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return zero3!();
        }
        self.color / PI * f32::abs(dot(normal, incoming))
    }

    fn sample_matte_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        sample_hemisphere_cos_pdf(&up_normal, incoming)
    }

    fn sample_glossy(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if rnl < fresnel_dielectric(self.ior, &up_normal, outgoing) {
            let halfway = sample_microfacet(self.roughness, &up_normal, rn, true);
            let incoming = glm::reflect_vec(&(-outgoing), &halfway);
            if !same_hemisphere(&up_normal, outgoing, &incoming) {
                zero3!()
            } else {
                incoming
            }
        } else {
            sample_hemisphere_cos(&up_normal, rn)
        }
    }

    fn eval_glossy(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return zero3!();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let f1 = fresnel_dielectric(self.ior, &up_normal, outgoing);
        let halfway = normalize(&(incoming + outgoing));
        let f = fresnel_dielectric(self.ior, &halfway, incoming);
        let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
        let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
        self.color * (1.0 - f1) / PI * f32::abs(dot(&up_normal, incoming))
            + one3!() * f * d * g / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
                * f32::abs(dot(&up_normal, incoming))
    }

    fn sample_glossy_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(outgoing + incoming));
        let f = fresnel_dielectric(self.ior, &up_normal, outgoing);
        f * sample_microfacet_pdf(self.roughness, &up_normal, &halfway)
            / (4.0 * f32::abs(dot(outgoing, &halfway)))
            + (1.0 - f) * sample_hemisphere_cos_pdf(&up_normal, incoming)
    }

    fn sample_reflective(&self, normal: &Vec3, outgoing: &Vec3, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = sample_microfacet(self.roughness, &up_normal, rn, true);
        let incoming = glm::reflect_vec(&(-outgoing), &halfway);
        if !same_hemisphere(&up_normal, outgoing, &incoming) {
            zero3!()
        } else {
            incoming
        }
    }

    fn sample_reflective_delta(&self, normal: &Vec3, outgoing: &Vec3) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        glm::reflect_vec(&(-outgoing), &up_normal)
    }

    fn eval_reflective(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return zero3!();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(incoming + outgoing));
        let f = fresnel_conductor(
            &reflectivity_to_eta(&self.color),
            &zero3!(),
            &halfway,
            incoming,
        );
        let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
        let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
        f * d * g / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
            * f32::abs(dot(&up_normal, incoming))
    }

    fn eval_reflective_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return zero3!();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        fresnel_conductor(
            &reflectivity_to_eta(&self.color),
            &zero3!(),
            &up_normal,
            outgoing,
        )
    }

    fn sample_reflective_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(outgoing + incoming));
        return sample_microfacet_pdf(self.roughness, &up_normal, &halfway)
            / (4.0 * f32::abs(dot(outgoing, &halfway)));
    }

    fn sample_reflective_pdf_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        1.0
    }

    fn sample_transparent(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = sample_microfacet(self.roughness, &up_normal, rn, true);
        if rnl < fresnel_dielectric(self.ior, &halfway, outgoing) {
            let incoming = glm::reflect_vec(&(-outgoing), &halfway);
            if !same_hemisphere(&up_normal, outgoing, &incoming) {
                return zero3!();
            } else {
                incoming
            }
        } else {
            let reflected = glm::reflect_vec(&(-outgoing), &halfway);
            let incoming = -glm::reflect_vec(&(-reflected), &up_normal);
            if same_hemisphere(&up_normal, outgoing, &incoming) {
                return zero3!();
            } else {
                incoming
            }
        }
    }

    fn sample_transparent_delta(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if rnl < fresnel_dielectric(self.ior, &up_normal, outgoing) {
            glm::reflect_vec(&(-outgoing), &up_normal)
        } else {
            -outgoing
        }
    }

    fn eval_transparent(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            let halfway = normalize(&(incoming + outgoing));
            let f = fresnel_dielectric(self.ior, &halfway, outgoing);
            let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
            let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
            one3!() * f * d * g / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
                * f32::abs(dot(&up_normal, incoming))
        } else {
            let reflected = glm::reflect_vec(incoming, &up_normal);
            let halfway = normalize(&(reflected + outgoing));
            let f = fresnel_dielectric(self.ior, &halfway, outgoing);
            let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
            let g =
                microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, &reflected);
            self.color * (1.0 - f) * d * g
                / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, &reflected))
                * f32::abs(dot(&up_normal, &reflected))
        }
    }

    fn eval_transparent_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            one3!() * fresnel_dielectric(self.ior, &up_normal, outgoing)
        } else {
            self.color * (1.0 - fresnel_dielectric(self.ior, &up_normal, outgoing))
        }
    }

    fn sample_transparent_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            let halfway = normalize(&(incoming + outgoing));
            fresnel_dielectric(self.ior, &halfway, outgoing)
                * sample_microfacet_pdf(self.roughness, &up_normal, &halfway)
                / (4.0 * f32::abs(dot(outgoing, &halfway)))
        } else {
            let reflected = glm::reflect_vec(incoming, &up_normal);
            let halfway = normalize(&(reflected + outgoing));
            let d = (1.0 - fresnel_dielectric(self.ior, &halfway, outgoing))
                * sample_microfacet_pdf(self.roughness, &up_normal, &halfway);
            d / (4.0 * f32::abs(dot(outgoing, &halfway)))
        }
    }

    fn sample_transparent_pdf_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            fresnel_dielectric(self.ior, &up_normal, outgoing)
        } else {
            1.0 - fresnel_dielectric(self.ior, &up_normal, outgoing)
        }
    }

    fn sample_refractive(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = sample_microfacet(self.roughness, &up_normal, rn, true);
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if rnl < fresnel_dielectric(rel_ior, &halfway, outgoing) {
            let incoming = glm::reflect_vec(&(-outgoing), &halfway);
            if !same_hemisphere(&up_normal, outgoing, &incoming) {
                zero3!()
            } else {
                incoming
            }
        } else {
            let incoming = glm::refract_vec(&(-outgoing), &halfway, rel_ior);
            if !same_hemisphere(&up_normal, outgoing, &incoming) {
                zero3!()
            } else {
                incoming
            }
        }
    }

    fn sample_refractive_delta(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32) -> Vec3 {
        if f32::abs(self.ior - 1.0) < 1e-3 {
            return -outgoing;
        }
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if rnl < fresnel_dielectric(rel_ior, &up_normal, outgoing) {
            glm::reflect_vec(&(-outgoing), &up_normal)
        } else {
            glm::refract_vec(&(-outgoing), &up_normal, 1.0 / rel_ior)
        }
    }

    fn eval_refractive(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            let halfway = normalize(&(incoming + outgoing));
            let f = fresnel_dielectric(rel_ior, &halfway, outgoing);
            let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
            let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
            one3!() * f * d * g / f32::abs(4.0 * dot(normal, outgoing) * dot(normal, incoming))
                * f32::abs(dot(normal, incoming))
        } else {
            let rel_sign = if entering { 1.0 } else { -1.0 };
            let halfway = -normalize(&(rel_ior * incoming + outgoing)) * rel_sign;
            let f = fresnel_dielectric(rel_ior, &halfway, outgoing);
            let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
            let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
            // [Walter 2007] equation 21
            one3!()
                * f32::abs(
                    (dot(outgoing, &halfway) * dot(incoming, &halfway))
                        / (dot(outgoing, normal) * dot(incoming, normal)),
                )
                * (1.0 - f)
                * d
                * g
                / f32::powf(
                    rel_ior * dot(&halfway, incoming) + dot(&halfway, outgoing),
                    2.0,
                )
                * f32::abs(dot(normal, incoming))
        }
    }

    fn eval_refractive_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if f32::abs(self.ior - 1.0) < 1e-3 {
            if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
                return one3!();
            } else {
                return zero3!();
            }
        }
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            one3!() * fresnel_dielectric(rel_ior, &up_normal, outgoing)
        } else {
            one3!()
                * (1.0 / (rel_ior * rel_ior))
                * (1.0 - fresnel_dielectric(rel_ior, &up_normal, outgoing))
        }
    }

    fn sample_refractive_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            let halfway = normalize(&(incoming + outgoing));
            fresnel_dielectric(rel_ior, &halfway, outgoing) *
                   sample_microfacet_pdf(self.roughness, &up_normal, &halfway) /
                   //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing) /
                   (4.0 * f32::abs(dot(outgoing, &halfway)))
        } else {
            let rel_sign = if entering { 1.0 } else { -1.0 };
            let halfway = -normalize(&(rel_ior * incoming + outgoing)) * rel_sign;
            // [Walter 2007] equation 17
            (1.0 - fresnel_dielectric(rel_ior, &halfway, outgoing)) *
                   sample_microfacet_pdf(self.roughness, &up_normal, &halfway) *
                   //  sample_microfacet_pdf(roughness, up_normal, halfway, outgoing) /
                   f32::abs(dot(&halfway, incoming)) /  // here we use incoming as from pbrt
                   f32::powf(rel_ior * dot(&halfway, incoming) + dot(&halfway, outgoing), 2.0)
        }
    }

    fn sample_refractive_pdf_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if f32::abs(self.ior - 1.0) < 1e-3 {
            if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        let entering = dot(normal, outgoing) >= 0.0;
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let rel_ior = if entering { self.ior } else { 1.0 / self.ior };
        if dot(normal, incoming) * dot(normal, outgoing) >= 0.0 {
            fresnel_dielectric(rel_ior, &up_normal, outgoing)
        } else {
            1.0 - fresnel_dielectric(rel_ior, &up_normal, outgoing)
        }
    }

    fn sample_gltfpbr(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let reflectivity = glm::lerp(
            &eta_to_reflectivity(&vec3(self.ior, self.ior, self.ior)),
            &self.color,
            self.metallic,
        );
        if rnl < mean3(&fresnel_schlick(&reflectivity, &up_normal, outgoing)) {
            let halfway = sample_microfacet(self.roughness, &up_normal, rn, true);
            let incoming = glm::reflect_vec(&(-outgoing), &halfway);
            if !same_hemisphere(&up_normal, outgoing, &incoming) {
                zero3!()
            } else {
                incoming
            }
        } else {
            sample_hemisphere_cos(&up_normal, rn)
        }
    }

    fn eval_gltfpbr(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return zero3!();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let reflectivity = glm::lerp(
            &eta_to_reflectivity(&vec3(self.ior, self.ior, self.ior)),
            &self.color,
            self.metallic,
        );
        let f1 = fresnel_schlick(&reflectivity, &up_normal, outgoing);
        let halfway = normalize(&(incoming + outgoing));
        let f = fresnel_schlick(&reflectivity, &halfway, incoming);
        let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
        let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
        vec_comp_mul!(self.color * (1.0 - self.metallic), &((one3!() - f1) / PI))
            * f32::abs(dot(&up_normal, incoming))
            + f * d * g / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
                * f32::abs(dot(&up_normal, incoming))
    }

    fn sample_gltfpbr_pdf(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(outgoing + incoming));
        let reflectivity = glm::lerp(
            &eta_to_reflectivity(&vec3(self.ior, self.ior, self.ior)),
            &self.color,
            self.metallic,
        );
        let f = mean3(&fresnel_schlick(&reflectivity, &up_normal, outgoing));
        f * sample_microfacet_pdf(self.roughness, &up_normal, &halfway)
            / (4.0 * f32::abs(dot(outgoing, &halfway)))
            + (1.0 - f) * sample_hemisphere_cos_pdf(&up_normal, incoming)
    }
}

#[inline(always)]
fn reflectivity_to_eta(reflectivity: &Vec3) -> Vec3 {
    let r_clamp = glm::clamp(reflectivity, 0.0, 0.99);
    return vec_comp_div!(
        one3!() + glm::sqrt(&r_clamp),
        &(one3!() - glm::sqrt(&r_clamp))
    );
}

#[inline(always)]
fn eta_to_reflectivity(eta: &Vec3) -> Vec3 {
    let eta_minus = vec_comp_mul!(eta - one3!(), &(eta - one3!()));
    let eta_plus = vec_comp_mul!(eta + one3!(), &(eta + one3!()));
    vec_comp_div!(eta_minus, &eta_plus)
}

fn sample_hemisphere_cos(normal: &Vec3, rn: &Vec2) -> Vec3 {
    let z = f32::sqrt(rn.y);
    let r = f32::sqrt(1.0 - z * z);
    let phi = 2.0 * PI * rn.x;
    let local_direction = vec3(r * f32::cos(phi), r * f32::sin(phi), z);
    transform_direction_mat(&basis_fromz(normal), &local_direction)
}

fn sample_hemisphere_cos_pdf(normal: &Vec3, incoming: &Vec3) -> f32 {
    let cosw = dot(normal, incoming);
    if cosw <= 0.0 {
        0.0
    } else {
        cosw / PI
    }
}

fn fresnel_schlick(specular: &Vec3, normal: &Vec3, outgoing: &Vec3) -> Vec3 {
    if is_null(specular, epsilon()) {
        return zero3!();
    }
    let cosine = dot(normal, outgoing);
    specular + (one3!() - specular) * f32::powf(f32::clamp(1.0 - f32::abs(cosine), 0.0, 1.0), 5.0)
}

fn fresnel_dielectric(eta: f32, normal: &Vec3, outgoing: &Vec3) -> f32 {
    // Implementation from
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    let cosw = f32::abs(dot(normal, outgoing));

    let sin2 = 1.0 - cosw * cosw;
    let eta2 = eta * eta;

    let cos2t = 1.0 - sin2 / eta2;
    if cos2t < 0.0 {
        return 1.0;
    } // tir

    let t0 = f32::sqrt(cos2t);
    let t1 = eta * t0;
    let t2 = eta * cosw;

    let rs = (cosw - t1) / (cosw + t1);
    let rp = (t0 - t2) / (t0 + t2);

    (rs * rs + rp * rp) / 2.0
}

fn fresnel_conductor(eta: &Vec3, etak: &Vec3, normal: &Vec3, outgoing: &Vec3) -> Vec3 {
    // Implementation from
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    let mut cosw = dot(normal, outgoing);
    if cosw <= 0.0 {
        return zero3!();
    }

    cosw = f32::clamp(cosw, -1.0, 1.0);
    let cos2 = cosw * cosw;
    let sin2 = f32::clamp(1.0 - cos2, 0.0, 1.0);
    let eta2 = vec_comp_mul!(eta, eta);
    let etak2 = vec_comp_mul!(etak, etak);

    let t0 = eta2 - etak2 - vec3(-sin2, -sin2, -sin2);
    let a2plusb2 = glm::sqrt(&(vec_comp_mul!(t0, &t0) + 4.0 * vec_comp_mul!(eta2, &etak2)));
    let t1 = a2plusb2 + vec3(cos2, cos2, cos2);
    let a = glm::sqrt(&((a2plusb2 + t0) / 2.0));
    let t2 = 2.0 * a * cosw;
    let rs = vec_comp_div!(t1 - t2, &(t1 + t2));

    let t3 = cos2 * a2plusb2 + vec3(sin2 * sin2, sin2 * sin2, sin2 * sin2);
    let t4 = t2 * sin2;
    let rp = vec_comp_mul!(rs, &vec_comp_div!(t3 - t4, &(t3 + t4)));
    (rp + rs) / 2.0
}

fn microfacet_shadowing(
    roughness: f32,
    normal: &Vec3,
    halfway: &Vec3,
    outgoing: &Vec3,
    incoming: &Vec3,
) -> f32 {
    microfacet_shadowing1(roughness, normal, halfway, outgoing, true)
        * microfacet_shadowing1(roughness, normal, halfway, incoming, true)
}

fn microfacet_shadowing1(
    roughness: f32,
    normal: &Vec3,
    halfway: &Vec3,
    direction: &Vec3,
    ggx: bool,
) -> f32 {
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    // https://github.com/KhronosGroup/glTF/tree/master/specification/2.0#appendix-b-brdf-implementation
    let cosine = dot(normal, direction);
    let cosineh = dot(halfway, direction);
    if cosine * cosineh <= 0.0 {
        return 0.0;
    }
    let roughness2 = roughness * roughness;
    let cosine2 = cosine * cosine;
    if ggx {
        2.0 * f32::abs(cosine)
            / (f32::abs(cosine) + f32::sqrt(cosine2 - roughness2 * cosine2 + roughness2))
    } else {
        let ci = f32::abs(cosine) / (roughness * f32::sqrt(1.0 - cosine2));
        if ci < 1.6 {
            (3.535 * ci + 2.181 * ci * ci) / (1.0 + 2.276 * ci + 2.577 * ci * ci)
        } else {
            1.0
        }
    }
}

fn microfacet_distribution(roughness: f32, normal: &Vec3, halfway: &Vec3, ggx: bool) -> f32 {
    // https://google.github.io/filament/Filament.html#materialsystem/specularbrdf
    // http://graphicrants.blogspot.com/2013/08/specular-brdf-reference.html
    let cosine = dot(normal, halfway);
    if cosine <= 0.0 {
        return 0.0;
    }
    let roughness2 = roughness * roughness;
    let cosine2 = cosine * cosine;
    if ggx {
        roughness2
            / (PI * (cosine2 * roughness2 + 1.0 - cosine2) * (cosine2 * roughness2 + 1.0 - cosine2))
    } else {
        f32::exp((cosine2 - 1.0) / (roughness2 * cosine2)) / (PI * roughness2 * cosine2 * cosine2)
    }
}

fn same_hemisphere(normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> bool {
    dot(normal, outgoing) * dot(normal, incoming) >= 0.0
}

fn sample_microfacet(roughness: f32, normal: &Vec3, rn: &Vec2, ggx: bool) -> Vec3 {
    let phi = 2.0 * PI * rn.x;
    let roughness2 = roughness * roughness;
    let theta = if ggx {
        f32::atan(roughness * f32::sqrt(rn.y / (1.0 - rn.y)))
    } else {
        f32::atan(f32::sqrt(-roughness2 * f32::ln(1.0 - rn.y)))
    };
    let local_half_vector = vec3(
        f32::cos(phi) * f32::sin(theta),
        f32::sin(phi) * f32::sin(theta),
        f32::cos(theta),
    );
    transform_direction_mat(&basis_fromz(&normal), &local_half_vector)
}

fn sample_microfacet_pdf(roughness: f32, normal: &Vec3, halfway: &Vec3) -> f32 {
    let cosine = dot(normal, halfway);
    if cosine < 0.0 {
        0.0
    } else {
        microfacet_distribution(roughness, normal, halfway, true) * cosine
    }
}

#[inline(always)]
pub fn sample_uniform(size: usize, r: f32) -> usize {
    usize::clamp((r * size as f32) as usize, 0, size - 1)
}

#[inline(always)]
pub fn sample_uniform_pdf(size: usize) -> f32 {
    1.0 / size as f32
}

#[inline(always)]
pub fn sample_discrete(cdf: &VecDeque<f32>, r: f32) -> usize {
    let r = f32::clamp(r * cdf.back().unwrap(), 0.0, cdf.back().unwrap() - 0.00001);
    let idx = cdf.partition_point(|&n| n <= r);
    usize::clamp(idx, 0, cdf.len() - 1)
}

pub fn sample_discrete_pdf(cdf: &VecDeque<f32>, idx: usize) -> f32 {
    if idx == 0 {
        cdf[0]
    } else {
        cdf[idx] - cdf[idx - 1]
    }
}

#[inline(always)]
pub fn sample_triangle(ruv: &Vec2) -> Vec2 {
    vec2(1.0 - f32::sqrt(ruv.x), ruv.y * f32::sqrt(ruv.x))
}

#[inline(always)]
pub fn sample_sphere(ruv: &Vec2) -> Vec3 {
    let z = 2.0 * ruv.y - 1.0;
    let r = f32::sqrt(f32::clamp(1.0 - z * z, 0.0, 1.0));
    let phi = 2.0 * PI * ruv.x;
    vec3(r * f32::cos(phi), r * f32::sin(phi), z)
}
