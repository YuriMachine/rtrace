use crate::scene_components::MaterialType;
use crate::utils::*;
use glm::{dot, normalize, vec3};
use glm::{Vec2, Vec3};
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
            emission: Vec3::zeros(),
            color: Vec3::zeros(),
            roughness: 0.0,
            metallic: 0.0,
            ior: 1.0,
            density: Vec3::zeros(),
            scattering: Vec3::zeros(),
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
            Vec3::zeros()
        }
    }

    pub fn sample_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32, rn: &Vec2) -> Vec3 {
        match self.m_type {
            MaterialType::Matte => self.sample_matte(normal, outgoing, rn),
            MaterialType::Glossy => self.sample_glossy(normal, outgoing, rnl, rn),
            MaterialType::Reflective => self.sample_reflective(normal, outgoing, rn),
            _ => Vec3::zeros(),
        }
    }

    pub fn eval_bsdfcos(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness == 0.0 {
            return Vec3::zeros();
        }
        match self.m_type {
            MaterialType::Matte => self.eval_matte(normal, outgoing, incoming),
            MaterialType::Glossy => self.eval_glossy(normal, outgoing, incoming),
            MaterialType::Reflective => self.eval_reflective(normal, outgoing, incoming),
            _ => Vec3::zeros(),
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
            _ => 0.0,
        }
    }

    pub fn sample_delta(&self, normal: &Vec3, outgoing: &Vec3, rnl: f32) -> Vec3 {
        if self.roughness != 0.0 {
            return Vec3::zeros();
        }
        match self.m_type {
            MaterialType::Reflective => self.sample_reflective_delta(normal, outgoing),
            _ => Vec3::zeros(),
        }
    }

    pub fn eval_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if self.roughness != 0.0 {
            return Vec3::zeros();
        }
        match self.m_type {
            MaterialType::Reflective => self.eval_reflective_delta(normal, outgoing, incoming),
            _ => Vec3::zeros(),
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
            return Vec3::zeros();
        }
        self.color / PI * dot(normal, incoming).abs()
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
                Vec3::zeros()
            } else {
                incoming
            }
        } else {
            sample_hemisphere_cos(&up_normal, rn)
        }
    }

    fn eval_glossy(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return Vec3::zeros();
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
        self.color * (1.0 - f1) / PI * (dot(&up_normal, incoming).abs())
            + vec3(1.0, 1.0, 1.0) * f * d * g
                / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
                * (dot(&up_normal, incoming).abs())
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
            / (4.0 * (dot(outgoing, &halfway).abs()))
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
            Vec3::zeros()
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
            return Vec3::zeros();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        let halfway = normalize(&(incoming + outgoing));
        let f = fresnel_conductor(
            &reflectivity_to_eta(&self.color),
            &Vec3::zeros(),
            &halfway,
            incoming,
        );
        let d = microfacet_distribution(self.roughness, &up_normal, &halfway, true);
        let g = microfacet_shadowing(self.roughness, &up_normal, &halfway, outgoing, incoming);
        f * d * g / (4.0 * dot(&up_normal, outgoing) * dot(&up_normal, incoming))
            * (dot(&up_normal, incoming).abs())
    }

    fn eval_reflective_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> Vec3 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return Vec3::zeros();
        }
        let up_normal = if dot(normal, outgoing) <= 0.0 {
            -normal
        } else {
            *normal
        };
        fresnel_conductor(
            &reflectivity_to_eta(&self.color),
            &Vec3::zeros(),
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
            / (4.0 * (dot(outgoing, &halfway).abs()));
    }

    fn sample_reflective_pdf_delta(&self, normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> f32 {
        if dot(normal, incoming) * dot(normal, outgoing) <= 0.0 {
            return 0.0;
        }
        1.0
    }
}

fn reflectivity_to_eta(reflectivity: &Vec3) -> Vec3 {
    let r_clamp = glm::clamp(reflectivity, 0.0, 0.99);
    return (vec3(1.0, 1.0, 1.0) + glm::sqrt(&r_clamp))
        .component_div(&(vec3(1.0, 1.0, 1.0) - glm::sqrt(&r_clamp)));
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

fn fresnel_dielectric(eta: f32, normal: &Vec3, outgoing: &Vec3) -> f32 {
    // Implementation from
    // https://seblagarde.wordpress.com/2013/04/29/memo-on-fresnel-equations/
    let cosw = dot(normal, outgoing).abs();

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
        return Vec3::zeros();
    }

    cosw = cosw.clamp(-1.0, 1.0);
    let cos2 = cosw * cosw;
    let sin2 = (1.0 - cos2).clamp(0.0, 1.0);
    let eta2 = eta.component_mul(eta);
    let etak2 = etak.component_mul(etak);

    let t0 = eta2 - etak2.add_scalar(-sin2);
    let a2plusb2 = glm::sqrt(&(t0.component_mul(&t0) + 4.0 * eta2.component_mul(&etak2)));
    let t1 = a2plusb2.add_scalar(cos2);
    let a = glm::sqrt(&((a2plusb2 + t0) / 2.0));
    let t2 = 2.0 * a * cosw;
    let rs = (t1 - t2).component_div(&(t1 + t2));

    let t3 = cos2 * a2plusb2.add_scalar(sin2) * sin2;
    let t4 = t2 * sin2;
    let rp = rs.component_mul(&(t3 - t4).component_div(&(t3 + t4)));
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
        return 2.0 * cosine.abs()
            / (cosine.abs() + f32::sqrt(cosine2 - roughness2 * cosine2 + roughness2));
    } else {
        let ci = cosine.abs() / (roughness * f32::sqrt(1.0 - cosine2));
        if ci < 1.6 {
            return (3.535 * ci + 2.181 * ci * ci) / (1.0 + 2.276 * ci + 2.577 * ci * ci);
        } else {
            return 1.0;
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
        return roughness2
            / (PI
                * (cosine2 * roughness2 + 1.0 - cosine2)
                * (cosine2 * roughness2 + 1.0 - cosine2));
    } else {
        return f32::exp((cosine2 - 1.0) / (roughness2 * cosine2))
            / (PI * roughness2 * cosine2 * cosine2);
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
