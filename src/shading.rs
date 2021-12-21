use crate::utils::*;
use glm::{dot, vec3};
use glm::{Vec2, Vec3};
use std::f32::consts::PI;

pub fn sample_hemisphere_cos(normal: &Vec3, rn: &Vec2) -> Vec3 {
    let z = f32::sqrt(rn.y);
    let r = f32::sqrt(1.0 - z * z);
    let phi = 2.0 * PI * rn.x;
    let local_direction = vec3(r * f32::cos(phi), r * f32::sin(phi), z);
    transform_direction_mat(&basis_fromz(normal), &local_direction)
}

pub fn sample_hemisphere_cos_pdf(normal: &Vec3, incoming: &Vec3) -> f32 {
    let cosw = dot(normal, incoming);
    if cosw <= 0.0 {
        0.0
    } else {
        cosw / PI
    }
}

pub fn fresnel_dielectric(eta: f32, normal: &Vec3, outgoing: &Vec3) -> f32 {
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

pub fn microfacet_shadowing(
    roughness: f32,
    normal: &Vec3,
    halfway: &Vec3,
    outgoing: &Vec3,
    incoming: &Vec3,
) -> f32 {
    microfacet_shadowing1(roughness, normal, halfway, outgoing, true)
        * microfacet_shadowing1(roughness, normal, halfway, incoming, true)
}

pub fn microfacet_shadowing1(
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

pub fn microfacet_distribution(roughness: f32, normal: &Vec3, halfway: &Vec3, ggx: bool) -> f32 {
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

pub fn same_hemisphere(normal: &Vec3, outgoing: &Vec3, incoming: &Vec3) -> bool {
    dot(normal, outgoing) * dot(normal, incoming) >= 0.0
}

pub fn sample_microfacet(roughness: f32, normal: &Vec3, rn: &Vec2, ggx: bool) -> Vec3 {
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

pub fn sample_microfacet_pdf(roughness: f32, normal: &Vec3, halfway: &Vec3) -> f32 {
    let cosine = dot(normal, halfway);
    if cosine < 0.0 {
        0.0
    } else {
        microfacet_distribution(roughness, normal, halfway, true) * cosine
    }
}
