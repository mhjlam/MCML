#include "tracer.hpp"

#include <limits>
#include <numbers>

#include "random.hpp"


Tracer::Tracer(RunParams& params, std::shared_ptr<Random>& random) : 
    m_params{ params }, m_random { random }, m_radiance{}
{
    m_radiance.R_spec = 0.0;
    m_radiance.Rb_total = 0.0;
    m_radiance.R_total = 0.0;
    m_radiance.T_total = 0.0;
    m_radiance.Tb_total = 0.0;
    m_radiance.A_total = 0.0;

    m_radiance.Rb_error = 0.0;
    m_radiance.R_error = 0.0;
    m_radiance.T_error = 0.0;
    m_radiance.Tb_error = 0.0;
    m_radiance.A_error = 0.0;

    auto alloc3 = [](std::size_t x, std::size_t y, std::size_t z) {
        return vec3<double>(x, vec2<double>(y, vec1<double>(z, 0.0)));
    };

    auto alloc2 = [](std::size_t x, std::size_t y) {
        return vec2<double>(x, vec1<double>(y, 0.0));
    };

    auto alloc1 = [](std::size_t x) {
        return vec1<double>(x, 0.0);
    };

    std::size_t nz = m_params.grid.num_z;
    std::size_t nr = m_params.grid.num_r;
    std::size_t na = m_params.grid.num_a;
    std::size_t nt = m_params.grid.num_t;

    // Reflectance
    if (m_params.record.R_rat) { m_radiance.R_rat = alloc3(nr, na, nt); }
    if (m_params.record.R_ra) { m_radiance.R_ra = alloc2(nr, na); }
    if (m_params.record.R_rt) { m_radiance.R_rt = alloc2(nr, nt); }
    if (m_params.record.R_at) { m_radiance.R_at = alloc2(na, nt); }
    if (m_params.record.R_r) { m_radiance.R_r = alloc1(nr); }
    if (m_params.record.R_a) { m_radiance.R_a = alloc1(na); }
    if (m_params.record.R_t) { m_radiance.R_t = alloc1(nt); }

    // Transmittance
    if (m_params.record.T_rat) { m_radiance.T_rat = alloc3(nr, na, nt); }
    if (m_params.record.T_ra) { m_radiance.T_ra = alloc2(nr, na); }
    if (m_params.record.T_rt) { m_radiance.T_rt = alloc2(nr, nt); }
    if (m_params.record.T_at) { m_radiance.T_at = alloc2(na, nt); }
    if (m_params.record.T_r) { m_radiance.T_r = alloc1(nr); }
    if (m_params.record.T_a) { m_radiance.T_a = alloc1(na); }
    if (m_params.record.T_t) { m_radiance.T_t = alloc1(nt); }

    // Absorption
    if (m_params.record.A_rzt) { m_radiance.A_rzt = alloc3(nr, nz, nt); m_radiance.Ab_zt = alloc2(nz, nt); }
    if (m_params.record.A_rz) { m_radiance.A_rz = alloc2(nr, nz); m_radiance.Ab_z = alloc1(nz); }
    if (m_params.record.A_zt) { m_radiance.A_zt = alloc2(nz, nt); }
    if (m_params.record.A_z) { m_radiance.A_z = alloc1(nz); }
    if (m_params.record.A_t) { m_radiance.A_t = alloc1(nt); }
}

Tracer::Tracer(RunParams& params, std::shared_ptr<Random>& random, Radiance radiance) :
    m_params{ params }, m_random{ random }, m_radiance{ radiance }
{

}

void Tracer::Spin(Photon& photon, double g)
{
    // Cosine and sine of Ψ
    double ux = photon.direction.x;
    double uy = photon.direction.y;
    double uz = photon.direction.z;

    // Sample θ
    // Generate a random variable based on the Henyey-Greenstein distribution
    double theta = (1 - g * g) / (1 - g + 2 * g * m_random->next());
    double cos_theta = (g == 0.0) ? 2 * m_random->next() - 1 : (1 + g * g - theta * theta) / (2 * g);
    double sin_theta = std::sqrt(1.0 - cos_theta * cos_theta);

    // Sample Ψ
    double psi = 2.0 * std::numbers::pi * m_random->next();
    double cos_psi = std::cos(psi);
    double sin_psi = (psi < std::numbers::pi) ? std::sqrt(1.0 - cos_psi * cos_psi) : -std::sqrt(1.0 - cos_psi * cos_psi);

    // Update directional cosines

    // Close to perpendicular
    if (1.0 - std::abs(uz) <= COS_0_TOLERANCE) {
        photon.direction.x = sin_theta * cos_psi;
        photon.direction.y = sin_theta * sin_psi;
        photon.direction.z = cos_theta * std::copysign(1.0, uz);
    }
    else {
        double temp = std::sqrt(1.0 - uz * uz);
        photon.direction.x = sin_theta * (ux * uz * cos_psi - uy * sin_psi) / temp + ux * cos_theta;
        photon.direction.y = sin_theta * (uy * uz * cos_psi + ux * sin_psi) / temp + uy * cos_theta;
        photon.direction.z = -sin_theta * cos_psi * temp + uz * cos_theta;
    }
}

Photon Tracer::Launch()
{
    Photon photon{
        .alive = true,
        .num_scatters = 0,
        .current_layer = (m_params.source.layer_index != 0) ? m_params.source.layer_index : 1,
        .position = {0.0, 0.0, m_params.source.z},
        .direction = {0.0, 0.0, 1.0},
        .weight = 1.0 - m_radiance.R_spec,
        .step_size = 0,
        .step_size_left = 0,
        .flight_time = 0,
        .R_i = 0.0,
        .Rb_i = 0.0,
        .T_i = 0.0,
        .Tb_i = 0.0,
        .A_i = 0.0
    };

    if (m_params.source.beam == BeamType::Isotropic) {
        Layer layer = m_params.layers[photon.current_layer];

        // To avoid scoring into reflectance or transmittance
        photon.num_scatters++;

        // Isotropically scatter the photon
        Spin(photon, 0.0);

        // Glass layer.
        if (layer.mu_a == 0.0 && layer.mu_s == 0.0) {
            // Total internal reflection
            if (std::abs(photon.direction.z) <= layer.cos_theta_c0 &&
                std::abs(photon.direction.z) <= layer.cos_theta_c1) {
                photon.alive = 0;
            }
        }
    }
    return photon;
}

void Tracer::Hop(Photon& photon, double dist, double n)
{
    photon.position.x += dist * photon.direction.x;
    photon.position.y += dist * photon.direction.y;
    photon.position.z += dist * photon.direction.z;
    photon.flight_time += dist * n * SPEED_OF_LIGHT_INV;
}

void Tracer::Drop(Photon& photon)
{
    // Absorbed weight
    double x = photon.position.x;
    double y = photon.position.y;

    std::size_t layer = photon.current_layer;

    // Update photon weight
    double mua = m_params.layers[layer].mu_a;
    double mus = m_params.layers[layer].mu_s;
    double dwa = photon.weight * mua / (mua + mus);
    photon.weight -= dwa;

    // Compute array indices
    std::size_t iz, ir, it;

    if (m_params.record.A_rzt || m_params.record.A_zt || m_params.record.A_z || m_params.record.A_rz) {
        if (photon.position.z >= m_params.grid.max_z) {
            iz = m_params.grid.num_z - 1;
        }
        else {
            iz = static_cast<short>(photon.position.z / m_params.grid.step_z);
        }
    }

    if (m_params.record.A_rzt || m_params.record.A_zt || m_params.record.A_t) {
        if (photon.flight_time >= m_params.grid.max_t) {
            it = m_params.grid.num_t - 1;
        }
        else {
            it = static_cast<short>(std::floor(photon.flight_time / m_params.grid.step_t));
        }
    }

    // Assign dwa to an absorption array element

    // Scattered
    if (photon.num_scatters) {
        if (m_params.record.A_rzt || m_params.record.A_rz) {
            double temp = std::sqrt(x * x + y * y);
            if (temp >= m_params.grid.max_r) {
                ir = m_params.grid.num_r - 1;
            }
            else {
                ir = static_cast<short>(temp / m_params.grid.step_r);
            }
        }
        if (m_params.record.A_rzt) {
            m_radiance.A_rzt[ir][iz][it] += dwa;
        }
        if (m_params.record.A_rz) {
            m_radiance.A_rz[ir][iz] += dwa;
        }
    }

    // Ballistic
    else {
        if (m_params.record.A_rzt) {
            m_radiance.Ab_zt[iz][it] += dwa;
        }
        if (m_params.record.A_rz) {
            m_radiance.Ab_z[iz] += dwa;
        }
    }

    if (m_params.record.A_zt) {
        m_radiance.A_zt[iz][it] += dwa;
    }
    if (m_params.record.A_z) {
        m_radiance.A_z[iz] += dwa;
    }

    if (m_params.record.A_t) {
        m_radiance.A_t[it] += dwa;
    }
    photon.A_i += dwa;
}

void Tracer::Roulette(Photon& photon)
{
    // Already dead
    if (photon.weight == 0.0) {
        photon.alive = 0;
    }

    // Survived the roulette
    else if (m_random->next() < ROULETTE_SURVIVAL) {
        photon.weight /= ROULETTE_SURVIVAL;
    }
    else {
        photon.alive = 0;
    }
}

double Tracer::Fresnel(double eta_i, double eta_t, double cos_ai, double& cos_at)
{
    // Matched boundary
    if (eta_i == eta_t) {
        cos_at = cos_ai;
        return 0.0;
    }
    // Normal incidence
    else if (1.0 - cos_ai <= COS_0_TOLERANCE) {
        cos_at = cos_ai;
        double r = (eta_t - eta_i) / (eta_t + eta_i);
        return r * r;
    }
    // Very slant incidence
    else if (cos_ai < COS_90_TOLERANCE) {
        cos_at = 0.0;
        return 1.0;
    }
    // General incidence
    else {
        // Sine of the angles ai & at
        double sin_ai = std::sqrt(1 - cos_ai * cos_ai);
        double sin_at = eta_i * sin_ai / eta_t;

        // Total internal reflection
        if (sin_at >= 1.0) {
            cos_at = 0.0;
            return 1.0;
        }

        // Cosine of at
        cos_at = std::sqrt(1 - sin_at * sin_at);

        // Cosines of (ai + at) and (ai - at)
        double cos_ap = cos_ai * cos_at - sin_ai * sin_at;  // c+ = cc - ss.
        double cos_am = cos_ai * cos_at + sin_ai * sin_at;  // c- = cc + ss.

        // Sines of (ai + at) and (ai - at)
        double sin_ap = sin_ai * cos_at + cos_ai * sin_at;	// s+ = sc + cs.
        double sin_am = sin_ai * cos_at - cos_ai * sin_at;	// s- = sc - cs.

        // Arranged for speed
        return (0.5 * sin_am * sin_am) *
            (cos_am * cos_am + cos_ap * cos_ap) /
            (sin_ap * sin_ap * cos_am * cos_am);
    }
    return 0.0;
}

void Tracer::RecordReflectance(Photon& photon, double reflectance)
{
    double x = photon.position.x;
    double y = photon.position.y;

    // Index to r & angle
    std::size_t ir, ia, it;

    if (m_params.record.R_rat || m_params.record.R_at || m_params.record.R_rt || m_params.record.R_t) {
        if (photon.flight_time >= m_params.grid.max_t) {
            it = m_params.grid.num_t - 1;
        }
        else {
            it = static_cast<short>(std::floor(photon.flight_time / m_params.grid.step_t));
        }
    }

    // Scattered
    if (photon.num_scatters) {
        if (m_params.record.R_rat || m_params.record.R_rt || m_params.record.R_ra || m_params.record.R_r) {
            double temp = std::sqrt(x * x + y * y);
            if (temp >= m_params.grid.max_r) {
                ir = m_params.grid.num_r - 1;
            }
            else {
                ir = static_cast<short>(temp / m_params.grid.step_r);
            }
        }
        if (m_params.record.R_rat || m_params.record.R_at || m_params.record.R_ra || m_params.record.R_a) {
            if ((ia = static_cast<short>(std::acos(-photon.direction.z) / m_params.grid.step_a) > m_params.grid.num_a - 1)) {
                ia = m_params.grid.num_a - 1;
            }
        }

        // Assign photon weight to the reflection array element
        if (m_params.record.R_rat) {
            m_radiance.R_rat[ir][ia][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.R_ra) {
            m_radiance.R_ra[ir][ia] += photon.weight * (1.0 - reflectance);
        }

        if (m_params.record.R_rt) {
            m_radiance.R_rt[ir][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.R_r) {
            m_radiance.R_r[ir] += photon.weight * (1.0 - reflectance);
        }

        if (m_params.record.R_at) {
            m_radiance.R_at[ia][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.R_a) {
            m_radiance.R_a[ia] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.R_t) {
            m_radiance.R_t[it] += photon.weight * (1.0 - reflectance);
        }
        photon.R_i += photon.weight * (1.0 - reflectance);
    }

    // Ballistic reflection
    else {
        photon.Rb_i += photon.weight * (1.0 - reflectance);
    }

    photon.weight *= reflectance;
}

void Tracer::RecordTransmittance(Photon& photon, double reflectance)
{
    double x = photon.position.x;
    double y = photon.position.y;

    // Index to r & angle
    std::size_t ir, ia, it;
    if (m_params.record.T_rat || m_params.record.T_at || m_params.record.T_rt || m_params.record.T_t) {
        if (photon.flight_time >= m_params.grid.max_t) {
            it = m_params.grid.num_t - 1;
        }
        else {
            it = static_cast<short>(std::floor(photon.flight_time / m_params.grid.step_t));
        }
    }

    // Scattered
    if (photon.num_scatters) {
        if (m_params.record.T_rat || m_params.record.T_rt || m_params.record.T_ra || m_params.record.T_r) {
            double temp = std::sqrt(x * x + y * y);
            if (temp >= m_params.grid.max_r) {
                ir = m_params.grid.num_r - 1;
            }
            else {
                ir = (short)(temp / m_params.grid.step_r);
            }
        }
        if (m_params.record.T_rat || m_params.record.T_at || m_params.record.T_ra || m_params.record.T_a) {
            if ((ia = static_cast<short>(std::acos(photon.direction.z) / m_params.grid.step_a) > m_params.grid.num_a - 1)) {
                ia = m_params.grid.num_a - 1;
            }
        }

        // Assign photon weight to the transmittance array element
        if (m_params.record.T_rat) {
            m_radiance.T_rat[ir][ia][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.T_ra) {
            m_radiance.T_ra[ir][ia] += photon.weight * (1.0 - reflectance);
        }

        if (m_params.record.T_rt) {
            m_radiance.T_rt[ir][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.T_r) {
            m_radiance.T_r[ir] += photon.weight * (1.0 - reflectance);
        }

        if (m_params.record.T_at) {
            m_radiance.T_at[ia][it] += photon.weight * (1.0 - reflectance);
        }
        if (m_params.record.T_a) {
            m_radiance.T_a[ia] += photon.weight * (1.0 - reflectance);
        }

        if (m_params.record.T_t) {
            m_radiance.T_t[it] += photon.weight * (1.0 - reflectance);
        }
        photon.T_i += photon.weight * (1.0 - reflectance);
    }

    // Collimated
    else {
        photon.Tb_i += photon.weight * (1.0 - reflectance);
    }

    photon.weight *= reflectance;
}

void Tracer::CrossUp(Photon& photon)
{
    // Z directional cosine
    double uz = photon.direction.z;

    // Cosines of transmission alpha.
    double uz1 = 0.0;

    std::size_t layer = photon.current_layer;
    double eta_i = m_params.layers[layer].eta;
    double eta_t = m_params.layers[layer - 1].eta;

    // Get reflectance; if 1.0, then total internal reflection
    double refl = (-uz <= m_params.layers[layer].cos_theta_c0) ? 1.0 : Fresnel(eta_i, eta_t, -uz, uz1);

    if (PARTIAL_REFLECTION) {
        // Partially transmitted
        if (layer == 1 && refl < 1.0) {
            photon.direction.z = -uz1; // Escaped photon
            RecordReflectance(photon, refl);
            photon.direction.z = -uz; // Reflected photon
        }
        // Transmitted to current_layer - 1
        else if (m_random->next() > refl) {
            photon.current_layer--;
            photon.direction.x *= eta_i / eta_t;
            photon.direction.y *= eta_i / eta_t;
            photon.direction.z = -uz1;
        }
        // Reflected
        else {
            photon.direction.z = -uz;
        }
    }
    else {
        // Transmitted to current_layer - 1
        if (m_random->next() > refl) {
            // Escaped
            if (layer == 1) {
                photon.direction.z = -uz1;
                RecordReflectance(photon, 0.0);
                photon.alive = 0;
            }
            else {
                photon.current_layer--;
                photon.direction.x *= eta_i / eta_t;
                photon.direction.y *= eta_i / eta_t;
                photon.direction.z = -uz1;
            }
        }
        // Reflected
        else {
            photon.direction.z = -uz;
        }
    }
}

void Tracer::CrossDown(Photon& photon)
{
    // Z directional cosine
    double uz = photon.direction.z;

    // Cosines of transmission alpha
    double uz1 = 0.0;

    std::size_t layer = photon.current_layer;
    double eta_i = m_params.layers[layer].eta;
    double eta_t = m_params.layers[layer + 1].eta;

    // Get reflectance; if 1.0, then total internal reflection
    double refl = (uz <= m_params.layers[layer].cos_theta_c1) ? 1.0 : Fresnel(eta_i, eta_t, uz, uz1);

    if (PARTIAL_REFLECTION) {
        if (layer == m_params.num_layers && refl < 1.0) {
            photon.direction.z = uz1;
            RecordTransmittance(photon, refl);
            photon.direction.z = -uz;
        }
        // Transmitted to current_layer + 1
        else if (m_random->next() > refl) {
            photon.current_layer++;
            photon.direction.x *= eta_i / eta_t;
            photon.direction.y *= eta_i / eta_t;
            photon.direction.z = uz1;
        }
        // Reflected
        else {
            photon.direction.z = -uz;
        }
    }
    else {
        // Transmitted to current_layer + 1
        if (m_random->next() > refl) {
            if (layer == m_params.num_layers) {
                photon.direction.z = uz1;
                RecordTransmittance(photon, 0.0);

                // Escaped
                photon.alive = 0;
            }
            else {
                photon.current_layer++;
                photon.direction.x *= eta_i / eta_t;
                photon.direction.y *= eta_i / eta_t;
                photon.direction.z = uz1;
            }
        }
        // Reflected
        else {
            photon.direction.z = -uz;
        }
    }
}

void Tracer::HopDropSpin(Photon& photon)
{
    Layer& layer = m_params.layers[photon.current_layer];
    double mu_t = layer.mu_a + layer.mu_s;

    // Pick a new step size if it is zero, or finish the leftover.
    if (photon.step_size == 0.0) {
        double rnd; // Avoid zero
        while ((rnd = m_random->next()) <= 0.0);
        photon.step_size = -std::log(rnd);
    }

    // Distance to the boundary
    double path = std::numeric_limits<double>::max(); // infinity

    // Path > 0
    if (photon.direction.z > 0.0) {
        path = (m_params.layers[photon.current_layer].z1 - photon.position.z) / photon.direction.z;
    }
    // Path > 0
    else if (photon.direction.z < 0.0) {
        path = (m_params.layers[photon.current_layer].z0 - photon.position.z) / photon.direction.z;
    }

    // Hit boundary
    if (path * mu_t <= photon.step_size) {
        // Move to boundary plane
        Hop(photon, path, layer.eta);

        // Update s
        photon.step_size -= path * mu_t;

        if (photon.direction.z < 0.0) {
            CrossUp(photon);
        }
        else {
            CrossDown(photon);
        }
    }
    // Fit in current_layer
    else {
        Hop(photon, photon.step_size / mu_t, layer.eta);

        // Update s
        photon.step_size = 0;
        Drop(photon);
        Spin(photon, m_params.layers[photon.current_layer].g);
        photon.num_scatters++;
    }
}

void Tracer::Trace(Photon& photon)
{
    do {
        HopDropSpin(photon);
        if (photon.alive && photon.weight < m_params.weight_threshold) {
            Roulette(photon);
        }
    } while (photon.alive);

    m_radiance.A_total += photon.A_i;
    m_radiance.A_error += photon.A_i * photon.A_i;
    
    m_radiance.Tb_total += photon.Tb_i;
    m_radiance.Tb_error += photon.Tb_i * photon.Tb_i;
    m_radiance.T_total += photon.T_i;
    m_radiance.T_error += photon.T_i * photon.T_i;
    
    m_radiance.Rb_total += photon.Rb_i;
    m_radiance.Rb_error += photon.Rb_i * photon.Rb_i;
    m_radiance.R_total += photon.R_i;
    m_radiance.R_error += photon.R_i * photon.R_i;
}
