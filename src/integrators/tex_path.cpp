
#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>

NAMESPACE_BEGIN(mitsuba)

template <typename Float, typename Spectrum>
class TexPathIntegrator : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth, m_stop)
    MTS_IMPORT_TYPES(Scene, Sampler, Shape, Medium, Emitter, EmitterPtr, BSDF,
                     BSDFPtr, ImageBlock, ProjectiveCamera, Film)

    ref<Sensor> m_sSensor;
    uint32_t m_samples_per_pass = 0;

    TexPathIntegrator(const Properties &props) : Base(props) {

        auto objs = props.objects(true);
        for (auto &obj : objs) {
            Sensor *sSensor = dynamic_cast<Sensor *>(obj.second.get());
            if (sSensor) {
                Log(Warn, "TexPath using sSensor: %s", obj.second->id());
                m_sSensor = ref<Sensor>(sSensor);
            }
        }
        m_samples_per_pass =
            (uint32_t) props.size_("samples_per_pass", (size_t) -1);
    }

    bool render(Scene *scene, Sensor *sensor) {

        ScopedPhase sp(ProfilerPhase::Render);
        m_stop = false;

        ref<Film> film           = sensor->film();
        ScalarVector2i film_size = film->crop_size();

        size_t total_spp = sensor->sampler()->sample_count();
        size_t samples_per_pass =
            (m_samples_per_pass == (size_t) -1)
                ? total_spp
                : std::min((size_t) m_samples_per_pass, total_spp);
        if ((total_spp % samples_per_pass) != 0)
            Throw("sample_count (%d) must be a multiple of samples_per_pass "
                  "(%d).",
                  total_spp, samples_per_pass);

        size_t n_passes = (total_spp + samples_per_pass - 1) / samples_per_pass;

        std::vector<std::string> channels = aov_names();
        bool has_aovs                     = !channels.empty();

        // Insert default channels and set up the film
        for (size_t i = 0; i < 5; ++i)
            channels.insert(channels.begin() + i, std::string(1, "XYZAW"[i]));
        film->prepare(channels);

        m_render_timer.reset();

        Log(Info, "Start rendering...");

        ref<Sampler> sampler = sensor->sampler();
        sampler->set_samples_per_wavefront((uint32_t) samples_per_pass);

        ScalarFloat diff_scale_factor =
            rsqrt((ScalarFloat) sampler->sample_count());
        ScalarUInt32 wavefront_size =
            hprod(film_size) * (uint32_t) samples_per_pass;
        if (sampler->wavefront_size() != wavefront_size)
            sampler->seed(0, wavefront_size);

        UInt32 idx = arange<UInt32>(wavefront_size);
        if (samples_per_pass != 1)
            idx /= (uint32_t) samples_per_pass;

        ref<ImageBlock> block =
            new ImageBlock(film_size, channels.size(),
                           film->reconstruction_filter(), !has_aovs);
        block->clear();
        block->set_offset(sensor->film()->crop_offset());

        Vector2f pos = Vector2f(Float(idx % uint32_t(film_size[0])),
                                Float(idx / uint32_t(film_size[0])));
        pos += block->offset();

        std::vector<Float> aovs(channels.size());

        for (size_t i = 0; i < n_passes; i++)
            render_sample(scene, sensor, sampler, block, aovs.data(), pos,
                          diff_scale_factor);

        film->put(block);

        if (!m_stop)
            Log(Info, "Rendering finished. (took %s)",
                util::time_string(m_render_timer.value(), true));

        return !m_stop;
    }

    void render_sample(const Scene *scene, Sensor *sensor, Sampler *sampler,
                       ImageBlock *block, Float *aovs, const Vector2f &pos,
                       ScalarFloat diff_scale_factor, Mask active = true) {

        ref<Film> film           = sensor->film();
        ScalarVector2i film_size = film->crop_size();

        // Vector2f position_sample = film_size * sampler->next_2d(active);
        Vector2f position_sample = pos;
        Vector2f adjusted_position =
            (position_sample - sensor->film()->crop_offset()) /
            sensor->film()->crop_size();

        Point2f aperture_sample = sampler->next_2d(active);
        Float wavelength_sample = sampler->next_1d(active);

        auto [ray, ray_weight] = m_sSensor->sample_ray_differential(
            sensor->shutter_open(), wavelength_sample, adjusted_position,
            aperture_sample, active);

        ray.scale_differential(diff_scale_factor);
        Vector3f world_dir = ray.d;

        // Spectrum ray_weight(1.0f);
        // RayDifferential3f ray;
        // {
        //     auto trafo  = sensor->m_world_transform->eval(ray.time, active);
        //     auto Itrafo = trafo.inverse();

        //     ray.time = sensor->shutter_open();
        //     SurfaceInteraction3f pt =
        //         m_shape->eval_parameterization(adjusted_position, active);
        //     Point3f cam_p = Itrafo.transform_affine(pt.p);

        //     Vector3f d  = normalize(Vector3f(cam_p));
        //     Float inv_z = rcp(d.z());
        //     ray.mint    = sensor->near_clip() * inv_z;
        //     ray.maxt    = sensor->far_clip() * inv_z;

        //     ray.o = trafo.transform_affine(Point3f(0.f));
        //     ray.d = trafo * d;
        //     ray.update();

        //     ray.o_x = ray.o_y = ray.o;

        //     ScalarVector3f m_dx =
        //         sensor->m_sample_to_camera *
        //             ScalarPoint3f(1.f / m_resolution.x(), 0.f, 0.f) -
        //         sensor->m_sample_to_camera * ScalarPoint3f(0.f);
        //     ScalarVector3f m_dy =
        //         sensor->m_sample_to_camera *
        //             ScalarPoint3f(0.f, 1.f / m_resolution.y(), 0.f) -
        //         sensor->m_sample_to_camera * ScalarPoint3f(0.f);

        //     ray.d_x               = trafo * normalize(Vector3f(cam_p) +
        //     m_dx); ray.d_y               = trafo * normalize(Vector3f(cam_p)
        //     + m_dy); ray.has_differentials = true;

        //     ray.scale_differential(diff_scale_factor);
        // }

        const Medium *medium = sensor->medium();
        std::pair<Spectrum, Mask> result =
            sample(scene, sampler, ray, medium, aovs + 5, active);
        result.first = ray_weight * result.first;

        UnpolarizedSpectrum spec_u = depolarize(result.first);

        auto trafo       = m_sSensor->m_world_transform->eval(ray.time, active);
        auto Itrafo      = trafo.inverse();
        Vector3f cam_dir = Itrafo * world_dir;
        Float imp        = m_sSensor->importance(cam_dir);

        Float ct = Frame3f::cos_theta(cam_dir), inv_ct = rcp(ct);
        Point2f p(cam_dir.x() * inv_ct, cam_dir.y() * inv_ct);

        Color3f xyz = imp * srgb_to_xyz(spec_u, active);

        aovs[0] = xyz.x();
        aovs[1] = xyz.y();
        aovs[2] = xyz.z();
        aovs[3] = select(result.second, Float(1.f), Float(0.f));
        aovs[4] = 1.f;

        block->put(p, aovs, active);

        sampler->advance();
    }

    std::pair<Spectrum, Mask> sample(const Scene *scene, Sampler *sampler,
                                     const RayDifferential3f &ray_,
                                     const Medium * /* medium */,
                                     Float * /* aovs */,
                                     Mask active) const override {
        MTS_MASKED_FUNCTION(ProfilerPhase::SamplingIntegratorSample, active);

        RayDifferential3f ray = ray_;

        // Tracks radiance scaling due to index of refraction changes
        Float eta(1.f);

        // MIS weight for intersected emitters (set by prev. iteration)
        Float emission_weight(1.f);

        Spectrum throughput(1.f), result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray          = si.is_valid();
        EmitterPtr emitter      = si.emitter(scene);

        for (int depth = 1;; ++depth) {

            // ---------------- Intersection with emitters ----------------

            if (any_or<true>(neq(emitter, nullptr)))
                result[active] +=
                    emission_weight * throughput * emitter->eval(si, active);

            active &= si.is_valid();

            /* Russian roulette: try to keep path weights equal to one,
               while accounting for the solid angle compression at refractive
               index boundaries. Stop with at least some probability to avoid
               getting stuck (e.g. due to total internal reflection) */
            if (depth > m_rr_depth) {
                Float q = min(hmax(depolarize(throughput)) * sqr(eta), .95f);
                active &= sampler->next_1d(active) < q;
                throughput *= rcp(q);
            }

            // Stop if we've exceeded the number of requested bounces, or
            // if there are no more active lanes. Only do this latter check
            // in GPU mode when the number of requested bounces is infinite
            // since it causes a costly synchronization.
            if ((uint32_t) depth >= (uint32_t) m_max_depth ||
                ((!is_cuda_array_v<Float> || m_max_depth < 0) && none(active)))
                break;

            // --------------------- Emitter sampling ---------------------

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray);
            Mask active_e =
                active && has_flag(bsdf->flags(), BSDFFlags::Smooth);

            if (likely(any_or<true>(active_e))) {
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                active_e &= neq(ds.pdf, 0.f);

                // Query the BSDF for that emitter-sampled direction
                Vector3f wo       = si.to_local(ds.d);
                Spectrum bsdf_val = bsdf->eval(ctx, si, wo, active_e);
                bsdf_val          = si.to_world_mueller(bsdf_val, -wo, si.wi);

                // Determine density of sampling that same direction using BSDF
                // sampling
                Float bsdf_pdf = bsdf->pdf(ctx, si, wo, active_e);

                Float mis = select(ds.delta, 1.f, mis_weight(ds.pdf, bsdf_pdf));
                result[active_e] += mis * throughput * bsdf_val * emitter_val;
            }

            // ----------------------- BSDF sampling ----------------------

            // Sample BSDF * cos(theta)
            auto [bs, bsdf_val] =
                bsdf->sample(ctx, si, sampler->next_1d(active),
                             sampler->next_2d(active), active);
            bsdf_val = si.to_world_mueller(bsdf_val, -bs.wo, si.wi);

            throughput = throughput * bsdf_val;
            active &= any(neq(depolarize(throughput), 0.f));
            if (none_or<false>(active))
                break;

            eta *= bs.eta;

            // Intersect the BSDF ray against the scene geometry
            ray                          = si.spawn_ray(si.to_world(bs.wo));
            SurfaceInteraction3f si_bsdf = scene->ray_intersect(ray, active);

            /* Determine probability of having sampled that same
               direction using emitter sampling. */
            emitter = si_bsdf.emitter(scene, active);
            DirectionSample3f ds(si_bsdf, si);
            ds.object = emitter;

            if (any_or<true>(neq(emitter, nullptr))) {
                Float emitter_pdf =
                    select(neq(emitter, nullptr) &&
                               !has_flag(bs.sampled_type, BSDFFlags::Delta),
                           scene->pdf_emitter_direction(si, ds), 0.f);

                emission_weight = mis_weight(bs.pdf, emitter_pdf);
            }

            si = std::move(si_bsdf);
        }

        return { result, valid_ray };
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()
};

MTS_IMPLEMENT_CLASS_VARIANT(TexPathIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(TexPathIntegrator, "Texture-Sampling Integrator");
NAMESPACE_END(mitsuba)
