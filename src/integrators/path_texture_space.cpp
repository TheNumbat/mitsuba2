#include <enoki/stl.h>
#include <mitsuba/core/properties.h>
#include <mitsuba/core/ray.h>
#include <mitsuba/render/bsdf.h>
#include <mitsuba/render/emitter.h>
#include <mitsuba/render/integrator.h>
#include <mitsuba/render/records.h>
#include <random>

NAMESPACE_BEGIN(mitsuba)

/**!

.. _integrator-path:

Path tracer (:monosp:`path`)
-------------------------------------------

.. pluginparameters::

 * - max_depth
   - |int|
   - Specifies the longest path depth in the generated output image (where -1
corresponds to :math:`\infty`). A value of 1 will only render directly visible
light sources. 2 will lead to single-bounce (direct-only) illumination, and so
on. (Default: -1)
 * - rr_depth
   - |int|
   - Specifies the minimum path depth, after which the implementation will start
to use the *russian roulette* path termination criterion. (Default: 5)
 * - hide_emitters
   - |bool|
   - Hide directly visible emitters. (Default: no, i.e. |false|)

This integrator implements a basic path tracer and is a **good default choice**
when there is no strong reason to prefer another method.

To use the path tracer appropriately, it is instructive to know roughly how
it works: its main operation is to trace many light paths using *random walks*
starting from the sensor. A single random walk is shown below, which entails
casting a ray associated with a pixel in the output image and searching for
the first visible intersection. A new direction is then chosen at the
intersection, and the ray-casting step repeats over and over again (until one of
several stopping criteria applies).

.. image:: ../images/integrator_path_figure.png
    :width: 95%
    :align: center

At every intersection, the path tracer tries to create a connection to
the light source in an attempt to find a *complete* path along which
light can flow from the emitter to the sensor. This of course only works
when there is no occluding object between the intersection and the emitter.

This directly translates into a category of scenes where
a path tracer can be expected to produce reasonable results: this is the case
when the emitters are easily "accessible" by the contents of the scene. For
instance, an interior scene that is lit by an area light will be considerably
harder to render when this area light is inside a glass enclosure (which
effectively counts as an occluder).

Like the :ref:`direct <integrator-direct>` plugin, the path tracer internally
relies on multiple importance sampling to combine BSDF and emitter samples. The
main difference in comparison to the former plugin is that it considers light
paths of arbitrary length to compute both direct and indirect illumination.

.. _sec-path-strictnormals:

.. Commented out for now
.. Strict normals
   --------------

.. Triangle meshes often rely on interpolated shading normals
   to suppress the inherently faceted appearance of the underlying geometry.
These "fake" normals are not without problems, however. They can lead to
paradoxical situations where a light ray impinges on an object from a direction
that is classified as "outside" according to the shading normal, and "inside"
according to the true geometric normal.

.. The :paramtype:`strict_normals` parameter specifies the intended behavior
when such cases arise. The default (|false|, i.e. "carry on") gives precedence
to information given by the shading normal and considers such light paths to be
valid. This can theoretically cause light "leaks" through boundaries, but it is
not much of a problem in practice.

.. When set to |true|, the path tracer detects inconsistencies and ignores these
paths. When objects are poorly tesselated, this latter option may cause them to
lose a significant amount of the incident radiation (or, in other words, they
will look dark).

.. note:: This integrator does not handle participating media

 */

template <typename Float, typename Spectrum>
class PathTextureSpaceIntegrator
    : public MonteCarloIntegrator<Float, Spectrum> {
public:
    MTS_IMPORT_BASE(MonteCarloIntegrator, m_max_depth, m_rr_depth)
    MTS_IMPORT_TYPES(Scene, Sampler, Medium, Emitter, EmitterPtr, BSDF, BSDFPtr,
                     Shape)

    PathTextureSpaceIntegrator(const Properties &props) : Base(props) {

        auto objs = props.objects(true);
        for (auto &obj : objs) {
            Shape *shape = dynamic_cast<Shape *>(obj.second.get());
            if (shape) {
                Log(Warn, "PathTextureSpaceIntegrator using shape: %s",
                    obj.second->id());
                sampled_shape = ref<Shape>(shape);
            }
            SamplingIntegrator *secondary =
                dynamic_cast<SamplingIntegrator *>(obj.second.get());
            if (secondary) {
                Log(Warn, "PathTextureSpaceIntegrator using integrator: %s",
                    obj.second->id());
                secondary_integrator = ref<SamplingIntegrator>(secondary);
            }
        }
    }

    DirectionSample3f sample_shape_direction(const RayDifferential3f &ray_,
                                             const Scene *, Sampler *sampler,
                                             const SurfaceInteraction3f &si,
                                             Mask &active) const {
        MTS_MASKED_FUNCTION(ProfilerPhase::SampleEmitterDirection, active);

        RayDifferential3f ray = ray_;
        Point2f sample(sampler->next_2d(active));

        auto ds = sampled_shape->sample_direction(si, sample, active);
        active &= neq(ds.pdf, 0.f);

        return ds;
    }

    std::tuple<Spectrum, Mask, SurfaceInteraction3f>
    get_emissive(const Scene *scene, Sampler *, const RayDifferential3f &ray_,
                 Mask active) const {

        RayDifferential3f ray = ray_;
        Spectrum result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray          = si.is_valid();
        EmitterPtr emitter      = si.emitter(scene);

        // ---------------- Intersection with emitters ----------------

        if (any_or<true>(neq(emitter, nullptr))) {
            result[active] += emitter->eval(si, active);
        }

        return { result, valid_ray, si };
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
        Spectrum throughput(1.f), result(0.f);

        // ---------------------- First intersection ----------------------

        SurfaceInteraction3f si = scene->ray_intersect(ray, active);
        Mask valid_ray          = si.is_valid();
        EmitterPtr emitter      = si.emitter(scene);

        // ---------------- Intersection with emitters ----------------

        for (int depth = 1;; ++depth) {

            if (any_or<true>(neq(emitter, nullptr))) {
                result[active] += throughput * emitter->eval(si, active);
                active &= eq(emitter, nullptr);
            }

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

            BSDFContext ctx;
            BSDFPtr bsdf = si.bsdf(ray);

            Vector3f wo_world(0.f);
            Float strategy = sampler->next_1d();

            Mask emit_mask  = strategy > 0.7f;
            Mask bsdf_mask  = strategy < 0.2f;
            Mask shape_mask = !emit_mask && !bsdf_mask;

            // --------------------- Emitter sampling ---------------------
            {
                Mask active_e          = emit_mask && active;
                auto [ds, emitter_val] = scene->sample_emitter_direction(
                    si, sampler->next_2d(active_e), true, active_e);
                wo_world[emit_mask] = ds.d;
                active &= !active_e | neq(ds.pdf, 0.f);
            }

            // ----------------------- BSDF sampling ----------------------
            {
                Mask active_bsdf = bsdf_mask && active;
                auto [bs, bsdf_val] =
                    bsdf->sample(ctx, si, sampler->next_1d(active_bsdf),
                                 sampler->next_2d(active_bsdf), active_bsdf);
                wo_world[bsdf_mask] = si.to_world(bs.wo);
                active &= !active_bsdf | neq(bs.pdf, 0.f);
            }

            // ----------------------- Shape sampling ----------------------
            {
                Mask active_shape = shape_mask && active;
                auto ds = sample_shape_direction(ray, scene, sampler, si,
                                                 active_shape);
                wo_world[shape_mask] = ds.d;
                active &= !active_shape | neq(ds.pdf, 0.f);
            }

            // ------------------

            RayDifferential3f nee_ray = si.spawn_ray(wo_world);

            SurfaceInteraction3f si2 = scene->ray_intersect(nee_ray, active);
            active &= si2.is_valid();
            auto [incoming, emask] = secondary_integrator->sample(
                scene, sampler, nee_ray, nullptr, nullptr, active);

            active &= emask;

            Vector3f wo_local = si.to_local(wo_world);
            DirectionSample3f ds(si2, si);

            auto bsdf_val = bsdf->eval(ctx, si, wo_local, active);
            bsdf_val      = si.to_world_mueller(bsdf_val, -wo_local, si.wi);

            Float bsdf_pdf    = bsdf->pdf(ctx, si, wo_local, active);
            Float emitter_pdf = scene->pdf_emitter_direction(si, ds);
            Float shape_pdf   = sampled_shape->pdf_direction(si, ds);
            Float pdf = 0.2f * bsdf_pdf + 0.3f * emitter_pdf + 0.5f * shape_pdf;

            active &= neq(pdf, 0.f);
            result[active] += throughput * incoming * bsdf_val / pdf;

            {
                auto [bs, bsdf_val2] =
                    bsdf->sample(ctx, si, sampler->next_1d(active),
                                 sampler->next_2d(active), active);
                active &= neq(bs.pdf, 0.f);
                throughput = throughput * bsdf_val2;
                ray        = si.spawn_ray(si.to_world(bs.wo));
                active &= any(neq(depolarize(throughput), 0.f));
                SurfaceInteraction3f si_bsdf =
                    scene->ray_intersect(ray, active);
                active &= si_bsdf.is_valid();
                si = std::move(si_bsdf);
            }
        }

        return { result, valid_ray };
    }

    //! @}
    // =============================================================

    std::string to_string() const override {
        return tfm::format("PathTextureSpaceIntegrator[\n"
                           "  max_depth = %i,\n"
                           "  rr_depth = %i\n"
                           "]",
                           m_max_depth, m_rr_depth);
    }

    Float mis_weight(Float pdf_a, Float pdf_b) const {
        pdf_a *= pdf_a;
        pdf_b *= pdf_b;
        return select(pdf_a > 0.f, pdf_a / (pdf_a + pdf_b), 0.f);
    }

    MTS_DECLARE_CLASS()

private:
    ref<Shape> sampled_shape;
    ref<SamplingIntegrator> secondary_integrator;
};

MTS_IMPLEMENT_CLASS_VARIANT(PathTextureSpaceIntegrator, MonteCarloIntegrator)
MTS_EXPORT_PLUGIN(PathTextureSpaceIntegrator, "Path Tracer integrator");
NAMESPACE_END(mitsuba)
