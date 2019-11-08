#include <mitsuba/core/properties.h>
#include <mitsuba/core/transform.h>
#include <mitsuba/render/endpoint.h>
#include <mitsuba/render/medium.h>

NAMESPACE_BEGIN(mitsuba)

MTS_VARIANT Endpoint<Float, Spectrum>::Endpoint(const Properties &props) {
    m_world_transform = props.animated_transform("to_world", ScalarTransform4f()).get();
}

MTS_VARIANT Endpoint<Float, Spectrum>::~Endpoint() { }

MTS_VARIANT void Endpoint<Float, Spectrum>::set_scene(const Scene *) {
}

MTS_VARIANT void Endpoint<Float, Spectrum>::set_shape(Shape * shape) {
    m_shape = shape;
}

MTS_VARIANT void Endpoint<Float, Spectrum>::set_medium(Medium *medium) {
    m_medium = medium;
}

MTS_VARIANT std::pair<typename Endpoint<Float, Spectrum>::Ray3f, Spectrum>
Endpoint<Float, Spectrum>::sample_ray(Float /*time*/, Float /*sample1*/,
                                      const Point2f & /*sample2*/,
                                      const Point2f & /*sample3*/,
                                      Mask /*active*/) const {
    NotImplementedError("sample_ray");
}

MTS_VARIANT std::pair<typename Endpoint<Float, Spectrum>::DirectionSample3f, Spectrum>
Endpoint<Float, Spectrum>::sample_direction(const Interaction3f & /*it*/,
                                            const Point2f & /*sample*/,
                                            Mask /*active*/) const {
    NotImplementedError("sample_direction");
}

MTS_VARIANT Float Endpoint<Float, Spectrum>::pdf_direction(const Interaction3f & /*it*/,
                                                           const DirectionSample3f & /*ds*/,
                                                           Mask /*active*/) const {
    NotImplementedError("pdf_direction");
}

MTS_VARIANT Spectrum Endpoint<Float, Spectrum>::eval(const SurfaceInteraction3f & /*si*/,
                                                     Mask /*active*/) const {
    NotImplementedError("eval");
}

MTS_INSTANTIATE_OBJECT(Endpoint)
NAMESPACE_END(mitsuba)
