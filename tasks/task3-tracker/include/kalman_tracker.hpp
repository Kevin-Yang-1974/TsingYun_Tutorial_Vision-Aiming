#pragma once

namespace hw
{
struct Vec3
{
    double x{0.0};
    double y{0.0};
    double z{0.0};
};

struct TrackState
{
    bool tracking{false};
    Vec3 position{};
    Vec3 velocity{};
};

class KalmanTracker
{
public:
    KalmanTracker();

    bool isTracking() const;
    void reset();
    TrackState update(const Vec3& measurement, double dt);
    TrackState predict(double dt);

    void set_process_noise(double q) { process_noise_ = q; }
    void set_measurement_noise(double r) { measurement_noise_ = r; }

private:
    struct AxisFilter
    {
        double position{0.0};
        double velocity{0.0};
        double p00{1.0};
        double p01{0.0};
        double p10{0.0};
        double p11{1.0};

        void reset(double measured_position);
        void predict(double dt, double process_noise);
        void update(double measured_position, double measurement_noise);
        AxisFilter forecast(double dt, double process_noise) const;
    };

    TrackState stateFromFilters() const;

    bool tracking_{false};
    AxisFilter x_{};
    AxisFilter y_{};
    AxisFilter z_{};
    double process_noise_{1.0};
    double measurement_noise_{0.05};
};
}  // namespace hw
