#include "kalman_tracker.hpp"

#include <exception>
#include <string>

#if defined(_WIN32)
#define HW_TRACKER_EXPORT __declspec(dllexport)
#else
#define HW_TRACKER_EXPORT
#endif

namespace
{
thread_local std::string g_last_error;

void clear_error() { g_last_error.clear(); }
void set_error(const std::string& msg) { g_last_error = msg; }
}  // namespace

extern "C"
{
    HW_TRACKER_EXPORT const char* tracker_last_error()
    {
        return g_last_error.empty() ? nullptr : g_last_error.c_str();
    }

    HW_TRACKER_EXPORT void* tracker_create()
    {
        try
        {
            return new hw::KalmanTracker();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            return nullptr;
        }
    }

    HW_TRACKER_EXPORT void* tracker_create_with_params(double process_noise, double measurement_noise)
    {
        try
        {
            auto* t = new hw::KalmanTracker();
            t->set_process_noise(process_noise);
            t->set_measurement_noise(measurement_noise);
            clear_error();
            return t;
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            return nullptr;
        }
    }

    HW_TRACKER_EXPORT void tracker_destroy(void* tracker)
    {
        delete static_cast<hw::KalmanTracker*>(tracker);
    }

    HW_TRACKER_EXPORT int tracker_is_tracking(void* tracker)
    {
        try
        {
            return static_cast<hw::KalmanTracker*>(tracker)->isTracking() ? 1 : 0;
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            return 0;
        }
    }

    HW_TRACKER_EXPORT void tracker_reset(void* tracker)
    {
        try
        {
            static_cast<hw::KalmanTracker*>(tracker)->reset();
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
        }
    }

    HW_TRACKER_EXPORT void tracker_set_process_noise(void* tracker, double process_noise)
    {
        try
        {
            static_cast<hw::KalmanTracker*>(tracker)->set_process_noise(process_noise);
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
        }
    }

    HW_TRACKER_EXPORT void tracker_set_measurement_noise(void* tracker, double measurement_noise)
    {
        try
        {
            static_cast<hw::KalmanTracker*>(tracker)->set_measurement_noise(measurement_noise);
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
        }
    }

    HW_TRACKER_EXPORT void tracker_get_position(
        void* tracker,
        double* out_x, double* out_y, double* out_z)
    {
        try
        {
            auto* t = static_cast<hw::KalmanTracker*>(tracker);
            hw::TrackState state = t->predict(0.0);
            *out_x = state.position.x;
            *out_y = state.position.y;
            *out_z = state.position.z;
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            *out_x = 0.0;
            *out_y = 0.0;
            *out_z = 0.0;
        }
    }

    HW_TRACKER_EXPORT void tracker_update(
        void* tracker,
        double x, double y, double z, double dt,
        double* out_x, double* out_y, double* out_z)
    {
        try
        {
            hw::Vec3 measurement{x, y, z};
            hw::TrackState state = static_cast<hw::KalmanTracker*>(tracker)->update(measurement, dt);
            *out_x = state.position.x;
            *out_y = state.position.y;
            *out_z = state.position.z;
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            *out_x = 0.0;
            *out_y = 0.0;
            *out_z = 0.0;
        }
    }

    HW_TRACKER_EXPORT void tracker_predict(
        void* tracker, double dt,
        double* out_x, double* out_y, double* out_z)
    {
        try
        {
            hw::TrackState state = static_cast<hw::KalmanTracker*>(tracker)->predict(dt);
            *out_x = state.position.x;
            *out_y = state.position.y;
            *out_z = state.position.z;
            clear_error();
        }
        catch (const std::exception& e)
        {
            set_error(e.what());
            *out_x = 0.0;
            *out_y = 0.0;
            *out_z = 0.0;
        }
    }
}
